import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax, expit
from sklearn.metrics import accuracy_score, balanced_accuracy_score


PACBM_LOGIT_CLIP = 1e-6
LINEAR_CLIP = 1e-6


def safe_get(data, key):
    return data[key] if key in data.files else None


def logit(x, eps=PACBM_LOGIT_CLIP):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def infer_metadata(path, root):
    """
    Expected path examples:
    recomputed_test_vectors/pacbm/AwA2/efficientnetb0/config/fold1/test_vectors.npz
    recomputed_test_vectors/jointcbm/aPY/mobilenetv2/config/fold3/test_vectors.npz

    Handles extra config subfolders by joining them.
    """
    rel = path.relative_to(root)
    parts = rel.parts

    meta = {
        "model": "unknown",
        "dataset": "unknown",
        "backbone": "unknown",
        "config": "unknown",
        "fold": "unknown",
        "path": str(path),
    }

    if len(parts) >= 4:
        meta["model"] = parts[0]
        meta["dataset"] = parts[1]
        meta["backbone"] = parts[2]

        fold_idx = None
        for i, p in enumerate(parts):
            if p.lower().startswith("fold") or p.lower().startswith("run"):
                fold_idx = i
                break

        if fold_idx is not None:
            meta["config"] = "/".join(parts[3:fold_idx]) if fold_idx > 3 else "unknown"
            meta["fold"] = parts[fold_idx]
        else:
            meta["config"] = "/".join(parts[3:-1]) if len(parts) > 4 else "unknown"

    return meta


def get_original_predictions(y_true, y_pred, class_probs):
    if y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int)
    elif class_probs is not None:
        y_pred = np.argmax(class_probs, axis=1).astype(int)
    else:
        raise ValueError("Missing both y_pred and class_probs.")

    if class_probs is not None:
        probs = np.asarray(class_probs)
    else:
        n = len(y_pred)
        k = int(max(np.max(y_true), np.max(y_pred))) + 1
        probs = np.zeros((n, k), dtype=float)
        probs[np.arange(n), y_pred] = 1.0

    return y_pred, probs


def detect_predictor(data):
    """
    Returns predictor type and parameters.

    PACBM:
        gamma, bias
        logits = logit(concepts) @ gamma + bias

    JointCBM:
        concept_to_class_kernel, concept_to_class_bias
        logits = concepts @ W + b

    KL-CBM:
        If only y_transparent is available, we cannot recompute interventions.
        If concept_to_class_kernel is available, we treat it as a Joint-like concept classifier.
    """
    gamma = safe_get(data, "gamma")
    bias = safe_get(data, "bias")

    if gamma is not None:
        return "pacbm", gamma, bias

    W = safe_get(data, "concept_to_class_kernel")
    b = safe_get(data, "concept_to_class_bias")

    if W is not None:
        return "linear_cbm", W, b

    return "unavailable", None, None


def recompute_probs(concepts, predictor_type, W, b):
    if predictor_type == "pacbm":
        X = logit(concepts, eps=PACBM_LOGIT_CLIP)
        logits = X @ W
    elif predictor_type == "linear_cbm":
        logits = concepts @ W
    else:
        raise ValueError("Cannot recompute predictions for this model.")

    if b is not None:
        logits = logits + b

    return softmax(logits, axis=1)


def contribution_scores(concepts, y_pred, predictor_type, W):
    """
    Contribution to the originally predicted class.

    PACBM:
        contribution = logit(a_m) * gamma[m, predicted_class]

    Linear CBM:
        contribution = a_m * W[m, predicted_class]
    """
    n, m = concepts.shape
    scores = np.zeros((n, m), dtype=float)

    for i in range(n):
        cls = y_pred[i]
        if predictor_type == "pacbm":
            scores[i] = logit(concepts[i], eps=PACBM_LOGIT_CLIP) * W[:, cls]
        else:
            scores[i] = concepts[i] * W[:, cls]

    return scores


def apply_topk_oracle_intervention(
    concepts,
    a_true,
    scores,
    k,
    predictor_type,
    topk_mode="supportive",
    presence_threshold=0.5,
):
    new_concepts = concepts.copy()
    n, m = concepts.shape
    k = min(k, m)

    top_idx = select_topk_indices(
        scores=scores,
        concepts=concepts,
        k=k,
        mode=topk_mode,
        presence_threshold=presence_threshold,
    )

    for i in range(n):
        valid_idx = top_idx[i]
        valid_idx = valid_idx[valid_idx >= 0]

        if len(valid_idx) == 0:
            continue

        values = a_true[i, valid_idx]

        if predictor_type == "pacbm":
            values = np.clip(values, PACBM_LOGIT_CLIP, 1.0 - PACBM_LOGIT_CLIP)

        new_concepts[i, valid_idx] = values

    return new_concepts, top_idx


def apply_random_oracle_intervention(concepts, a_true, k, rng, predictor_type):
    new_concepts = concepts.copy()
    n, m = concepts.shape
    k = min(k, m)

    for i in range(n):
        idx = rng.choice(m, size=k, replace=False)
        values = a_true[i, idx]

        if predictor_type == "pacbm":
            values = np.clip(values, PACBM_LOGIT_CLIP, 1.0 - PACBM_LOGIT_CLIP)

        new_concepts[i, idx] = values

    return new_concepts


def apply_negative_intervention(concepts, scores, correct_mask, k, mode="neutralize"):
    new_concepts = concepts.copy()
    n, m = concepts.shape
    k = min(k, m)

    idx_correct = np.where(correct_mask)[0]

    for i in idx_correct:
        supportive = np.where(scores[i] > 0)[0]

        if len(supportive) > 0:
            ranked = supportive[np.argsort(-scores[i, supportive])]
        else:
            ranked = np.argsort(-scores[i])

        chosen = ranked[:k]

        if mode == "flip":
            new_concepts[i, chosen] = 1.0 - new_concepts[i, chosen]
        else:
            new_concepts[i, chosen] = 0.5

    return new_concepts, idx_correct


def compute_basic_metrics(y_true, y_orig, y_new):
    y_true = np.asarray(y_true).astype(int)

    orig_correct = y_orig == y_true
    new_correct = y_new == y_true

    originally_wrong = ~orig_correct
    originally_correct = orig_correct

    correction = np.mean(new_correct[originally_wrong]) if np.any(originally_wrong) else np.nan
    degradation = np.mean(~new_correct[originally_correct]) if np.any(originally_correct) else np.nan

    return {
        "original_accuracy": accuracy_score(y_true, y_orig),
        "intervened_accuracy": accuracy_score(y_true, y_new),
        "accuracy_change": accuracy_score(y_true, y_new) - accuracy_score(y_true, y_orig),
        "original_balanced_accuracy": balanced_accuracy_score(y_true, y_orig),
        "intervened_balanced_accuracy": balanced_accuracy_score(y_true, y_new),
        "balanced_accuracy_change": (
            balanced_accuracy_score(y_true, y_new)
            - balanced_accuracy_score(y_true, y_orig)
        ),
        "correction_rate": correction,
        "degradation_rate": degradation,
    }


def evaluate_file(path, root, random_repeats, seed, negative_mode):
    rows = []
    meta = infer_metadata(path, root)

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        row = meta.copy()
        row.update({"status": "load_failed", "reason": str(e)})
        return [row]

    y_true = safe_get(data, "y_true")
    y_pred = safe_get(data, "y_pred")
    class_probs = safe_get(data, "class_probs")
    
    concepts = safe_get(data, "concept_probs")
    a_true = safe_get(data, "a_true")

    required = {
        "y_true": y_true,
        "concept_probs": concepts,
        "a_true": a_true,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        row = meta.copy()
        row.update({
            "status": "skipped_missing_arrays",
            "reason": ",".join(missing),
        })
        return [row]

    y_true = np.asarray(y_true).astype(int)
    concepts = np.asarray(concepts).astype(float)
    a_true = np.asarray(a_true).astype(float)

    n_concepts = concepts.shape[1]

    ks_this_file = list(range(1, n_concepts + 1))
    

    if not ks_this_file:
        ks_this_file = [n_concepts]

    missing = [k for k, v in required.items() if v is None]
    if missing:
        row = meta.copy()
        row.update({
            "status": "skipped_missing_arrays",
            "reason": ",".join(missing),
        })
        return [row]



    try:
        y_orig, orig_probs = get_original_predictions(y_true, y_pred, class_probs)
    except Exception as e:
        row = meta.copy()
        row.update({"status": "skipped_missing_predictions", "reason": str(e)})
        return [row]

    predictor_type, W, b = detect_predictor(data)

    if predictor_type == "unavailable":
        row = meta.copy()
        row.update({
            "status": "skipped_no_recomputable_concept_classifier",
            "reason": "Need gamma/bias or concept_to_class_kernel/bias. y_transparent alone is not enough for interventions.",
        })
        return [row]

    W = np.asarray(W).astype(float)
    if b is not None:
        b = np.asarray(b).astype(float)

    rng = np.random.default_rng(seed)
    scores = contribution_scores(concepts, y_orig, predictor_type, W)

    # Original prediction recomputed from saved concept classifier.
    # This is useful because some saved y_pred may come from another head.
    recomputed_probs = recompute_probs(concepts, predictor_type, W, b)
    y_recomputed = np.argmax(recomputed_probs, axis=1)

    base_row = meta.copy()
    base_row.update({
        "status": "ok",
        "predictor_type": predictor_type,
        "intervention": "baseline_recomputed",
        "k": 0,
        "random_repeat": np.nan,
    })
    base_row.update(compute_basic_metrics(y_true, y_orig, y_recomputed))
    rows.append(base_row)

    # 1. Oracle intervention: all concepts
    if predictor_type == "pacbm":
        oracle_concepts = np.clip(a_true.copy(), PACBM_LOGIT_CLIP, 1.0 - PACBM_LOGIT_CLIP)
    else:
        oracle_concepts = a_true.copy()
    oracle_probs = recompute_probs(oracle_concepts, predictor_type, W, b)
    y_oracle = np.argmax(oracle_probs, axis=1)

    row = meta.copy()
    row.update({
        "status": "ok",
        "predictor_type": predictor_type,
        "intervention": "oracle_all",
        "k": concepts.shape[1],
        "random_repeat": np.nan,
    })
    row.update(compute_basic_metrics(y_true, y_orig, y_oracle))
    rows.append(row)

    # 2. Partial top-k interventions with different ranking definitions
    topk_modes = [
        ("partial_topk_oracle", "supportive"),
        ("partial_topk_abs_oracle", "absolute"),
        ("partial_topk_present_supportive_oracle", "present_supportive"),
        ("partial_topk_absent_supportive_oracle", "absent_supportive"),
    ]

    for intervention_name, topk_mode in topk_modes:
        for k in ks_this_file:
            int_concepts, top_idx = apply_topk_oracle_intervention(
                concepts=concepts,
                a_true=a_true,
                scores=scores,
                k=k,
                predictor_type=predictor_type,
                topk_mode=topk_mode,
                presence_threshold=0.5,
            )

            probs = recompute_probs(int_concepts, predictor_type, W, b)
            y_new = np.argmax(probs, axis=1)

            row = meta.copy()
            row.update({
                "status": "ok",
                "predictor_type": predictor_type,
                "intervention": intervention_name,
                "topk_mode": topk_mode,
                "k": k,
                "random_repeat": np.nan,
            })
            row.update(compute_basic_metrics(y_true, y_orig, y_new))
            row.update(topk_diagnostics(concepts, scores, top_idx))
            rows.append(row)

    # 3. Random intervention baseline
    for k in ks_this_file:
        for r in range(random_repeats):
            int_concepts = apply_random_oracle_intervention(concepts, a_true, k, rng, predictor_type)
            probs = recompute_probs(int_concepts, predictor_type, W, b)
            y_new = np.argmax(probs, axis=1)

            row = meta.copy()
            row.update({
                "status": "ok",
                "predictor_type": predictor_type,
                "intervention": "random_oracle",
                "k": k,
                "random_repeat": r,
            })
            row.update(compute_basic_metrics(y_true, y_orig, y_new))
            rows.append(row)

    # 4. Targeted negative intervention
    correct_mask = y_orig == y_true

    for k in ks_this_file:
        neg_concepts, idx_correct = apply_negative_intervention(
            concepts=concepts,
            scores=scores,
            correct_mask=correct_mask,
            k=k,
            mode=negative_mode,
        )

        probs = recompute_probs(neg_concepts, predictor_type, W, b)
        y_new = np.argmax(probs, axis=1)

        if len(idx_correct) > 0:
            orig_pred_class = y_orig[idx_correct]
            p_orig = orig_probs[idx_correct, orig_pred_class]
            p_new = probs[idx_correct, orig_pred_class]
            prob_drop = float(np.mean(p_orig - p_new))

            acc_after_on_correct_subset = accuracy_score(
                y_true[idx_correct],
                y_new[idx_correct],
            )
            acc_drop = 1.0 - acc_after_on_correct_subset
        else:
            prob_drop = np.nan
            acc_drop = np.nan

        row = meta.copy()
        row.update({
            "status": "ok",
            "predictor_type": predictor_type,
            "intervention": f"targeted_negative_{negative_mode}",
            "k": k,
            "random_repeat": np.nan,
            "original_accuracy": accuracy_score(y_true, y_orig),
            "intervened_accuracy": accuracy_score(y_true, y_new),
            "accuracy_change": accuracy_score(y_true, y_new) - accuracy_score(y_true, y_orig),
            "original_balanced_accuracy": balanced_accuracy_score(y_true, y_orig),
            "intervened_balanced_accuracy": balanced_accuracy_score(y_true, y_new),
            "balanced_accuracy_change": (
                balanced_accuracy_score(y_true, y_new)
                - balanced_accuracy_score(y_true, y_orig)
            ),
            "correction_rate": np.nan,
            "degradation_rate": np.nan,
            "accuracy_drop_on_originally_correct": acc_drop,
            "predicted_class_probability_drop": prob_drop,
        })
        rows.append(row)

    return rows


def topk_diagnostics(concepts, scores, top_idx, presence_threshold=0.5):
    selected_concepts_all = []
    selected_scores_all = []
    selected_counts = []

    for i in range(top_idx.shape[0]):
        valid_idx = top_idx[i]
        valid_idx = valid_idx[valid_idx >= 0]

        selected_counts.append(len(valid_idx))

        if len(valid_idx) == 0:
            continue

        selected_concepts_all.append(concepts[i, valid_idx])
        selected_scores_all.append(scores[i, valid_idx])

    if len(selected_concepts_all) == 0:
        return {
            "topk_mean_concept_value": np.nan,
            "topk_frac_present": np.nan,
            "topk_frac_absent": np.nan,
            "topk_mean_score": np.nan,
            "topk_frac_positive_score": np.nan,
            "topk_frac_negative_score": np.nan,
            "topk_effective_k_mean": float(np.mean(selected_counts)),
            "topk_effective_k_std": float(np.std(selected_counts)),
        }

    selected_concepts = np.concatenate(selected_concepts_all)
    selected_scores = np.concatenate(selected_scores_all)

    return {
        "topk_mean_concept_value": float(np.mean(selected_concepts)),
        "topk_frac_present": float(np.mean(selected_concepts >= presence_threshold)),
        "topk_frac_absent": float(np.mean(selected_concepts < presence_threshold)),
        "topk_mean_score": float(np.mean(selected_scores)),
        "topk_frac_positive_score": float(np.mean(selected_scores > 0)),
        "topk_frac_negative_score": float(np.mean(selected_scores < 0)),
        "topk_effective_k_mean": float(np.mean(selected_counts)),
        "topk_effective_k_std": float(np.std(selected_counts)),
    }

def aggregate_results(df):
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "model",
        "dataset",
        "backbone",
        "config",
        "intervention",
        "topk_mode",
        "k",
    ]

    metric_cols = [
        "original_accuracy",
        "intervened_accuracy",
        "accuracy_change",
        "original_balanced_accuracy",
        "intervened_balanced_accuracy",
        "balanced_accuracy_change",
        "correction_rate",
        "degradation_rate",
        "accuracy_drop_on_originally_correct",
        "predicted_class_probability_drop",
        "topk_mean_concept_value",
        "topk_frac_present",
        "topk_frac_absent",
        "topk_mean_score",
        "topk_frac_positive_score",
        "topk_frac_negative_score",
        "topk_effective_k_mean",
        "topk_effective_k_std",
    ]

    existing_metric_cols = [c for c in metric_cols if c in ok.columns]

    agg = ok.groupby(group_cols, dropna=False)[existing_metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    n_folds = ok.groupby(group_cols, dropna=False).size().reset_index(name="n_rows")
    agg = agg.merge(n_folds, on=group_cols, how="left")

    return agg


def select_topk_indices(scores, concepts, k, mode="supportive", presence_threshold=0.5):
    """
    Select up to k concept indices per sample.

    For present_supportive:
        selected concepts must be predicted present AND have positive contribution.

    For absent_supportive:
        selected concepts must be predicted absent AND have positive contribution.

    If fewer than k valid concepts exist, remaining positions are padded with -1.
    The intervention code ignores -1 positions.
    """
    n, m = scores.shape
    k = min(k, m)

    # -1 means: no valid selected concept for this slot
    top_idx = np.full((n, k), -1, dtype=int)

    for i in range(n):
        s = scores[i].copy()

        if mode == "supportive":
            # Standard top-k positive support.
            mask = s > 0
            ranking_score = np.full_like(s, -np.inf)
            ranking_score[mask] = s[mask]

        elif mode == "absolute":
            # Largest magnitude contributions, regardless of sign.
            ranking_score = np.abs(s)

        elif mode == "present_supportive":
            # Present concepts that positively support the predicted class.
            mask = (concepts[i] >= presence_threshold) & (s > 0)
            ranking_score = np.full_like(s, -np.inf)
            ranking_score[mask] = s[mask]

        elif mode == "absent_supportive":
            # Absent concepts whose absence positively supports the predicted class.
            mask = (concepts[i] < presence_threshold) & (s > 0)
            ranking_score = np.full_like(s, -np.inf)
            ranking_score[mask] = s[mask]

        else:
            raise ValueError(f"Unknown top-k mode: {mode}")

        valid = np.where(np.isfinite(ranking_score))[0]

        if len(valid) == 0:
            continue

        ranked_valid = valid[np.argsort(-ranking_score[valid])]
        selected = ranked_valid[:k]

        top_idx[i, :len(selected)] = selected

    return top_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="recomputed_test_vectors_native_20260810")
    parser.add_argument("--output_dir", type=str, default="intervenability_metrics_native")
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Specific k values to evaluate. Ignored if --all_k is set.",
    )
    parser.add_argument(
        "--all_k",
        default=True,
        help="Evaluate all possible k values from 1 to the number of concepts M for each file.",
    )    
    parser.add_argument("--random_repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_mode", type=str, default="neutralize",
                        choices=["neutralize", "flip"])
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("test_vectors.npz"))

    all_rows = []

    if not files:
        print(f"No test_vectors.npz files found under {root}")

    for path in files:
        print(f"[eval] {path}")
        rows = evaluate_file(
            path=path,
            root=root,
            random_repeats=args.random_repeats,
            seed=args.seed,
            negative_mode=args.negative_mode,
        )
        all_rows.extend(rows)

    per_fold = pd.DataFrame(all_rows)

    per_fold_path = output_dir / "intervenability_per_fold.csv"
    agg_path = output_dir / "intervenability_aggregated.csv"

    per_fold.to_csv(per_fold_path, index=False)

    agg = aggregate_results(per_fold)
    agg.to_csv(agg_path, index=False)

    print(f"\nSaved: {per_fold_path}")
    print(f"Saved: {agg_path}")

    if "status" in per_fold.columns:
        print("\nStatus counts:")
        print(per_fold["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()