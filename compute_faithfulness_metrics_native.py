"""
Compute faithfulness metrics from saved test_vectors.npz files only.

Outputs:
- faithfulness_per_fold.csv
- faithfulness_aggregated.csv
- pacbm_stability.csv
- qualitative_top_concepts.csv
"""

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.special import softmax, logit
from scipy.stats import spearmanr

from sklearn.metrics import f1_score, roc_auc_score, homogeneity_score, accuracy_score, balanced_accuracy_score
from sklearn.cluster import KMeans


EPS = 1e-7


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def safe_get(npz, key):
    return npz[key] if key in npz.files else None


def to_labels(y):
    """Convert one-hot/probabilities to integer labels if needed."""
    if y is None:
        return None
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return y.astype(int)


def is_binary_matrix(a):
    """Check whether concept labels are binary."""
    if a is None:
        return False
    vals = np.unique(a[~np.isnan(a)])
    return np.all(np.isin(vals, [0, 1]))


def clip_probs(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def binary_cross_entropy(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = clip_probs(y_prob)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def safe_mean(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def topk_indices(x, k):
    k = min(k, x.shape[-1])
    return np.argsort(-x, axis=-1)[..., :k]


# ---------------------------------------------------------------------
# Path parsing
# Expected pattern:
# recomputed_test_vectors/model/dataset/backbone/config/foldX/test_vectors.npz
# config can contain multiple subfolders.
# ---------------------------------------------------------------------

def infer_metadata(path, root):
    rel = Path(path).relative_to(root)
    parts = rel.parts

    model = parts[0] if len(parts) > 0 else "unknown"
    dataset = parts[1] if len(parts) > 1 else "unknown"
    backbone = parts[2] if len(parts) > 2 else "unknown"

    fold_idx = None
    for i, p in enumerate(parts):
        if p.lower().startswith("fold"):
            fold_idx = i

    if fold_idx is not None:
        fold = parts[fold_idx]
        config_parts = parts[3:fold_idx]
    else:
        fold = Path(path).parent.name
        config_parts = parts[3:-1]

    config = "/".join(config_parts) if config_parts else "unknown"

    return {
        "model": model,
        "dataset": dataset,
        "backbone": backbone,
        "config": config,
        "fold": fold,
        "path": str(path),
    }


# ---------------------------------------------------------------------
# GROUP 1: Real-world faithfulness
# input -> concept
# ---------------------------------------------------------------------

def compute_concept_quality(a_true, concept_probs):
    metrics = {}

    if a_true is None or concept_probs is None:
        return metrics

    a_true = np.asarray(a_true, dtype=float)
    concept_probs = np.asarray(concept_probs, dtype=float)

    if a_true.shape != concept_probs.shape:
        return metrics

    err = a_true - concept_probs
    abs_err = np.abs(err)
    sq_err = err ** 2

    metrics["concept_micro_mse"] = float(np.mean(sq_err))
    metrics["concept_macro_mse"] = float(np.mean(np.mean(sq_err, axis=0)))
    metrics["concept_micro_mae"] = float(np.mean(abs_err))
    metrics["concept_macro_mae"] = float(np.mean(np.mean(abs_err, axis=0)))

    if is_binary_matrix(a_true):
        y_bin = a_true.astype(int)
        p = clip_probs(concept_probs)
        pred_bin = (p >= 0.5).astype(int)

        metrics["concept_bce"] = binary_cross_entropy(y_bin, p)
        metrics["concept_micro_f1"] = float(
            f1_score(y_bin.ravel(), pred_bin.ravel(), average="micro", zero_division=0)
        )

        per_concept_f1 = []
        per_concept_auc = []

        for m in range(y_bin.shape[1]):
            yt = y_bin[:, m]
            yp = pred_bin[:, m]
            ps = p[:, m]

            per_concept_f1.append(
                f1_score(yt, yp, average="binary", zero_division=0)
            )

            if len(np.unique(yt)) == 2:
                try:
                    per_concept_auc.append(roc_auc_score(yt, ps))
                except ValueError:
                    pass

        metrics["concept_macro_f1"] = safe_mean(per_concept_f1)

        if len(np.unique(y_bin.ravel())) == 2:
            try:
                metrics["concept_micro_auc"] = float(
                    roc_auc_score(y_bin.ravel(), p.ravel())
                )
            except ValueError:
                metrics["concept_micro_auc"] = np.nan

        metrics["concept_macro_auc"] = safe_mean(per_concept_auc)

    return metrics


def quantile_bins(x, n_bins=5):
    """Discretize continuous labels into quantile bins."""
    x = np.asarray(x, dtype=float)
    unique = np.unique(x)

    if len(unique) <= 2:
        return x.astype(int)

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))

    if len(edges) <= 2:
        return None

    return np.digitize(x, edges[1:-1], right=True)




def compute_cas_with_cem(a_true, concept_probs, y_true, step=20):
    metrics = {}

    if a_true is None or concept_probs is None or y_true is None:
        return metrics

    c_vec = np.asarray(concept_probs, dtype=float)
    c_test = np.asarray(a_true)
    y_test = to_labels(y_true)

    if c_vec.shape != c_test.shape or y_test is None:
        return metrics

    # Needed for AwA2 continuous concepts
    if not is_binary_matrix(c_test):
        c_test = (c_test >= 0.5).astype(int)
    else:
        c_test = c_test.astype(int)

    try:
        from external_cem_metrics.metrics.cas import concept_alignment_score
        concept_auc, task_auc = concept_alignment_score(
            c_vec=c_vec,
            c_test=c_test,
            y_test=y_test,
            step=step,
            progress_bar=False,
        )

        metrics["cas"] = float(concept_auc)
        metrics["cas_concept_auc"] = float(concept_auc)
        metrics["cas_task_auc"] = float(task_auc)

    except Exception as e:
        metrics["cas_error"] = str(e)

    return metrics


def compute_homogeneity(a_true, concept_probs, random_state=0):
    """
    Simple homogeneity metric from saved concept scores only.

    For each concept:
    - cluster predicted concept scores for that concept
    - compare clusters to ground-truth concept labels
    - average homogeneity over concepts

    This is a practical approximation because saved vectors contain concept
    scores, not full latent concept embeddings.
    """
    metrics = {}

    if a_true is None or concept_probs is None:
        return metrics

    a_true = np.asarray(a_true, dtype=float)
    concept_probs = np.asarray(concept_probs, dtype=float)

    if a_true.shape != concept_probs.shape:
        return metrics

    scores = []

    for m in range(a_true.shape[1]):
        gt = a_true[:, m]
        pred = concept_probs[:, m]

        labels = quantile_bins(gt, n_bins=5)
        if labels is None or len(np.unique(labels)) < 2:
            continue

        n_clusters = min(len(np.unique(labels)), len(np.unique(pred)), 5)
        if n_clusters < 2:
            continue

        try:
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            clusters = km.fit_predict(pred.reshape(-1, 1))
            scores.append(homogeneity_score(labels, clusters))
        except Exception:
            continue

    homogeneity = safe_mean(scores)

    metrics["concept_clustering_homogeneity"] = homogeneity

    return metrics


# ---------------------------------------------------------------------
# GROUP 2: Model faithfulness
# concept -> class
# ---------------------------------------------------------------------

def compute_classification_metrics(y_true, y_pred, class_probs):
    metrics = {}

    y_true_lab = to_labels(y_true)
    y_pred_lab = to_labels(y_pred)

    if y_pred_lab is None and class_probs is not None:
        y_pred_lab = np.argmax(class_probs, axis=1)

    if y_true_lab is not None and y_pred_lab is not None:
        metrics["accuracy"] = float(accuracy_score(y_true_lab, y_pred_lab))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true_lab, y_pred_lab))

    return metrics


def compute_klcbm_alignment(y_transparent, class_probs):
    """
    Compare transparent KL-CBM head with dense class probabilities.
    Requires y_transparent and class_probs to both be probability matrices.
    """
    metrics = {}

    if y_transparent is None or class_probs is None:
        return metrics

    p_t = np.asarray(y_transparent, dtype=float)
    p_d = np.asarray(class_probs, dtype=float)

    if p_t.ndim != 2 or p_d.ndim != 2 or p_t.shape != p_d.shape:
        return metrics

    p_t = clip_probs(p_t)
    p_d = clip_probs(p_d)

    p_t = p_t / p_t.sum(axis=1, keepdims=True)
    p_d = p_d / p_d.sum(axis=1, keepdims=True)

    kl = np.sum(p_t * (np.log(p_t) - np.log(p_d)), axis=1)

    pred_t = np.argmax(p_t, axis=1)
    pred_d = np.argmax(p_d, axis=1)

    top5_t = topk_indices(p_t, 5)
    top5_d = topk_indices(p_d, 5)

    top5_overlap = [
        len(set(top5_t[i]).intersection(set(top5_d[i]))) > 0
        for i in range(p_t.shape[0])
    ]

    metrics["kl_transparent_dense"] = float(np.mean(kl))
    metrics["transparent_dense_top1_agreement"] = float(np.mean(pred_t == pred_d))
    metrics["transparent_dense_prediction_agreement"] = float(np.mean(pred_t == pred_d))
    metrics["transparent_dense_top5_agreement"] = float(np.mean(top5_overlap))

    return metrics


def compute_pacbm_perturbation(
    y_true,
    concept_probs,
    gamma,
    bias,
    k_values=None,
):
    """
    PACBM concept contribution faithfulness.

    contribution_mk(x) = gamma[m, k] * logit(concept_probs[x, m])

    For each sample:
    - use predicted class from PACBM logits
    - rank concepts by positive contribution to that class
    - neutralize top-k concepts by setting logit to 0, i.e. concept prob = 0.5
    - recompute logits and probabilities
    """
    metrics = {}

    if concept_probs is None or gamma is None or bias is None:
        return metrics

    concept_probs = clip_probs(concept_probs)
    gamma = np.asarray(gamma, dtype=float)
    bias = np.asarray(bias, dtype=float).reshape(-1)

    if gamma.ndim != 2:
        return metrics

    n_samples, n_concepts = concept_probs.shape

    if k_values is None:
        # Compute Top-k Concept Neutralization for all k from 0 to M,
        # where M is the number of concepts in the dataset.
        k_values = range(n_concepts + 1)

    if gamma.shape[0] != n_concepts:
        # Try transpose if saved as [classes, concepts]
        if gamma.shape[1] == n_concepts:
            gamma = gamma.T
        else:
            return metrics

    n_classes = gamma.shape[1]

    if bias.shape[0] != n_classes:
        return metrics

    y_true_lab = to_labels(y_true)

    concept_logits = logit(concept_probs)
    logits_orig = concept_logits @ gamma + bias
    probs_orig = softmax(logits_orig, axis=1)

    pred_class = np.argmax(probs_orig, axis=1)
    orig_pred_prob = probs_orig[np.arange(n_samples), pred_class]
    orig_pred_logit = logits_orig[np.arange(n_samples), pred_class]

    if y_true_lab is not None:
        orig_acc = accuracy_score(y_true_lab, pred_class)
    else:
        orig_acc = np.nan

    for k in k_values:
        kk = min(k, n_concepts)
        pert_logits_concepts = concept_logits.copy()

        for i in range(n_samples):
            cls = pred_class[i]
            contrib = gamma[:, cls] * concept_logits[i, :]

            # Top positive supporting concepts
            top_idx = np.argsort(-contrib)[:kk]

            # Neutralize concept evidence: logit(0.5) = 0
            pert_logits_concepts[i, top_idx] = 0.0

        logits_pert = pert_logits_concepts @ gamma + bias
        probs_pert = softmax(logits_pert, axis=1)

        pred_prob_pert = probs_pert[np.arange(n_samples), pred_class]
        pred_logit_pert = logits_pert[np.arange(n_samples), pred_class]

        prob_drop = orig_pred_prob - pred_prob_pert
        logit_drop = orig_pred_logit - pred_logit_pert

        pred_pert = np.argmax(probs_pert, axis=1)

        if y_true_lab is not None:
            pert_acc = accuracy_score(y_true_lab, pred_pert)
            acc_drop = orig_acc - pert_acc
        else:
            acc_drop = np.nan

        prefix = f"pacbm_top{k}_neutralize"

        metrics[f"{prefix}_pred_class_probability_drop"] = float(np.mean(prob_drop))
        metrics[f"{prefix}_average_probability_drop"] = float(np.mean(prob_drop))
        metrics[f"{prefix}_average_logit_drop"] = float(np.mean(logit_drop))
        metrics[f"{prefix}_accuracy_drop"] = float(acc_drop)

    metrics["pacbm_sufficiency_note"] = (
        "built_in_class_probs_computed_only_from_concept_probs"
    )

    return metrics


def compute_kernel_neutralization(
    y_true,
    concept_probs,
    concept_to_class_kernel,
    k_values=None,
    model_prefix="kernel",
):
    """
    Neutralization drops for models whose classifier is:

        logits = concept_probs @ concept_to_class_kernel
        probs = softmax(logits)

    This is valid for KL-CBM because final_output has:
        use_bias=False
    """
    metrics = {}

    if concept_probs is None or concept_to_class_kernel is None:
        return metrics

    concept_probs = np.asarray(concept_probs, dtype=float)
    kernel = np.asarray(concept_to_class_kernel, dtype=float)

    if concept_probs.ndim != 2 or kernel.ndim != 2:
        return metrics

    n_samples, n_concepts = concept_probs.shape

    if k_values is None:
        # Compute Top-k Concept Neutralization for all k from 0 to M,
        # where M is the number of concepts in the dataset.
        k_values = range(n_concepts + 1)

    # Expected shape: [concepts, classes]
    if kernel.shape[0] != n_concepts:
        # Try transpose if saved as [classes, concepts]
        if kernel.shape[1] == n_concepts:
            kernel = kernel.T
        else:
            return metrics

    logits_orig = concept_probs @ kernel
    probs_orig = softmax(logits_orig, axis=1)

    pred_class = np.argmax(probs_orig, axis=1)
    orig_pred_prob = probs_orig[np.arange(n_samples), pred_class]
    orig_pred_logit = logits_orig[np.arange(n_samples), pred_class]

    y_true_lab = to_labels(y_true)

    if y_true_lab is not None:
        orig_acc = accuracy_score(y_true_lab, pred_class)
    else:
        orig_acc = np.nan

    for k in k_values:
        kk = min(k, n_concepts)
        pert_concepts = concept_probs.copy()

        for i in range(n_samples):
            cls = pred_class[i]

            # Contribution to predicted class:
            # contribution_mk(x) = concept_probs[x, m] * kernel[m, cls]
            contrib = concept_probs[i, :] * kernel[:, cls]

            # Top positive supporting concepts
            top_idx = np.argsort(-contrib)[:kk]

            # Neutralize concept probability
            pert_concepts[i, top_idx] = 0.5

        logits_pert = pert_concepts @ kernel
        probs_pert = softmax(logits_pert, axis=1)

        pred_prob_pert = probs_pert[np.arange(n_samples), pred_class]
        pred_logit_pert = logits_pert[np.arange(n_samples), pred_class]

        prob_drop = orig_pred_prob - pred_prob_pert
        logit_drop = orig_pred_logit - pred_logit_pert

        pred_pert = np.argmax(probs_pert, axis=1)

        if y_true_lab is not None:
            pert_acc = accuracy_score(y_true_lab, pred_pert)
            acc_drop = orig_acc - pert_acc
        else:
            acc_drop = np.nan

        prefix = f"{model_prefix}_top{k}_neutralize"

        metrics[f"{prefix}_pred_class_probability_drop"] = float(np.mean(prob_drop))
        metrics[f"{prefix}_average_probability_drop"] = float(np.mean(prob_drop))
        metrics[f"{prefix}_average_logit_drop"] = float(np.mean(logit_drop))
        metrics[f"{prefix}_accuracy_drop"] = float(acc_drop)

    return metrics


# ---------------------------------------------------------------------
# Qualitative CSV
# ---------------------------------------------------------------------

def make_qualitative_rows(meta, y_true, y_pred, class_probs, concept_probs, top_n=10):
    rows = []

    if concept_probs is None:
        return rows

    concept_probs = np.asarray(concept_probs, dtype=float)
    n_samples = concept_probs.shape[0]

    y_true_lab = to_labels(y_true)
    y_pred_lab = to_labels(y_pred)

    if y_pred_lab is None and class_probs is not None:
        y_pred_lab = np.argmax(class_probs, axis=1)

    top_idx = topk_indices(concept_probs, top_n)

    for i in range(n_samples):
        idxs = top_idx[i]
        scores = concept_probs[i, idxs]

        row = dict(meta)
        row.update({
            "sample_index": i,
            "true_class": int(y_true_lab[i]) if y_true_lab is not None else np.nan,
            "predicted_class": int(y_pred_lab[i]) if y_pred_lab is not None else np.nan,
            "top_concepts": ";".join(map(str, idxs.tolist())),
            "top_concept_scores": ";".join([f"{s:.6f}" for s in scores.tolist()]),
        })
        rows.append(row)

    return rows


# ---------------------------------------------------------------------
# PACBM stability across folds
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Concept-to-class weight stability across folds
# ---------------------------------------------------------------------

def jaccard(a, b):
    a, b = set(a), set(b)
    if len(a) == 0 and len(b) == 0:
        return np.nan
    return len(a.intersection(b)) / len(a.union(b))


def safe_spearman_on_indices(w1, w2, idx):
    """
    Spearman correlation between two weight vectors restricted to idx.

    This is useful for explanation stability because it ignores concepts
    that are not part of the top-k explanation-relevant set.
    """
    idx = np.asarray(sorted(set(idx)), dtype=int)

    if len(idx) < 2:
        return np.nan

    v1 = np.asarray(w1[idx], dtype=float)
    v2 = np.asarray(w2[idx], dtype=float)

    # Spearman is undefined if one side is constant.
    if np.allclose(v1, v1[0]) or np.allclose(v2, v2[0]):
        return np.nan

    rho, _ = spearmanr(v1, v2)

    if np.isnan(rho):
        return np.nan

    return float(rho)


def compute_weight_stability(saved_weights):
    """
    Compute stability of concept-to-class matrices across folds.

    Reports:
    - weight_spearman_per_class: Spearman over all concepts
    - top5/top10 positive Jaccard
    - top5/top10 negative Jaccard
    - top5/top10 positive Spearman
    - top5/top10 negative Spearman
    - spearman_at5 / spearman_at10 over top-k absolute weights
    """
    rows = []

    if not saved_weights:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "model": item["model"],
            "dataset": item["dataset"],
            "backbone": item["backbone"],
            "config": item["config"],
            "fold": item["fold"],
            "weight_type": item["weight_type"],
            "idx": idx,
        }
        for idx, item in enumerate(saved_weights)
    ])

    group_cols = ["model", "dataset", "backbone", "config", "weight_type"]

    for keys, group in df.groupby(group_cols):
        
        if 'AwA2' in keys:
            top_ks = np.arange(0,86)
        elif 'aPY' in keys:
            top_ks = np.arange(0,65)
        elif 'CUB' in keys:
            top_ks = np.arange(0,113)

        if len(group) < 2:
            continue

        spearman_vals = []

        # Store values for each k
        metric_lists = {}
        for k in top_ks:
            metric_lists[f"top{k}_positive_jaccard"] = []
            metric_lists[f"top{k}_negative_jaccard"] = []
            metric_lists[f"top{k}_positive_spearman"] = []
            metric_lists[f"top{k}_negative_spearman"] = []
            metric_lists[f"spearman_at{k}"] = []

        for _, r1 in group.iterrows():
            for _, r2 in group.iterrows():
                if r1["idx"] >= r2["idx"]:
                    continue

                w1 = np.asarray(saved_weights[int(r1["idx"])]["weight_matrix"], dtype=float)
                w2 = np.asarray(saved_weights[int(r2["idx"])]["weight_matrix"], dtype=float)

                if w1.shape != w2.shape:
                    continue

                n_concepts, n_classes = w1.shape

                for c in range(n_classes):
                    v1 = w1[:, c]
                    v2 = w2[:, c]

                    # Full/global Spearman over all concepts
                    if not np.allclose(v1, v1[0]) and not np.allclose(v2, v2[0]):
                        rho, _ = spearmanr(v1, v2)
                        if not np.isnan(rho):
                            spearman_vals.append(float(rho))

                    for k in top_ks:
                        kk = min(k, n_concepts)

                        # Top-k positive
                        top_pos_1 = np.argsort(-v1)[:kk]
                        top_pos_2 = np.argsort(-v2)[:kk]
                        top_pos_union = np.union1d(top_pos_1, top_pos_2)

                        metric_lists[f"top{k}_positive_jaccard"].append(
                            jaccard(top_pos_1, top_pos_2)
                        )

                        rho_pos = safe_spearman_on_indices(v1, v2, top_pos_union)
                        if not np.isnan(rho_pos):
                            metric_lists[f"top{k}_positive_spearman"].append(rho_pos)

                        # Top-k negative
                        top_neg_1 = np.argsort(v1)[:kk]
                        top_neg_2 = np.argsort(v2)[:kk]
                        top_neg_union = np.union1d(top_neg_1, top_neg_2)

                        metric_lists[f"top{k}_negative_jaccard"].append(
                            jaccard(top_neg_1, top_neg_2)
                        )

                        rho_neg = safe_spearman_on_indices(v1, v2, top_neg_union)
                        if not np.isnan(rho_neg):
                            metric_lists[f"top{k}_negative_spearman"].append(rho_neg)

                        # Top-k absolute / highest-magnitude concepts
                        top_abs_1 = np.argsort(-np.abs(v1))[:kk]
                        top_abs_2 = np.argsort(-np.abs(v2))[:kk]
                        top_abs_union = np.union1d(top_abs_1, top_abs_2)

                        rho_abs = safe_spearman_on_indices(v1, v2, top_abs_union)
                        if not np.isnan(rho_abs):
                            metric_lists[f"spearman_at{k}"].append(rho_abs)

        row = {
            "model": keys[0],
            "dataset": keys[1],
            "backbone": keys[2],
            "config": keys[3],
            "weight_type": keys[4],
            "n_folds": len(group),
            "weight_spearman_per_class": safe_mean(spearman_vals),
        }

        for metric_name, values in metric_lists.items():
            row[metric_name] = safe_mean(values)

        rows.append(row)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Main per-file computation
# ---------------------------------------------------------------------

def process_file(
    path,
    root,
    qualitative_top_n=10,
    random_repeats=20,
    random_seed=42,
):    
    meta = infer_metadata(path, root)

    npz = np.load(path, allow_pickle=True)

    y_true = safe_get(npz, "y_true")
    y_pred = safe_get(npz, "y_pred")
    class_probs = safe_get(npz, "class_probs")

    a_true = safe_get(npz, "a_true")
    concept_probs = safe_get(npz, "concept_probs")

    y_transparent = safe_get(npz, "y_transparent")

    """if meta["model"] in ["pacbm", "pacbm_2"]:
        gamma = safe_get(npz, "gamma")
        bias = safe_get(npz, "bias")
        class_logits = safe_get(npz, "class_logits")

    else:
        gamma = safe_get(npz, "concept_to_class_kernel")
        bias = safe_get(npz, "concept_to_class_bias")

    concept_to_class_kernel = safe_get(npz, "concept_to_class_kernel")"""

    model_name = meta["model"].lower()

    gamma = None
    bias = None
    class_logits = None

    if model_name in {"pacbm", "pacbm_2"}:
        gamma = safe_get(npz, "gamma")
        bias = safe_get(npz, "bias")
        class_logits = safe_get(npz, "class_logits")

    concept_to_class_kernel = safe_get(
        npz, "concept_to_class_kernel"
    )
    concept_to_class_bias = safe_get(
        npz, "concept_to_class_bias"
    )

    row = dict(meta)

    row.update(compute_classification_metrics(y_true, y_pred, class_probs))
    row.update(compute_concept_quality(a_true, concept_probs))
    #row.update(compute_homogeneity(a_true, concept_probs))
    #row.update(compute_cas_with_cem(a_true, concept_probs,y_true))

    # KL-CBM dense-vs-transparent faithfulness
    row.update(compute_klcbm_alignment(y_transparent, class_probs))

    # PACBM sufficiency / perturbation
    if gamma is not None and bias is not None:
        row.update(
            compute_pacbm_perturbation(
            y_true,
            concept_probs,
            gamma,
            bias,
        )
    )

        row.update(
            compute_random_order_neutralization(
            y_true=y_true,
            concept_probs=concept_probs,
            concept_to_class_weights=gamma,
            bias=bias,
            model_prefix="pacbm",
            use_log_odds=True,
            random_repeats=random_repeats,
            random_seed=random_seed,
        )
    )    

    # KL-CBM / JointCBM dense-head neutralization.
    # For KL-CBM this is valid because final_output uses use_bias=False:
    # logits = concept_probs @ concept_to_class_kernel
    if concept_to_class_kernel is not None:
        row["has_concept_to_class_kernel"] = True

        model_name = meta["model"].lower()

        if "klcbm" in model_name or "kl-cbm" in model_name:
            row.update(
                compute_kernel_neutralization(
                    y_true=y_true,
                    concept_probs=concept_probs,
                    concept_to_class_kernel=concept_to_class_kernel,
                    model_prefix="klcbm_dense",
                )
            )
            row.update(
        compute_random_order_neutralization(
            y_true=y_true,
            concept_probs=concept_probs,
            concept_to_class_weights=concept_to_class_kernel,
            bias=None,
            model_prefix="klcbm_dense",
            use_log_odds=False,
            random_repeats=random_repeats,
            random_seed=random_seed,
        )
    )

        elif "jointcbm" in model_name or "joint_cbm" in model_name or "cbm" in model_name:
            row.update(
                compute_kernel_neutralization(
                    y_true=y_true,
                    concept_probs=concept_probs,
                    concept_to_class_kernel=concept_to_class_kernel,
                    model_prefix="jointcbm",
                )
            )
            row.update(
        compute_random_order_neutralization(
            y_true=y_true,
            concept_probs=concept_probs,
            concept_to_class_weights=concept_to_class_kernel,
            bias=None,
            model_prefix="jointcbm",
            use_log_odds=False,
            random_repeats=random_repeats,
            random_seed=random_seed,
        )
    )
    else:
        row["has_concept_to_class_kernel"] = False

    qualitative_rows = make_qualitative_rows(
        meta=meta,
        y_true=y_true,
        y_pred=y_pred,
        class_probs=class_probs,
        concept_probs=concept_probs,
        top_n=qualitative_top_n,
    )

    stability_items = []

    # PACBM stability: gamma
    if gamma is not None:
        g = np.asarray(gamma, dtype=float)

        # Normalize shape to [concepts, classes]
        if concept_probs is not None and g.ndim == 2:
            n_concepts = concept_probs.shape[1]
            if g.shape[0] != n_concepts and g.shape[1] == n_concepts:
                g = g.T

        item = dict(meta)
        item["weight_type"] = "gamma"
        item["weight_matrix"] = g
        stability_items.append(item)

    # JointCBM / KL-CBM stability: concept_to_class_kernel
    if concept_to_class_kernel is not None:
        w = np.asarray(concept_to_class_kernel, dtype=float)

        # Normalize shape to [concepts, classes]
        if concept_probs is not None and w.ndim == 2:
            n_concepts = concept_probs.shape[1]
            if w.shape[0] != n_concepts and w.shape[1] == n_concepts:
                w = w.T

        item = dict(meta)
        item["weight_type"] = "concept_to_class_kernel"
        item["weight_matrix"] = w
        stability_items.append(item)

    return row, qualitative_rows, stability_items


def aggregate_per_fold(df):
    group_cols = ["model", "dataset", "backbone", "config"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in []]

    agg = df.groupby(group_cols)[numeric_cols].agg(["mean", "std", "count"])
    agg.columns = [
        f"{metric}_{stat}"
        for metric, stat in agg.columns
    ]
    agg = agg.reset_index()

    return agg


def compute_random_order_neutralization(
    y_true,
    concept_probs,
    concept_to_class_weights,
    bias=None,
    k_values=None,
    model_prefix="model",
    use_log_odds=False,
    random_repeats=20,
    random_seed=42,
):
    """
    Random-order concept neutralization baseline.

    For each repeat and sample:
    - randomly order all concepts;
    - progressively neutralize the first k concepts in that order;
    - use the same neutral value as targeted neutralization;
    - average the resulting drops across random repetitions.

    PACBM:
        evidence = logit(concept probability)
        neutral evidence = 0, corresponding to probability 0.5

    JointCBM / KL-CBM:
        evidence = concept probability
        neutral value = 0.5
    """
    metrics = {}

    if concept_probs is None or concept_to_class_weights is None:
        return metrics

    concepts = np.asarray(concept_probs, dtype=float)
    weights = np.asarray(concept_to_class_weights, dtype=float)

    if concepts.ndim != 2 or weights.ndim != 2:
        return metrics

    n_samples, n_concepts = concepts.shape

    # Normalize weight shape to [concepts, classes].
    if weights.shape[0] != n_concepts:
        if weights.shape[1] == n_concepts:
            weights = weights.T
        else:
            return metrics

    n_classes = weights.shape[1]

    if bias is None:
        bias_array = np.zeros(n_classes, dtype=float)
    else:
        bias_array = np.asarray(bias, dtype=float).reshape(-1)

        if len(bias_array) != n_classes:
            return metrics

    if use_log_odds:
        evidence = logit(clip_probs(concepts))
        neutral_value = 0.0
    else:
        evidence = concepts.copy()
        neutral_value = 0.5

    logits_original = evidence @ weights + bias_array
    probabilities_original = softmax(logits_original, axis=1)

    original_prediction = np.argmax(probabilities_original, axis=1)
    sample_indices = np.arange(n_samples)

    original_probability = probabilities_original[
        sample_indices,
        original_prediction,
    ]
    original_logit = logits_original[
        sample_indices,
        original_prediction,
    ]

    y_true_labels = to_labels(y_true)

    if y_true_labels is not None:
        original_accuracy = accuracy_score(
            y_true_labels,
            original_prediction,
        )
    else:
        original_accuracy = np.nan

    if k_values is None:
        requested_k = list(range(n_concepts + 1))
    else:
        requested_k = sorted(
            set(min(int(k), n_concepts) for k in k_values)
        )

    requested_k_set = set(requested_k)
    max_k = max(requested_k)

    stored = {
        k: {
            "probability_drop": [],
            "logit_drop": [],
            "accuracy_drop": [],
        }
        for k in requested_k
    }

    for repeat in range(random_repeats):
        rng = np.random.default_rng(random_seed + repeat)

        # One random ordering per sample. The same ordering is used
        # progressively for all k, so the curve is nested.
        random_order = np.argsort(
            rng.random((n_samples, n_concepts)),
            axis=1,
        )

        current_logits = logits_original.copy()

        for k in range(max_k + 1):
            if k > 0:
                selected_concept = random_order[:, k - 1]

                old_value = evidence[
                    sample_indices,
                    selected_concept,
                ]

                delta = neutral_value - old_value

                current_logits += (
                    delta[:, None]
                    * weights[selected_concept, :]
                )

            if k not in requested_k_set:
                continue

            current_probabilities = softmax(
                current_logits,
                axis=1,
            )

            probability_after = current_probabilities[
                sample_indices,
                original_prediction,
            ]
            logit_after = current_logits[
                sample_indices,
                original_prediction,
            ]

            prediction_after = np.argmax(
                current_probabilities,
                axis=1,
            )

            probability_drop = np.mean(
                original_probability - probability_after
            )
            logit_drop = np.mean(
                original_logit - logit_after
            )

            if y_true_labels is not None:
                accuracy_after = accuracy_score(
                    y_true_labels,
                    prediction_after,
                )
                accuracy_drop = (
                    original_accuracy - accuracy_after
                )
            else:
                accuracy_drop = np.nan

            stored[k]["probability_drop"].append(
                probability_drop
            )
            stored[k]["logit_drop"].append(logit_drop)
            stored[k]["accuracy_drop"].append(accuracy_drop)

    for k in requested_k:
        prefix = (
            f"{model_prefix}_random_top{k}_neutralize"
        )

        probability_drop = np.mean(
            stored[k]["probability_drop"]
        )
        logit_drop = np.mean(
            stored[k]["logit_drop"]
        )
        accuracy_drop = np.mean(
            stored[k]["accuracy_drop"]
        )

        metrics[
            f"{prefix}_pred_class_probability_drop"
        ] = float(probability_drop)

        metrics[
            f"{prefix}_average_probability_drop"
        ] = float(probability_drop)

        metrics[
            f"{prefix}_average_logit_drop"
        ] = float(logit_drop)

        metrics[
            f"{prefix}_accuracy_drop"
        ] = float(accuracy_drop)

    metrics[
        f"{model_prefix}_random_neutralization_repeats"
    ] = int(random_repeats)

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="recomputed_test_vectors_native_20260810",
        help="Root directory containing saved test_vectors.npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="faithfulness_results_native",
        help="Directory where CSV files will be saved.",
    )
    parser.add_argument(
        "--qualitative_top_n",
        type=int,
        default=10,
        help="Number of top predicted concepts to save per image.",
    )

    parser.add_argument(
        "--random_repeats",
        type=int,
        default=20,
        help="Number of random concept orderings used for the baseline.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Base random seed for random-order neutralization.",
    )

    parser.add_argument(
        "--jointcbm_root",
        default="recomputed_test_vectors/jointcbm",
    )

    parser.add_argument(
        "--pacbm_co_root",
        default="recomputed_test_vectors/2/pacbm_2",
    )


    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_root = Path(args.root)

    sources = [
        ("CNN", new_root / "cnn", new_root),
        ("KL-CBM", new_root / "klcbm", new_root),
        ("PACBM-Cl", new_root / "pacbm", new_root),
        (
            "JointCBM",
            Path(args.jointcbm_root),
            Path(args.jointcbm_root).parent,
        ),
        (
            "PACBM-Co",
            Path(args.pacbm_co_root),
            Path(args.pacbm_co_root).parent,
        ),
    ]

    jobs = []

    for label, source_dir, metadata_root in sources:
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing {label} directory: {source_dir}")

        source_paths = sorted(source_dir.rglob("test_vectors.npz"))
        print(f"{label}: found {len(source_paths)} files")

        jobs.extend(
            (path, metadata_root)
            for path in source_paths
        )

    print(f"Total: {len(jobs)} test-vector files")
    #print(f"Found {len(paths)} test_vectors.npz files under {root}")

    per_fold_rows = []
    qualitative_rows_all = []
    stability_items_all = []
    
    for path, metadata_root in jobs:
        #if "fold1" in str(path) and "mobilenetv2" in str(path) and "CUB" in str(path):
        print(f"[PROCESS] {path}")

        try:
            row, qualitative_rows, stability_items = process_file(
                path,
                root=metadata_root,
                qualitative_top_n=args.qualitative_top_n,
                random_repeats=args.random_repeats,
                random_seed=args.random_seed,
            )

            per_fold_rows.append(row)
            qualitative_rows_all.extend(qualitative_rows)
            stability_items_all.extend(stability_items)

        except Exception as e:
            print(f"[SKIP] {path} because of error: {e}")

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_path = output_dir / "faithfulness_per_fold.csv"
    per_fold_df.to_csv(per_fold_path, index=False)

    if len(per_fold_df) > 0:
        agg_df = aggregate_per_fold(per_fold_df)
    else:
        agg_df = pd.DataFrame()

    agg_path = output_dir / "faithfulness_aggregated.csv"
    agg_df.to_csv(agg_path, index=False)

    stability_df = compute_weight_stability(stability_items_all)
    stability_path = output_dir / "concept_weight_stability.csv"
    stability_df.to_csv(stability_path, index=False)

    qualitative_df = pd.DataFrame(qualitative_rows_all)
    qualitative_path = output_dir / "qualitative_top_concepts.csv"
    qualitative_df.to_csv(qualitative_path, index=False)

    print("\nSaved:")
    print(f"- {per_fold_path}")
    print(f"- {agg_path}")
    print(f"- {stability_path}")
    print(f"- {qualitative_path}")


if __name__ == "__main__":
    main()