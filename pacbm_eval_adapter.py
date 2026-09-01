"""
pacbm_eval_adapter.py

Adapter for extract_pacbm_results.py.

This version fixes two things:

1. Legacy PACBM HDF5 checkpoint loading:
   Keras 3 may not correctly restore these .weights.h5 files with plain
   model.load_weights(...). We manually assign HDF5 variables by compatible
   path/name and verify gamma_mk.

2. Data loading / preprocessing mismatch:
   The adapter reconstructs the train/val/test split, then tries multiple
   evaluation preprocessing modes on the validation set. It selects the mode
   whose recomputed validation balanced accuracy best matches the saved
   history.csv best validation balanced accuracy. The selected mode is then
   used for the test set.
"""

from __future__ import annotations

import os
import re
import h5py
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from data_loaders.data_loader import get_dataset_loaders
from data_loaders.configs import dataset_config
from models.pacbm.train_model import (
    extract_pil_augs,
    apply_pil_augs_np,
    compute_P_true_global,
    build_and_compile,
)
from models.pacbm.utils import make_tfds, get_backbone_preprocess, numpy_from_dl, set_seeds

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
_CACHE: Dict[tuple, Dict[str, Any]] = {}

# Modes are intentionally broad because the old training pipeline mixed:
# PIL augment outputs, dataset-loader tensors, and backbone-specific preprocessing.
PREPROCESS_MODES = [
    "backbone_preprocess_255",
    "identity_255",
    "identity_0_1",
    "minus1_1",
]


def _norm_dataset(x: str) -> str:
    s = str(x)
    if s.lower() == "apy":
        return "aPY"
    if s.lower() in {"awa", "awa2"}:
        return "AwA2"
    if s.lower() == "cub":
        return "CUB"
    return s


def _get_row(run) -> Dict[str, Any]:
    return dict(getattr(run, "tracker_row", None) or {})


def _safe_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def _path_tokens(
    run_dir: Path,
) -> Tuple[
    Optional[Tuple[int, int, str, float]],
    Optional[Tuple[float, float, float]],
    Optional[Tuple[float, float, float]],
]:
    parts = list(Path(run_dir).parts)
    main = coeffs = dyn = None

    for p in parts:
        m = re.match(r"^(\d+)_(\d+)_([A-Za-z0-9]+)_([0-9.eE+-]+)$", p)
        if m:
            main = (int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4)))

        nums = p.split("_")
        if len(nums) == 3:
            try:
                vals = tuple(float(z) for z in nums)
            except Exception:
                continue
            if any(v >= 1.0 for v in vals) and coeffs is None:
                coeffs = vals
            elif all(v < 1.0 for v in vals) or vals[-1] <= 1.0:
                dyn = vals

    return main, coeffs, dyn


def _hparams(run) -> Dict[str, Any]:
    row = _get_row(run)
    main, coeffs, dyn = _path_tokens(Path(run.run_dir))

    epochs = _safe_int(row.get("epochs"), main[0] if main else 300)
    batch_size = _safe_int(row.get("batch_size"), main[1] if main else 64)
    optimizer = str(row.get("optimizer", main[2] if main else "adam")).strip()
    lr = _safe_float(row.get("lr"), main[3] if main else 1e-4)

    coeff_l_a_CE = _safe_float(row.get("coeff_l_a_CE"), coeffs[0] if coeffs else 1.0)
    coeff_l_cls_CE = _safe_float(row.get("coeff_l_cls_CE"), coeffs[1] if coeffs else 1.0)
    coeff_prior_anch = _safe_float(row.get("coeff_prior_anch"), coeffs[2] if coeffs else 1.0)

    start_score = row.get("start_mse", row.get("start_bce", None))
    end_score = row.get("end_mse", row.get("end_bce", None))
    start_mse = _safe_float(start_score, dyn[0] if dyn else 0.17)
    end_mse = _safe_float(end_score, dyn[1] if dyn else 0.07)
    ema_momentum = _safe_float(row.get("ema_momentum"), dyn[2] if dyn else 0.9)

    use_ema = str(row.get("use_ema_prior", "True")).lower() not in {"false", "0", "no", "nan"}

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "lr": lr,
        "coeff_l_a_CE": coeff_l_a_CE,
        "coeff_l_cls_CE": coeff_l_cls_CE,
        "coeff_prior_anch": coeff_prior_anch,
        "start_mse": start_mse,
        "end_mse": end_mse,
        "ema_momentum": ema_momentum,
        "use_ema_prior": use_ema,
        "train_backbone": True,
        "backbone_weights": "imagenet",
    }


def _name_list(obj: Any, n: int, prefix: str) -> List[str]:
    if obj is None:
        return [f"{prefix}_{i}" for i in range(n)]

    if isinstance(obj, dict):
        inv = {}
        for k, v in obj.items():
            try:
                inv[int(v)] = str(k)
            except Exception:
                pass
        if inv:
            return [inv.get(i, f"{prefix}_{i}") for i in range(n)]

    try:
        vals = list(obj)
        if len(vals) >= n:
            return [str(vals[i]) for i in range(n)]
    except Exception:
        pass

    return [f"{prefix}_{i}" for i in range(n)]


def _as_255_float(xb):
    xb = tf.cast(xb, tf.float32)
    return tf.cond(tf.reduce_max(xb) <= 1.5, lambda: xb * 255.0, lambda: xb)


def _as_0_1_float(xb):
    xb = tf.cast(xb, tf.float32)
    return tf.cond(tf.reduce_max(xb) > 1.5, lambda: xb / 255.0, lambda: xb)


def _make_eval_ds(X, y, a, batch_size: int, backbone: str, preprocess_mode: str):
    preprocess_fn = get_backbone_preprocess(backbone)

    def _map_preprocess(xb, yb, ab):
        if preprocess_mode == "backbone_preprocess_255":
            xb = _as_255_float(xb)
            xb = preprocess_fn(xb)
        elif preprocess_mode == "identity_255":
            xb = _as_255_float(xb)
        elif preprocess_mode == "identity_0_1":
            xb = _as_0_1_float(xb)
        elif preprocess_mode == "minus1_1":
            xb = _as_255_float(xb)
            xb = xb / 127.5 - 1.0
        else:
            raise ValueError(f"Unknown preprocess_mode={preprocess_mode!r}")
        return xb, yb, ab

    ds = make_tfds(
        X,
        np.asarray(y).reshape(-1).astype(np.int32),
        np.asarray(a).astype(np.float32),
        batch_size,
        shuffle=False,
        cache=False,
        seed=SEED,
        reshuffle_each_iteration=False,
    )
    return ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def _training_loader_save_dir_from_run(run) -> str:
    """
    Reconstruct the save_dir argument used by train_model.py.

    Training run dirs look like:
      <stage_root>/<dataset>_<backbone>/<epochs_batch_opt_lr>/<coeffs>/<schedule>/foldN

    The loader was called before run_dir was constructed, with save_dir=<stage_root>.
    """
    p = Path(run.run_dir)
    dataset = _norm_dataset(run.dataset)
    backbone = str(run.backbone).lower()
    marker1 = f"{dataset}_{backbone}"
    marker2 = f"{dataset.lower()}_{backbone}"

    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part in {marker1, marker2}:
            return str(Path(*parts[:i])) if i > 0 else "."
    # fallback: foldN -> schedule -> coeffs -> hparams -> dataset_backbone -> stage_root
    return str(p.parents[4])

def _load_dataset_once(dataset: str, args, loader_save_dir: str) -> Dict[str, Any]:
    dataset = _norm_dataset(dataset)
    global _CACHE

    cache_key = (dataset, str(Path(loader_save_dir)))
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    
    set_seeds(SEED)

    data_dir = str(getattr(args, "data_dir", None) or "../data/")

    if dataset in {"AwA2", "aPY"}:
        data_name = f"{dataset}_cv"
        data_config = dataset_config.get(dataset)

        (
            images,
            _,
            labels,
            _,
            annotations,
            _,
            mat_pd,
            _,
            concepts,
            label_to_idx,
            mat_GT,
            train_tf,
            test_tf,
        ) = get_dataset_loaders(data_name, data_config, SEED, loader_save_dir, 64, data_dir)

        labels = np.asarray(labels).reshape(-1).astype(np.int32)
        annotations = np.asarray(annotations).astype(np.float32)
        images = np.asarray(images)

        # Keep raw image scale as close as possible to the training script:
        # PIL augment functions expect uint8-style image arrays.
        base_images = (images * 255.0).astype("uint8") if float(np.max(images)) <= 1.01 else images.astype("uint8")

        M, K = int(mat_pd.shape[0]), int(mat_pd.shape[1])

        info = {
            "dataset": dataset,
            "base_images": base_images,
            "labels": labels,
            "annotations": annotations,
            "M": M,
            "K": K,
            "concept_names": _name_list(concepts, M, "concept"),
            "class_names": _name_list(label_to_idx, K, "class"),
            "train_pil_aug": extract_pil_augs(train_tf),
            "test_pil_aug": extract_pil_augs(test_tf),
        }

    else:
        data_config = dataset_config["CUB"]
        result = get_dataset_loaders("CUB", data_config, SEED, loader_save_dir, 64, data_dir)
        train_dl, val_dl, test_dl, _, concepts, classes, concept_map, label_to_idx = result

        X_train, y_train, a_train = numpy_from_dl(train_dl)
        X_val, y_val, a_val = numpy_from_dl(val_dl)
        X_test, y_test, a_test = numpy_from_dl(test_dl)

        # Do NOT force CUB to 255 here. The replay selector below will decide
        # whether the tensors should be used as 0..1, 0..255, or backbone-preprocessed.
        X_train = np.asarray(X_train).astype("float32")
        X_val = np.asarray(X_val).astype("float32")
        X_test = np.asarray(X_test).astype("float32")

        y_train = np.asarray(y_train).reshape(-1).astype(np.int32)
        y_val = np.asarray(y_val).reshape(-1).astype(np.int32)
        y_test = np.asarray(y_test).reshape(-1).astype(np.int32)

        a_train = np.asarray(a_train).astype(np.float32)
        a_val = np.asarray(a_val).astype(np.float32)
        a_test = np.asarray(a_test).astype(np.float32)

        M = int(a_train.shape[1])
        K = int(len(classes))

        info = {
            "dataset": dataset,
            "X_train": X_train,
            "y_train": y_train,
            "a_train": a_train,
            "X_val": X_val,
            "y_val": y_val,
            "a_val": a_val,
            "X_test": X_test,
            "y_test": y_test,
            "a_test": a_test,
            "M": M,
            "K": K,
            "concept_names": _name_list(concepts, M, "concept"),
            "class_names": _name_list(classes, K, "class"),
        }

    _CACHE[cache_key] = info
    return info


def _split_for_run(run, args) -> Dict[str, Any]:
    dataset = _norm_dataset(run.dataset)
    hp = _hparams(run)
    loader_save_dir = _training_loader_save_dir_from_run(run)
    info = _load_dataset_once(dataset, args, loader_save_dir)
    fold_id = int(str(run.fold_or_run))
    img_size = int(getattr(args, "img_size", 224))

    if dataset in {"AwA2", "aPY"}:
        labels = info["labels"]
        base_images = info["base_images"]
        annotations = info["annotations"]

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        chosen = None
        for i, (train_idx, test_idx) in enumerate(kf.split(base_images, labels), start=1):
            if i == fold_id:
                chosen = (train_idx, test_idx)
                break

        if chosen is None:
            raise ValueError(f"Cannot reconstruct fold {fold_id} for {dataset}")

        train_idx, test_idx = chosen
        tr_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.2,
            random_state=SEED,
            stratify=labels[train_idx],
        )

        X_train_raw = base_images[tr_idx]
        X_val_raw = base_images[val_idx]
        X_test_raw = base_images[test_idx]

        y_train, a_train = labels[tr_idx], annotations[tr_idx]
        y_val, a_val = labels[val_idx], annotations[val_idx]
        y_test, a_test = labels[test_idx], annotations[test_idx]

        # Match train_model.py RNG + augmentation order exactly.
        # Training generated X_train_aug first, then X_val_aug, then X_test_aug.
        set_seeds(SEED)
        try:
            import torch
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        except Exception:
            pass

        X_train_aug = apply_pil_augs_np(X_train_raw, info["train_pil_aug"], img_size).astype(np.uint8)
        X_val = apply_pil_augs_np(X_val_raw, info["test_pil_aug"], img_size).astype(np.uint8)
        X_test = apply_pil_augs_np(X_test_raw, info["test_pil_aug"], img_size).astype(np.uint8)

        assert X_train_aug.dtype == np.uint8 and X_train_aug.max() <= 255 and X_train_aug.min() >= 0
        assert X_val.dtype == np.uint8 and X_val.max() <= 255 and X_val.min() >= 0
        assert X_test.dtype == np.uint8 and X_test.max() <= 255 and X_test.min() >= 0

        # X_train_aug is intentionally computed to consume the same RNG as training.
        #del X_train_aug

        P_true_global, class_count_np = compute_P_true_global(a_train, y_train, info["K"])

    else:
        X_train_aug = info["X_train"]
        y_train = info["y_train"]
        a_train = info["a_train"]

        X_val = info["X_val"]
        y_val = info["y_val"]
        a_val = info["a_val"]

        X_test = info["X_test"]
        y_test = info["y_test"]
        a_test = info["a_test"]

        y_train = info["y_train"]
        a_train = info["a_train"]

        P_true_global, class_count_np = compute_P_true_global(a_train, y_train, info["K"])

    return {
        "X_train": X_train_aug if dataset in {"AwA2", "aPY"} else info["X_train"],
        "y_train": np.asarray(y_train).reshape(-1).astype(np.int32),
        "a_train": np.asarray(a_train).astype(np.float32),
        "X_val": X_val,
        "y_val": np.asarray(y_val).reshape(-1).astype(np.int32),
        "a_val": np.asarray(a_val).astype(np.float32),
        "X_test": X_test,
        "y_test": np.asarray(y_test).reshape(-1).astype(np.int32),
        "a_test": np.asarray(a_test).astype(np.float32),
        "P_true_global": P_true_global,
        "class_count": class_count_np.astype(np.float32),
        "M": info["M"],
        "K": info["K"],
        "concept_names": info["concept_names"],
        "class_names": info["class_names"],
        "hparams": hp,
    }


def _find_weights_for_run(run) -> Optional[Path]:
    for attr in ("best_weights_path", "weights_path"):
        val = getattr(run, attr, None)
        if val:
            q = Path(val)
            if q.exists():
                return q

    run_dir = Path(getattr(run, "run_dir", ""))
    for q in [
        run_dir / "models" / "best_val_bal_acc.weights.h5",
        run_dir / "final.weights.h5",
    ]:
        if q.exists():
            return q

    return None


def _strip_var_suffix(name: str) -> str:
    name = str(name).replace(":0", "")
    while name.startswith("/"):
        name = name[1:]
    return name


def _var_path(v) -> str:
    return _strip_var_suffix(getattr(v, "path", getattr(v, "name", "")))


def _h5_weight_map(weights_path: Path) -> Dict[str, np.ndarray]:

    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(weights_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                arrays[_strip_var_suffix(name)] = np.asarray(obj)
        f.visititems(visit)
    return arrays


def _candidate_keys_for_variable(var_path: str) -> List[str]:
    parts = [p for p in _strip_var_suffix(var_path).split("/") if p]
    keys = []

    if parts:
        keys.append("/".join(parts))

    # Remove top model prefix if present.
    if parts and parts[0] == "pacb_model":
        keys.append("/".join(parts[1:]))

    if len(parts) >= 2:
        layer, weight_name = parts[-2], parts[-1]

        keys.append(f"{layer}/{weight_name}")

        # Legacy concept heads.
        if layer.startswith("attribute_output_concept_"):
            keys.append(f"{layer}/pacb_model/{layer}/{weight_name}")

        # PACBM class/prior layers.
        if layer in {"class_layer", "concept_class_priors"}:
            keys.append(f"{layer}/{weight_name}")

        # IMPORTANT: Keras may expose DepthwiseConv2D variable as "kernel",
        # while H5 saved it as "depthwise_kernel".
        if layer.endswith("_dwconv") and weight_name == "kernel":
            keys.append(f"{layer}/depthwise_kernel")

    if len(parts) >= 3:
        suffix3 = "/".join(parts[-3:])
        keys.append(suffix3)

        # IMPORTANT: EfficientNet H5 paths include the backbone prefix:
        # efficientnetb0/block1a_dwconv/depthwise_kernel
        backbone_prefixes = [
            "efficientnetb0",
            "mobilenetv2_1.00_224",
            "inception_v3",
            "inceptionv3",
        ]

        for prefix in backbone_prefixes:
            keys.append(f"{prefix}/{suffix3}")

        # Also try depthwise_kernel with backbone prefix.
        layer, weight_name = parts[-2], parts[-1]
        if layer.endswith("_dwconv") and weight_name == "kernel":
            for prefix in backbone_prefixes:
                keys.append(f"{prefix}/{layer}/depthwise_kernel")

    return list(dict.fromkeys(keys))

    

def _find_h5_key_for_var(var_path: str, var_shape: Tuple[int, ...], h5_arrays: Dict[str, np.ndarray]) -> Optional[str]:
    var_path = _strip_var_suffix(var_path)
    parts = [p for p in var_path.split("/") if p]

    # 1. Exact/candidate matches.
    for key in _candidate_keys_for_variable(var_path):
        arr = h5_arrays.get(key)
        if arr is not None and tuple(arr.shape) == tuple(var_shape):
            return key

    # 2. General suffix match.
    if len(parts) >= 2:
        suffix = "/".join(parts[-2:])
        matches = [
            k for k, arr in h5_arrays.items()
            if k.endswith(suffix) and tuple(arr.shape) == tuple(var_shape)
        ]
        if len(matches) == 1:
            return matches[0]

    # 3. IMPORTANT: DepthwiseConv2D naming mismatch.
    # Model variable may be:
    #   block1a_dwconv/kernel
    # H5 may contain:
    #   efficientnetb0/block1a_dwconv/depthwise_kernel
    # or similar.
    if len(parts) >= 2:
        layer, weight_name = parts[-2], parts[-1]
        if weight_name == "kernel" and ("dwconv" in layer.lower() or "depthwise" in layer.lower()):
            depthwise_suffix = f"{layer}/depthwise_kernel"
            matches = [
                k for k, arr in h5_arrays.items()
                if k.endswith(depthwise_suffix) and tuple(arr.shape) == tuple(var_shape)
            ]
            if len(matches) == 1:
                return matches[0]

    # 4. Last-resort unique shape + final weight-name match.
    # Keep this conservative: only for depthwise kernels.
    if len(parts) >= 2:
        layer, weight_name = parts[-2], parts[-1]
        if weight_name == "kernel" and ("dwconv" in layer.lower() or "depthwise" in layer.lower()):
            matches = [
                k for k, arr in h5_arrays.items()
                if k.endswith("/depthwise_kernel") and tuple(arr.shape) == tuple(var_shape)
            ]
            if len(matches) == 1:
                return matches[0]

    return None


def _load_pacbm_h5_weights_by_assignment(model, weights_path: Path) -> str:
    h5_arrays = _h5_weight_map(weights_path)
    assigned = []
    missing = []

    for var in model.weights:
        vp = _var_path(var)
        key = _find_h5_key_for_var(vp, tuple(var.shape), h5_arrays)

        if key is None:
            missing.append(vp)
            continue

        arr = h5_arrays[key]
        if tuple(arr.shape) != tuple(var.shape):
            missing.append(vp)
            continue

        var.assign(arr)
        assigned.append((vp, key))

    loaded_var_names = {vp for vp, _ in assigned}

    def _has_loaded_suffix(suffix: str) -> bool:
        return any(vp.endswith(suffix) for vp in loaded_var_names)

    essential_errors = []

    if not _has_loaded_suffix("class_layer/gamma_mk"):
        essential_errors.append("class_layer/gamma_mk was not loaded")

    if not _has_loaded_suffix("class_layer/bias_k"):
        essential_errors.append("class_layer/bias_k was not loaded")

    if not any("attribute_output_concept_0" in vp and vp.endswith("kernel") for vp in loaded_var_names):
        essential_errors.append("attribute concept heads were not loaded")

    gamma_h5_key = next((k for k in h5_arrays if k.endswith("class_layer/gamma_mk")), None)
    gamma_var = next((v for v in model.weights if _var_path(v).endswith("class_layer/gamma_mk")), None)

    gamma_ok = False
    if gamma_h5_key is not None and gamma_var is not None:
        gamma_ok = np.allclose(np.asarray(gamma_var), h5_arrays[gamma_h5_key], rtol=1e-5, atol=1e-6)
        if not gamma_ok:
            essential_errors.append("loaded class_layer/gamma_mk does not match H5 reference")

    if essential_errors:
        examples = "; ".join(missing[:12])
        raise ValueError(
            "PACBM legacy H5 assignment failed: "
            + "; ".join(essential_errors)
            + f". assigned={len(assigned)}/{len(model.weights)}. missing_examples={examples}"
        )

    return (
        "loaded_legacy_h5_by_assignment"
        f": assigned={len(assigned)}/{len(model.weights)}"
        f", h5_datasets={len(h5_arrays)}"
        f", gamma_verified={gamma_ok}"
        f", missing_noncritical={len(missing)}"
    )


def _history_best_val_bal_acc(run) -> Optional[float]:
    hist_path = getattr(run, "history_path", None)
    if not hist_path:
        return None

    hist_path = Path(hist_path)
    if not hist_path.exists():
        return None

    try:
        hist = pd.read_csv(hist_path)
    except Exception:
        return None

    monitor = None
    for c in ["val_bal_acc", "val_balanced_accuracy", "val_accuracy", "val_acc"]:
        if c in hist.columns:
            monitor = c
            break

    if monitor is None:
        return None

    vals = pd.to_numeric(hist[monitor], errors="coerce")
    if not vals.notna().any():
        return None

    return float(vals.loc[vals.idxmax()])


def _eval_bal_acc(model, ds) -> float:
    ys, ps = [], []

    for xb, yb, ab in ds:
        outputs = model([xb, yb, ab], training=False)
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 5:
            probs = outputs[4]
        elif isinstance(outputs, dict):
            probs = outputs.get("class_probs")
        else:
            raise ValueError("Cannot parse PACBM outputs while selecting preprocessing mode.")

        ys.append(np.asarray(yb).reshape(-1))
        ps.append(np.argmax(np.asarray(probs), axis=1))

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float(balanced_accuracy_score(y, p))


def _select_preprocess_mode(model, run, split: Dict[str, Any], hp: Dict[str, Any], backbone: str) -> Tuple[str, str]:
    target = _history_best_val_bal_acc(run)

    # If no history is available, use the historically intended mode.
    if target is None:
        return "backbone_preprocess_255", "selected_preprocess=backbone_preprocess_255; no_history_target_available"

    results = []
    for mode in PREPROCESS_MODES:
        try:
            ds = _make_eval_ds(
                split["X_val"],
                split["y_val"],
                split["a_val"],
                hp["batch_size"],
                backbone,
                mode,
            )
            bal = _eval_bal_acc(model, ds)
            results.append((abs(bal - target), mode, bal))
        except Exception as exc:
            results.append((float("inf"), mode, f"ERR:{type(exc).__name__}:{str(exc)[:120]}"))

    results_sorted = sorted(results, key=lambda x: x[0])
    best_delta, best_mode, best_val = results_sorted[0]

    compact = "; ".join(
        f"{mode}={val if isinstance(val, str) else f'val:.6f'},delta={delta if np.isfinite(delta) else 'inf'}"
        for delta, mode, val in results_sorted
    )

    note = (
        f"selected_preprocess={best_mode}; "
        f"history_best_val_bal={target:.6f}; "
        f"selected_replay_val_bal={best_val if isinstance(best_val, str) else f'best_val:.6f'}; "
        f"selected_abs_delta={best_delta:.6f}; "
        f"all_preprocess_trials=[{compact}]"
    )

    return best_mode, note


def get_eval_bundle(run, args) -> Dict[str, Any]:
    dataset = _norm_dataset(run.dataset)
    backbone = str(run.backbone).lower()
    split = _split_for_run(run, args)
    hp = split["hparams"]
    img_size = int(getattr(args, "img_size", 224))

    class_count = tf.convert_to_tensor(split["class_count"], dtype=tf.float32)
    p_init = split["P_true_global"] if dataset in {"AwA2", "aPY"} else None

    model = build_and_compile(
        M=split["M"],
        K=split["K"],
        backbone=backbone,
        img_size=img_size,
        lr=hp["lr"],
        dataset_name=dataset,
        train_backbone=hp["train_backbone"],
        coeff_l_a_CE=hp["coeff_l_a_CE"],
        coeff_l_cls_CE=hp["coeff_l_cls_CE"],
        coeff_prior_anch=hp["coeff_prior_anch"],
        save_dir=str(run.run_dir),
        class_count=class_count,
        P_true_global=p_init,
        backbone_weights=hp["backbone_weights"],
        use_ema_prior=hp["use_ema_prior"],
        ema_momentum=hp["ema_momentum"],
        optimizer_name=hp["optimizer"],
    )

    weights_path = _find_weights_for_run(run)
    load_info = "no_weights_found"

    if weights_path is not None:
        # New ablation checkpoints use the native Keras 3 format.
        # Older PACBM checkpoints require manual legacy HDF5 assignment.
        with h5py.File(weights_path, "r") as f:
            root_keys = set(f.keys())

        is_native_keras3 = (
            "layers" in root_keys
            or "vars" in root_keys
            or "optimizer" in root_keys
        )

        if is_native_keras3:
            gamma_before = model.class_layer.gamma.numpy().copy()

            # The optimizer-state warning is harmless for inference.
            model.load_weights(str(weights_path))

            gamma_after = model.class_layer.gamma.numpy().copy()

            load_info = (
                "loaded_native_keras3_weights"
                f"; gamma_shape={gamma_after.shape}"
                f"; gamma_changed={not np.allclose(gamma_before, gamma_after)}"
            )
        else:
            load_info = _load_pacbm_h5_weights_by_assignment(
                model,
                weights_path,
            )
    """if weights_path is not None:
        load_info = _load_pacbm_h5_weights_by_assignment(model, weights_path)"""

    

    preprocess_mode = "backbone_preprocess_255"
    preprocess_note = "selected_preprocess=backbone_preprocess_255; forced_to_match_training_script"

    val_ds = _make_eval_ds(
        split["X_val"],
        split["y_val"],
        split["a_val"],
        hp["batch_size"],
        backbone,
        preprocess_mode,
    )

    test_ds = _make_eval_ds(
        split["X_test"],
        split["y_test"],
        split["a_test"],
        hp["batch_size"],
        backbone,
        preprocess_mode,
    )

    return {
        "model": model,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "concept_names": split["concept_names"],
        "class_names": split["class_names"],
        "P_true_global": split["P_true_global"],
        "weights_already_loaded": weights_path is not None,
        "adapter_weight_load_info": load_info + "; " + preprocess_note,
        "selected_preprocess_mode": preprocess_mode,
    }


def get_prior_matrix(run, args):
    return _split_for_run(run, args)["P_true_global"]


def get_concept_names(run, args):
    return _split_for_run(run, args)["concept_names"]


def get_class_names(run, args):
    return _split_for_run(run, args)["class_names"]