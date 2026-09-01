
import os

# Must be before TensorFlow import.
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "0")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

import gc
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import tensorflow as tf
from torchvision import transforms
from PIL import Image

from data_loaders.data_loader import get_dataset_loaders
from data_loaders.configs import dataset_config

try:
    from models.cnn.utils import numpy_from_dl as cnn_numpy_from_dl
except Exception:
    cnn_numpy_from_dl = None

try:
    from models.klcbm.utils import numpy_from_dl as kl_numpy_from_dl
except Exception:
    kl_numpy_from_dl = None

try:
    from models.jointcbm.utils import numpy_from_dl as joint_numpy_from_dl
except Exception:
    joint_numpy_from_dl = None

try:
    from models.pacbm.utils import numpy_from_dl as pac_numpy_from_dl
except Exception:
    pac_numpy_from_dl = None


AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

DATASETS = ["AwA2", "aPY", "CUB"]
BACKBONES = ["mobilenetv2", "efficientnetb0", "inceptionv3"]
MODELS = ["cnn", "jointcbm", "klcbm", "pacbm"]


# =============================================================================
# General helpers
# =============================================================================

def set_all_seeds(seed: int = SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        import random
        random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Could not set GPU memory growth:", e)


def get_backbone_preprocess(backbone_name: str):
    name = backbone_name.lower()

    if name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input

    if name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input

    if name == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input

    if name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input

    return lambda x: x


def build_backbone(backbone_name: str, input_shape, weights="imagenet", train_backbone=True):
    name = backbone_name.lower()

    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )

    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )

    elif name == "inceptionv3":
        base = tf.keras.applications.InceptionV3(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )

    elif name == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    base.trainable = bool(train_backbone)
    return base


def extract_pil_augs(torch_transform):
    if isinstance(torch_transform, transforms.Compose):
        pil_transforms = []

        for t in torch_transform.transforms:
            if isinstance(t, transforms.ToTensor):
                break
            if isinstance(t, transforms.Normalize):
                continue
            pil_transforms.append(t)

        return transforms.Compose(pil_transforms) if pil_transforms else None

    return torch_transform


def to_uint8_images(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    if float(np.nanmax(x)) <= 1.01:
        return np.clip(x * 255.0, 0, 255).astype("uint8")

    return np.clip(x, 0, 255).astype("uint8")


def apply_pil_augs_np(X, pil_transform, image_size):
    X_out = []

    for x in X:
        img = Image.fromarray(x.astype(np.uint8))

        if pil_transform is not None:
            img = pil_transform(img)

        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BILINEAR)

        x_aug = np.array(img, dtype=np.uint8)

        if x_aug.ndim == 2:
            x_aug = np.stack([x_aug] * 3, axis=-1)

        X_out.append(x_aug)

    return np.stack(X_out, axis=0).astype("uint8")


def maybe_resize_uint8(X: np.ndarray, image_size: int) -> np.ndarray:
    X = to_uint8_images(X)

    if X.shape[1] == image_size and X.shape[2] == image_size and X.shape[-1] == 3:
        return X

    return apply_pil_augs_np(X, None, image_size)


def make_image_ds(
    X: np.ndarray,
    y: np.ndarray,
    a: Optional[np.ndarray],
    batch_size: int,
    backbone: str,
    mode: str,
):
    """
    mode:
      cnn      -> xb, yb
      joint    -> xb, (ab, yb)
      klcbm    -> (xb, yb), (ab, yb)
      pacbm    -> xb, yb, ab
    """
    y = np.asarray(y).reshape(-1).astype("int32")

    if a is None:
        a = np.zeros((len(y), 1), dtype="float32")
    else:
        a = np.asarray(a).astype("float32")

    X = np.asarray(X).astype("uint8")
    preprocess_fn = get_backbone_preprocess(backbone)

    ds = tf.data.Dataset.from_tensor_slices((X, y, a))
    ds = ds.batch(batch_size, drop_remainder=False)

    def _preprocess(xb, yb, ab):
        xb = tf.cast(xb, tf.float32)
        xb_max = tf.reduce_max(xb)
        xb = tf.cond(xb_max <= 1.5, lambda: xb * 255.0, lambda: xb)
        xb = preprocess_fn(xb)

        if mode == "cnn":
            return xb, yb

        if mode == "joint":
            return xb, (ab, yb)

        if mode == "klcbm":
            return (xb, yb), (ab, yb)

        if mode == "pacbm":
            return xb, yb, ab

        raise ValueError(f"Unknown ds mode: {mode}")

    return ds.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype("int32")
    y_pred = np.asarray(y_pred).reshape(-1).astype("int32")

    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=K,
        dtype=tf.float32,
    ).numpy()

    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)

    recall = np.divide(
        tp,
        row_sum,
        out=np.zeros_like(tp),
        where=row_sum > 0,
    )

    mask = row_sum > 0
    return float(recall[mask].mean() if mask.any() else 0.0)


def parse_train_config(config_name: str) -> Tuple[int, int, str, float]:
    parts = config_name.split("_")

    if len(parts) < 4:
        raise ValueError(f"Cannot parse training config: {config_name}")

    epochs = int(parts[0])
    batch_size = int(parts[1])
    optimizer = parts[2]
    lr = float(parts[3])

    return epochs, batch_size, optimizer, lr


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# Robust HDF5 weight loader
# =============================================================================

def _leaf_name(name: str) -> str:
    name = str(name)
    leaf = name.split("/")[-1]

    if leaf.endswith(":0"):
        leaf = leaf[:-2]

    return leaf


def _path_parts(path: str) -> list[str]:
    parts = []

    for p in str(path).split("/"):
        if p.endswith(":0"):
            p = p[:-2]
        parts.append(p)

    return parts


def _collect_h5_datasets(weights_path: Path):
    items = []

    with h5py.File(weights_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append((name, obj[()]))

        f.visititems(visit)

    return items


def _iter_leaf_layers_recursive(layer):
    """
    Yield only real leaf layers, not model/container layers.

    This is important for subclassed models. The root PACBModel/KLCBMmodel
    contains all variables, so matching from it causes many ambiguous matches.
    """
    children = getattr(layer, "layers", None)

    if children:
        for sub in children:
            yield from _iter_leaf_layers_recursive(sub)
    else:
        yield layer


def _h5_path_matches_layer_var(h5_path: str, layer_name: str, var_leaf: str) -> bool:
    parts = _path_parts(h5_path)

    if not parts:
        return False

    h5_leaf = parts[-1]

    # Exact path-segment layer match only.
    if layer_name not in parts[:-1]:
        return False

    # Normal exact variable-name match.
    if h5_leaf == var_leaf:
        return True

    # Keras sometimes exposes DepthwiseConv2D variable as "kernel"
    # but stores it in H5 as "depthwise_kernel".
    if var_leaf == "kernel" and h5_leaf == "depthwise_kernel":
        return True

    return False

def _manual_load_weights_by_layer_and_shape(model: tf.keras.Model, weights_path: Path) -> dict:
    h5_items = _collect_h5_datasets(weights_path)

    used_h5_paths = set()
    assigned = []
    unmatched = []
    ambiguous = []

    for layer in _iter_leaf_layers_recursive(model):
        layer_name = getattr(layer, "name", None)

        if layer_name is None:
            continue

        for var in getattr(layer, "weights", []):
            var_name = getattr(var, "path", getattr(var, "name", ""))
            var_leaf = _leaf_name(var_name)
            var_shape = tuple(var.shape)

            candidates = []

            for h5_path, arr in h5_items:
                if h5_path in used_h5_paths:
                    continue

                if tuple(arr.shape) != var_shape:
                    continue

                if _h5_path_matches_layer_var(h5_path, layer_name, var_leaf):
                    candidates.append((h5_path, arr))

            if len(candidates) == 1:
                h5_path, arr = candidates[0]
                var.assign(arr)
                used_h5_paths.add(h5_path)
                assigned.append((var_name, h5_path))

            elif len(candidates) == 0:
                unmatched.append(var_name)

            else:
                ambiguous.append({
                    "var": var_name,
                    "layer": layer_name,
                    "leaf": var_leaf,
                    "shape": var_shape,
                    "candidates": [p for p, _ in candidates[:20]],
                    "n_candidates": len(candidates),
                })

    harmless_leaf_names = {
        "anchor_alpha",
        "cm_train",
        "cm_val",
        "class_weights",
    }

    real_unmatched = []

    for v in unmatched:
        leaf = _leaf_name(v)

        if leaf not in harmless_leaf_names:
            real_unmatched.append(v)

    report = {
        "assigned_count": len(assigned),
        "unmatched_count": len(real_unmatched),
        "ambiguous_count": len(ambiguous),
        "assigned_preview": assigned[:20],
        "unmatched_preview": real_unmatched[:50],
        "ambiguous_preview": ambiguous[:10],
    }

    if real_unmatched or ambiguous:
        raise ValueError(
            "Manual H5 loading did not fully succeed.\n"
            f"Assigned: {len(assigned)}\n"
            f"Unmatched non-harmless vars: {len(real_unmatched)}\n"
            f"Ambiguous vars: {len(ambiguous)}\n"
            f"Unmatched preview: {real_unmatched[:20]}\n"
            f"Ambiguous preview: {ambiguous[:3]}"
        )

    return report


def load_weights_strict_or_manual(model: tf.keras.Model, weights_path: Path, label: str) -> str:
    try:
        model.load_weights(str(weights_path))
        print(f"[LOAD OK] {label}: strict Keras load")
        return "strict"

    except Exception as e:
        print(f"[LOAD WARN] {label}: strict Keras load failed:")
        print(f"  {type(e).__name__}: {str(e).splitlines()[0]}")
        print("  Trying manual HDF5 leaf-layer fallback...")

    report = _manual_load_weights_by_layer_and_shape(model, weights_path)

    print(
        f"[LOAD OK] {label}: manual HDF5 leaf-layer fallback "
        f"(assigned={report['assigned_count']}, "
        f"unmatched={report['unmatched_count']}, "
        f"ambiguous={report['ambiguous_count']})"
    )

    return "manual_h5_leaf"


def load_weights_strict(
    model: tf.keras.Model,
    weights_path: Path,
    label: str,
) -> str:

    model.load_weights(str(weights_path))

    print(f"[LOAD OK] {label}: strict native Keras load")
    return "strict_native"


# =============================================================================
# Data loading and fold reconstruction
# =============================================================================

def numpy_from_dl_any(dl):
    for fn in [pac_numpy_from_dl, joint_numpy_from_dl, kl_numpy_from_dl, cnn_numpy_from_dl]:
        if fn is not None:
            return fn(dl)

    xs, ys, attrs = [], [], []

    for batch in dl:
        if len(batch) == 3:
            x, y, a = batch
        elif len(batch) == 2:
            x, y = batch
            a = None
        else:
            raise ValueError("Unknown dataloader batch format.")

        xs.append(np.asarray(x))
        ys.append(np.asarray(y))

        if a is not None:
            attrs.append(np.asarray(a))

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    a = np.concatenate(attrs, axis=0) if attrs else None

    return X, y, a


class DatasetCache:
    def __init__(
        self,
        data_dir: str,
        scratch_dir: str,
        runs: int,
        val_split: float,
        batch_size: int,
    ):
        self.data_dir = data_dir
        self.scratch_dir = scratch_dir
        self.runs = runs
        self.val_split = val_split
        self.batch_size = batch_size
        self.cache: Dict[str, dict] = {}

    def load(self, dataset: str) -> dict:
        if dataset in self.cache:
            return self.cache[dataset]

        print(f"[DATA] Loading {dataset}")
        set_all_seeds(SEED)

        data_cfg = dataset_config.get(dataset)

        if data_cfg is None and dataset == "CUB":
            data_cfg = dataset_config.get("CUB")

        if dataset in ["AwA2", "aPY"]:
            data_name = f"{dataset}_cv"

            (
                images, _,
                labels, _,
                annotations, _,
                mat_pd, _,
                concepts,
                label_to_idx_awa,
                mat_GT,
                train_tf,
                test_tf,
            ) = get_dataset_loaders(
                data_name,
                data_cfg,
                SEED,
                self.scratch_dir,
                self.batch_size,
                self.data_dir,
            )

            images = to_uint8_images(images)
            labels = np.asarray(labels).reshape(-1).astype("int32")
            annotations = np.asarray(annotations).astype("float32")

            K = int(np.max(labels)) + 1
            M = int(annotations.shape[1])

            kf = StratifiedKFold(
                n_splits=self.runs,
                shuffle=True,
                random_state=SEED,
            )

            folds = {}
            fold_id = 0

            for full_train_idx, test_idx in kf.split(images, labels):
                fold_id += 1

                tr_idx, val_idx = train_test_split(
                    full_train_idx,
                    test_size=self.val_split,
                    random_state=SEED,
                    stratify=labels[full_train_idx],
                )

                folds[fold_id] = {
                    "train_idx": tr_idx,
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                    "X_train_raw": images[tr_idx],
                    "y_train": labels[tr_idx],
                    "a_train": annotations[tr_idx],
                    "X_val_raw": images[val_idx],
                    "y_val": labels[val_idx],
                    "a_val": annotations[val_idx],
                    "X_test_raw": images[test_idx],
                    "y_test": labels[test_idx],
                    "a_test": annotations[test_idx],
                }

            obj = {
                "dataset": dataset,
                "K": K,
                "M": M,
                "folds": folds,
                "train_pil_aug": extract_pil_augs(train_tf),
                "test_pil_aug": extract_pil_augs(test_tf),
                "concepts": concepts,
            }

        elif dataset == "CUB":
            result = get_dataset_loaders(
                "CUB",
                data_cfg,
                SEED,
                self.scratch_dir,
                self.batch_size,
                self.data_dir,
            )

            train_dl, val_dl, test_dl, _, concepts, classes, concept_map, label_to_idx_awa = result

            X_train, y_train, a_train = numpy_from_dl_any(train_dl)
            X_val, y_val, a_val = numpy_from_dl_any(val_dl)
            X_test, y_test, a_test = numpy_from_dl_any(test_dl)

            X_train = to_uint8_images(X_train)
            X_val = to_uint8_images(X_val)
            X_test = to_uint8_images(X_test)

            y_train = np.asarray(y_train).reshape(-1).astype("int32")
            y_val = np.asarray(y_val).reshape(-1).astype("int32")
            y_test = np.asarray(y_test).reshape(-1).astype("int32")

            a_train = np.asarray(a_train).astype("float32")
            a_val = np.asarray(a_val).astype("float32")
            a_test = np.asarray(a_test).astype("float32")

            K = len(classes)
            M = int(a_train.shape[1])

            folds = {}

            for fold_id in range(1, self.runs + 1):
                folds[fold_id] = {
                    "train_idx": None,
                    "val_idx": None,
                    "test_idx": None,
                    "X_train_raw": X_train,
                    "y_train": y_train,
                    "a_train": a_train,
                    "X_val_raw": X_val,
                    "y_val": y_val,
                    "a_val": a_val,
                    "X_test_raw": X_test,
                    "y_test": y_test,
                    "a_test": a_test,
                }

            obj = {
                "dataset": dataset,
                "K": K,
                "M": M,
                "folds": folds,
                "train_pil_aug": None,
                "test_pil_aug": None,
                "concepts": concepts,
            }

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.cache[dataset] = obj
        return obj


def prepare_fold_arrays(data_obj: dict, fold_id: int, image_size: int) -> dict:
    fold = data_obj["folds"][fold_id]

    X_train = maybe_resize_uint8(fold["X_train_raw"], image_size)
    X_val = maybe_resize_uint8(fold["X_val_raw"], image_size)
    X_test = maybe_resize_uint8(fold["X_test_raw"], image_size)

    if data_obj["dataset"] in ["AwA2", "aPY"]:
        test_aug = data_obj["test_pil_aug"]

        if test_aug is not None:
            X_val = apply_pil_augs_np(fold["X_val_raw"], test_aug, image_size)
            X_test = apply_pil_augs_np(fold["X_test_raw"], test_aug, image_size)

    return {
        "X_train": X_train,
        "y_train": np.asarray(fold["y_train"]).reshape(-1).astype("int32"),
        "a_train": np.asarray(fold["a_train"]).astype("float32"),
        "X_val": X_val,
        "y_val": np.asarray(fold["y_val"]).reshape(-1).astype("int32"),
        "a_val": np.asarray(fold["a_val"]).astype("float32"),
        "X_test": X_test,
        "y_test": np.asarray(fold["y_test"]).reshape(-1).astype("int32"),
        "a_test": np.asarray(fold["a_test"]).astype("float32"),
        "test_idx": fold["test_idx"],
    }


# =============================================================================
# Saved-run discovery
# =============================================================================

def weight_path_for_fold_dir(fold_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    best = fold_dir / "models" / "best_val_bal_acc.weights.h5"
    final = fold_dir / "final.weights.h5"

    if best.exists():
        return best, "best_val_bal_acc"

    if final.exists():
        return final, "final"

    return None, None


def model_search_roots(model_name: str) -> List[str]:
    if model_name == "cnn":
        return ["opt_lr_search"]

    if model_name == "jointcbm":
        return ["coeff_search", "opt_lr_search"]

    if model_name == "klcbm":
        return ["coeff_search", "opt_lr_search"]

    if model_name == "pacbm":
        return ["coeff_search", "pacbm_dynamic_search", "opt_lr_search"]

    raise ValueError(model_name)


def find_fold_dir(
    trained_root: Path,
    model_name: str,
    dataset: str,
    backbone: str,
    fold_id: int,
) -> Tuple[Path, str]:

    native_roots = {
        "cnn": Path("trained_models/cnn_native_20260804"),
        "klcbm": Path("trained_models/klcbm_native_20260804"),
        "pacbm": Path("trained_models/pacbm_cl_native_20260804/pacbm_cl"),
    }

    if model_name not in native_roots:
        raise ValueError(
            f"This native recomputation script only supports "
            f"cnn, klcbm and pacbm. Got: {model_name}"
        )

    combo_root = native_roots[model_name] / f"{dataset}_{backbone}"

    if not combo_root.exists():
        raise FileNotFoundError(
            f"Native rerun directory does not exist: {combo_root}"
        )

    candidates = [
        p for p in combo_root.rglob(f"fold{fold_id}")
        if p.is_dir() and weight_path_for_fold_dir(p)[0] is not None
    ]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No native checkpoint found for "
            f"{model_name} {dataset}/{backbone} fold{fold_id}"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple native checkpoints found for "
            f"{model_name} {dataset}/{backbone} fold{fold_id}:\n"
            + "\n".join(str(p) for p in candidates)
        )

    return candidates[0], "native_rerun"


def config_parts_from_fold_dir_old(model_name: str, fold_dir: Path) -> List[str]:
    if model_name == "cnn":
        return [fold_dir.parent.name]

    if model_name in ["jointcbm", "klcbm"]:
        return [
            fold_dir.parent.parent.name,
            fold_dir.parent.name,
        ]

    if model_name in ["pacbm", "pacbm_2"]:
        return [
            fold_dir.parent.parent.parent.name,
            fold_dir.parent.parent.name,
            fold_dir.parent.name,
        ]

    raise ValueError(model_name)


def config_parts_from_fold_dir(model_name: str, fold_dir: Path) -> List[str]:

    if model_name == "cnn":
        return [
            fold_dir.parent.name,
        ]

    if model_name in ["klcbm", "pacbm"]:
        return [
            fold_dir.parent.parent.name,
            fold_dir.parent.name,
        ]

    raise ValueError(model_name)

# =============================================================================
# CNN
# =============================================================================

def build_cnn_model(K: int, backbone: str, img_size: int, weights="imagenet", train_backbone=True):
    input_shape = (img_size, img_size, 3)

    inp = tf.keras.Input(shape=input_shape, name="image")
    base = build_backbone(
        backbone,
        input_shape,
        weights=weights,
        train_backbone=train_backbone,
    )

    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(K, activation="softmax", name="class")(x)

    return tf.keras.Model(inputs=inp, outputs=out)


# =============================================================================
# JointCBM
# =============================================================================

def build_joint_cbm_model(
    M: int,
    K: int,
    backbone: str,
    img_size: int,
    weights="imagenet",
    train_backbone=True,
):
    input_shape = (img_size, img_size, 3)

    inp = tf.keras.Input(shape=input_shape, name="image")
    base = build_backbone(
        backbone,
        input_shape,
        weights=weights,
        train_backbone=train_backbone,
    )

    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    concepts = tf.keras.layers.Dense(
        M,
        activation="sigmoid",
        name="concepts",
    )(x)

    class_out = tf.keras.layers.Dense(
        K,
        activation="softmax",
        name="class",
    )(concepts)

    return tf.keras.Model(inputs=inp, outputs=[concepts, class_out])


# =============================================================================
# KL-CBM
# =============================================================================

class AttributeClassProbability(tf.keras.layers.Layer):
    def __init__(self, K, M, class_count, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.M = M
        self.class_count = tf.cast(class_count, tf.float32)
        self.trainable = False

    def call(self, inputs):
        concatenated_attributes, labels = inputs

        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        labels_oh = tf.one_hot(labels, depth=self.K, dtype=tf.float32)
        labels_oh = tf.expand_dims(labels_oh, axis=1)

        a = tf.expand_dims(concatenated_attributes, axis=-1)
        attribute_class_probs = a * labels_oh

        P_A_mk = tf.reduce_sum(attribute_class_probs, axis=0)
        P_A_mk = tf.math.divide(P_A_mk, self.class_count)
        P_A_mk.set_shape((self.M, self.K))

        return P_A_mk


class ProdLayer(tf.keras.layers.Layer):
    def __init__(self, K, M, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.M = M

    def call(self, inputs):
        attr_output, sum_output, labels = inputs

        labels_int = tf.cast(tf.reshape(labels, [-1]), tf.int32)

        counts = tf.cast(
            tf.math.bincount(labels_int, minlength=self.K),
            tf.float32,
        )
        class_prob = counts / tf.reduce_sum(counts)

        labels_oh = tf.one_hot(labels_int, self.K)

        top_k_indices = tf.argsort(attr_output, axis=-1)[..., -5:]
        attr_top_k = tf.experimental.numpy.take_along_axis(
            attr_output,
            top_k_indices,
            axis=-1,
        )

        sum_top_k = tf.gather(sum_output, top_k_indices, axis=0)

        div_output = attr_top_k[..., tf.newaxis] / tf.maximum(
            sum_top_k,
            tf.keras.backend.epsilon(),
        )

        product_output = tf.reduce_prod(div_output, axis=1)
        product_output *= class_prob
        product_output = product_output * labels_oh
        product_output.set_shape((None, self.K))

        return div_output, product_output, tf.nn.softmax(product_output, axis=-1)


class KLCBMmodel(tf.keras.Model):
    def __init__(
        self,
        img_size,
        M,
        K,
        backbone,
        coeff_l_attr,
        coeff_l_p_y,
        class_count,
    ):
        super().__init__()

        self.M = M
        self.K = K

        self.backbone_model = backbone
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.attr_layers = [
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                name=f"attribute_output_{i}",
                use_bias=True,
            )
            for i in range(M)
        ]

        self.conc_attr_layer = tf.keras.layers.Concatenate(
            name="concatenated_attributes",
        )

        self.sum_layer = AttributeClassProbability(
            K,
            M,
            class_count,
            name="sum_layer",
        )

        self.prod_layer = ProdLayer(
            K,
            M,
            name="prod_layer",
        )

        self.final_layer = tf.keras.layers.Dense(
            K,
            activation="softmax",
            name="final_output",
            use_bias=False,
            trainable=True,
        )

        self.coeff_l_attr = coeff_l_attr
        self.coeff_l_p_y = coeff_l_p_y

    def call(self, inputs, training=None):
        images, labels = inputs

        x = self.backbone_model(images, training=training)
        x = self.global_avg_pool(x)

        attr_outs = [layer(x) for layer in self.attr_layers]
        concatenated_attributes = self.conc_attr_layer(attr_outs)

        sum_output = self.sum_layer([concatenated_attributes, labels])

        div_output, prod_output, soft_prod_output = self.prod_layer(
            [concatenated_attributes, sum_output, labels]
        )

        final_output = self.final_layer(concatenated_attributes)
        final_output.set_shape((None, self.K))

        return (
            concatenated_attributes,
            sum_output,
            div_output,
            prod_output,
            soft_prod_output,
            final_output,
        )


def build_klcbm_model(
    M,
    K,
    backbone_name,
    img_size,
    class_count,
    weights="imagenet",
    train_backbone=True,
):
    input_shape = (img_size, img_size, 3)

    backbone = build_backbone(
        backbone_name,
        input_shape,
        weights=weights,
        train_backbone=train_backbone,
    )

    model = KLCBMmodel(
        img_size=img_size,
        M=M,
        K=K,
        backbone=backbone,
        coeff_l_attr=1.0,
        coeff_l_p_y=1.0,
        class_count=class_count,
    )

    dummy_x = tf.zeros((1, img_size, img_size, 3), dtype=tf.float32)
    dummy_y = tf.zeros((1,), dtype=tf.int32)

    _ = model([dummy_x, dummy_y], training=False)

    return model


# =============================================================================
# PACBM
# =============================================================================

class ConceptClassPriorsLayer(tf.keras.layers.Layer):
    def __init__(self, K, M, class_count, use_ema=False, momentum=0.9, **kwargs):
        super().__init__(**kwargs)

        self.K = K
        self.M = M
        self.class_count = tf.cast(class_count, tf.float32)
        self.use_ema = bool(use_ema)
        self.momentum = float(momentum)

        self.P_pred_ema = self.add_weight(
            name="P_pred_ema",
            shape=(M, K),
            initializer=tf.keras.initializers.Constant(1e-3),
            trainable=False,
            dtype=tf.float32,
        )

    def call(self, inputs, training=None):
        concepts_pred, labels, attr_ann = inputs

        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        attr_ann = tf.cast(attr_ann, tf.float32)

        one_hot_labels = tf.one_hot(labels, depth=self.K, dtype=tf.float32)
        batch_counts = tf.reduce_sum(one_hot_labels, axis=0)

        P_true_batch = tf.transpose(attr_ann) @ one_hot_labels
        P_true_batch = tf.math.divide_no_nan(
            P_true_batch,
            tf.maximum(batch_counts, 1.0),
        )

        labels_oh = tf.expand_dims(one_hot_labels, axis=1)
        conc = tf.expand_dims(concepts_pred, axis=-1)

        pred_class_probs = conc * labels_oh

        P_pred_batch = tf.reduce_sum(pred_class_probs, axis=0)
        P_pred_batch = tf.math.divide_no_nan(
            P_pred_batch,
            tf.maximum(batch_counts, 1.0),
        )

        if self.use_ema:
            is_first = tf.reduce_all(tf.equal(self.P_pred_ema, 1e-3))

            new_ema = tf.cond(
                is_first,
                lambda: P_pred_batch,
                lambda: self.momentum * self.P_pred_ema
                + (1.0 - self.momentum) * P_pred_batch,
            )

            self.P_pred_ema.assign(new_ema)
            P_pred_used = self.P_pred_ema

        else:
            P_pred_used = P_pred_batch

        return P_pred_used, P_true_batch


class ClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, K, M, **kwargs):
        super().__init__(**kwargs)

        self.K = K
        self.M = M

        self.gamma = self.add_weight(
            name="gamma_mk",
            shape=(M, K),
            initializer="glorot_uniform",
            trainable=True,
            regularizer=tf.keras.regularizers.l2(1e-4),
        )

        self.bias = self.add_weight(
            name="bias_k",
            shape=(K,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, concepts_pred, training=None):
        eps = 1e-2

        a = tf.clip_by_value(concepts_pred, eps, 1.0 - eps)
        logit_a = tf.math.log(a) - tf.math.log(1.0 - a)

        logits = tf.linalg.matmul(logit_a, self.gamma) + self.bias
        probs = tf.nn.softmax(logits, axis=-1)

        return logits, probs


class PACBModel(tf.keras.Model):
    def __init__(
        self,
        input_size,
        M,
        K,
        backbone_name="mobilenetv2",
        backbone_weights="imagenet",
        dataset_name=None,
        train_backbone=True,
        coeff_l_a_CE=1.0,
        coeff_l_cls_CE=1.0,
        coeff_prior_anch=1.0,
        save_dir=None,
        class_count=None,
        use_ema_prior=False,
        ema_momentum=0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.M = M
        self.K = K
        self.save_dir = save_dir

        self.backbone_model = self._get_backbone(
            backbone_name,
            input_size,
            weights=backbone_weights,
        )

        self.train_backbone = bool(train_backbone)

        for layer in self.backbone_model.layers:
            layer.trainable = self.train_backbone

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.5)

        self.attr_layers = [
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                use_bias=True,
                name=f"attribute_output_concept_{c}",
            )
            for c in range(M)
        ]

        self.conc_attr_layer = tf.keras.layers.Concatenate(
            name="concatenated_attributes",
        )

        if class_count is None:
            class_count = tf.ones((K,), dtype=tf.float32)

        cc = tf.maximum(tf.cast(class_count, tf.float32), 1.0)
        w = tf.reduce_sum(cc) / (tf.cast(self.K, tf.float32) * cc)
        self.class_weights = w / tf.reduce_mean(w)

        self.concept_class_priors = ConceptClassPriorsLayer(
            K,
            M,
            class_count,
            use_ema=use_ema_prior,
            momentum=ema_momentum,
            name="concept_class_priors",
        )

        self.class_layer = ClassificationLayer(
            K,
            M,
            name="class_layer",
        )

        self.coeff_l_a_CE = float(coeff_l_a_CE)
        self.coeff_l_cls_CE = float(coeff_l_cls_CE)
        self.coeff_l_prior_anch = float(coeff_prior_anch)

        self.anchor_alpha = self.add_weight(
            name="anchor_alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
            dtype=tf.float32,
        )

        self.cm_train = self.add_weight(
            name="cm_train",
            shape=(K, K),
            initializer="zeros",
            trainable=False,
            dtype=tf.float32,
        )

        self.cm_val = self.add_weight(
            name="cm_val",
            shape=(K, K),
            initializer="zeros",
            trainable=False,
            dtype=tf.float32,
        )

    def _get_backbone(self, name: str, input_shape, weights="imagenet"):
        name = name.lower()

        if name == "mobilenetv2":
            return tf.keras.applications.MobileNetV2(
                include_top=False,
                input_shape=input_shape,
                weights=weights,
            )

        if name == "efficientnetb0":
            return tf.keras.applications.EfficientNetB0(
                include_top=False,
                input_shape=input_shape,
                weights=weights,
            )

        if name == "inceptionv3":
            return tf.keras.applications.InceptionV3(
                include_top=False,
                input_shape=input_shape,
                weights=weights,
            )

        if name == "resnet50":
            return tf.keras.applications.ResNet50(
                include_top=False,
                input_shape=input_shape,
                weights=weights,
            )

        raise NotImplementedError(f"Backbone '{name}' not supported.")

    def call(self, inputs, training=False):
        input_data, labels, attr_ann = inputs

        x = self.backbone_model(input_data, training=training)

        if len(x.shape) == 4:
            x = self.global_avg_pool(x)

        x = self.dropout(x, training=training)

        per_concepts_pred = [head(x) for head in self.attr_layers]
        concepts_pred = self.conc_attr_layer(per_concepts_pred)

        P_pred_used, P_true_batch = self.concept_class_priors(
            [concepts_pred, labels, attr_ann],
            training=training,
        )

        class_logits, class_probs = self.class_layer(
            concepts_pred,
            training=training,
        )

        return (
            per_concepts_pred,
            concepts_pred,
            (P_pred_used, P_true_batch),
            class_logits,
            class_probs,
        )


def build_pacbm_model(
    M,
    K,
    backbone_name,
    img_size,
    class_count,
    dataset_name,
    weights="imagenet",
    train_backbone=True,
    use_ema_prior=True,
    ema_momentum=0.9,
):
    model = PACBModel(
        input_size=(img_size, img_size, 3),
        M=M,
        K=K,
        backbone_name=backbone_name,
        backbone_weights=weights,
        dataset_name=dataset_name,
        train_backbone=train_backbone,
        coeff_l_a_CE=1.0,
        coeff_l_cls_CE=1.0,
        coeff_prior_anch=1.0,
        save_dir=None,
        class_count=class_count,
        use_ema_prior=use_ema_prior,
        ema_momentum=ema_momentum,
    )

    dummy_x = tf.zeros((1, img_size, img_size, 3), dtype=tf.float32)
    dummy_y = tf.zeros((1,), dtype=tf.int32)
    dummy_a = tf.zeros((1, M), dtype=tf.float32)

    _ = model([dummy_x, dummy_y, dummy_a], training=False)

    return model


# =============================================================================
# Recompute routines
# =============================================================================

def recompute_cnn(
    fold_arrays: dict,
    K: int,
    backbone: str,
    img_size: int,
    batch_size: int,
    weights_path: Path,
    backbone_weights,
    train_backbone: bool,
):
    tf.keras.backend.clear_session()

    model = build_cnn_model(
        K,
        backbone,
        img_size,
        weights=backbone_weights,
        train_backbone=train_backbone,
    )

    load_mode = load_weights_strict(model, weights_path, "cnn")

    ds = make_image_ds(
        fold_arrays["X_test"],
        fold_arrays["y_test"],
        None,
        batch_size,
        backbone,
        mode="cnn",
    )

    y_true_all, probs_all = [], []

    for xb, yb in ds:
        probs = model(xb, training=False).numpy()

        y_true_all.append(yb.numpy())
        probs_all.append(probs)

    y_true = np.concatenate(y_true_all, axis=0).astype("int32")
    class_probs = np.concatenate(probs_all, axis=0).astype("float32")
    y_pred = np.argmax(class_probs, axis=1).astype("int32")

    return {
        "load_mode": load_mode,
        "arrays": {
            "y_true": y_true,
            "y_pred": y_pred,
            "class_probs": class_probs,
            "test_idx": fold_arrays["test_idx"]
            if fold_arrays["test_idx"] is not None
            else np.array([], dtype="int32"),
        },
        "extras": {},
    }


def recompute_jointcbm(
    fold_arrays: dict,
    M: int,
    K: int,
    backbone: str,
    img_size: int,
    batch_size: int,
    weights_path: Path,
    backbone_weights,
    train_backbone: bool,
):
    tf.keras.backend.clear_session()

    model = build_joint_cbm_model(
        M,
        K,
        backbone,
        img_size,
        weights=backbone_weights,
        train_backbone=train_backbone,
    )

    load_mode = load_weights_strict(model, weights_path, "jointcbm")

    ds = make_image_ds(
        fold_arrays["X_test"],
        fold_arrays["y_test"],
        fold_arrays["a_test"],
        batch_size,
        backbone,
        mode="joint",
    )

    y_true_all, a_true_all, concept_all, probs_all = [], [], [], []

    for xb, (ab, yb) in ds:
        concept_probs, class_probs = model(xb, training=False)

        y_true_all.append(yb.numpy())
        a_true_all.append(ab.numpy())
        concept_all.append(concept_probs.numpy())
        probs_all.append(class_probs.numpy())

    y_true = np.concatenate(y_true_all, axis=0).astype("int32")
    a_true = np.concatenate(a_true_all, axis=0).astype("float32")
    concept_probs = np.concatenate(concept_all, axis=0).astype("float32")
    class_probs = np.concatenate(probs_all, axis=0).astype("float32")
    y_pred = np.argmax(class_probs, axis=1).astype("int32")

    extras = {}

    try:
        class_layer = model.get_layer("class")
        extras["concept_to_class_kernel"] = class_layer.kernel.numpy().astype("float32")
        extras["concept_to_class_bias"] = class_layer.bias.numpy().astype("float32")
    except Exception:
        pass

    return {
        "load_mode": load_mode,
        "arrays": {
            "y_true": y_true,
            "y_pred": y_pred,
            "a_true": a_true,
            "concept_probs": concept_probs,
            "class_probs": class_probs,
            "test_idx": fold_arrays["test_idx"]
            if fold_arrays["test_idx"] is not None
            else np.array([], dtype="int32"),
        },
        "extras": extras,
    }


def recompute_klcbm(
    fold_arrays: dict,
    M: int,
    K: int,
    backbone: str,
    img_size: int,
    batch_size: int,
    weights_path: Path,
    backbone_weights,
    train_backbone: bool,
):
    tf.keras.backend.clear_session()

    class_count_np = np.bincount(
        fold_arrays["y_train"].astype("int32"),
        minlength=K,
    ).astype("float32")

    class_count = tf.convert_to_tensor(class_count_np, dtype=tf.float32)

    model = build_klcbm_model(
        M,
        K,
        backbone,
        img_size,
        class_count,
        weights=backbone_weights,
        train_backbone=train_backbone,
    )

    load_mode = load_weights_strict(model, weights_path, "klcbm")

    ds = make_image_ds(
        fold_arrays["X_test"],
        fold_arrays["y_test"],
        fold_arrays["a_test"],
        batch_size,
        backbone,
        mode="klcbm",
    )

    y_true_all, a_true_all = [], []
    concept_all, class_probs_all = [], []
    soft_prod_all, prod_all, div_all = [], [], []
    sum_batches = []

    for (xb, yb_in), (ab, yb_true) in ds:
        (
            concept_probs,
            sum_output,
            div_output,
            prod_output,
            soft_prod_output,
            class_probs,
        ) = model([xb, yb_in], training=False)

        y_true_all.append(yb_true.numpy())
        a_true_all.append(ab.numpy())
        concept_all.append(concept_probs.numpy())
        class_probs_all.append(class_probs.numpy())
        soft_prod_all.append(soft_prod_output.numpy())
        prod_all.append(prod_output.numpy())
        div_all.append(div_output.numpy())
        sum_batches.append(sum_output.numpy())

    y_true = np.concatenate(y_true_all, axis=0).astype("int32")
    a_true = np.concatenate(a_true_all, axis=0).astype("float32")
    concept_probs = np.concatenate(concept_all, axis=0).astype("float32")
    class_probs = np.concatenate(class_probs_all, axis=0).astype("float32")
    y_pred = np.argmax(class_probs, axis=1).astype("int32")

    soft_prod_output = np.concatenate(soft_prod_all, axis=0).astype("float32")
    prod_output = np.concatenate(prod_all, axis=0).astype("float32")
    div_output = np.concatenate(div_all, axis=0).astype("float32")
    sum_output_batches = np.stack(sum_batches, axis=0).astype("float32")

    extras = {
        "class_count": class_count_np.astype("float32"),
        "kl_soft_prod_output": soft_prod_output,
        "kl_prod_output": prod_output,
        "kl_div_output": div_output,
        "kl_sum_output_batches": sum_output_batches,
        "concept_to_class_kernel": model.final_layer.kernel.numpy().astype("float32"),
    }

    return {
        "load_mode": load_mode,
        "arrays": {
            "y_true": y_true,
            "y_pred": y_pred,
            "a_true": a_true,
            "concept_probs": concept_probs,
            "class_probs": class_probs,
            "y_transparent": soft_prod_output,
            "test_idx": fold_arrays["test_idx"]
            if fold_arrays["test_idx"] is not None
            else np.array([], dtype="int32"),
        },
        "extras": extras,
    }


def recompute_pacbm(
    fold_arrays: dict,
    M: int,
    K: int,
    backbone: str,
    img_size: int,
    batch_size: int,
    weights_path: Path,
    dataset: str,
    backbone_weights,
    train_backbone: bool,
):
    tf.keras.backend.clear_session()

    class_count_np = np.bincount(
        fold_arrays["y_train"].astype("int32"),
        minlength=K,
    ).astype("float32")

    class_count = tf.convert_to_tensor(class_count_np, dtype=tf.float32)

    model = build_pacbm_model(
        M,
        K,
        backbone,
        img_size,
        class_count,
        dataset_name=dataset,
        weights=backbone_weights,
        train_backbone=train_backbone,
        use_ema_prior=True,
        ema_momentum=0.9,
    )

    load_mode = load_weights_strict(model, weights_path, "pacbm")

    ds = make_image_ds(
        fold_arrays["X_test"],
        fold_arrays["y_test"],
        fold_arrays["a_test"],
        batch_size,
        backbone,
        mode="pacbm",
    )

    y_true_all, a_true_all = [], []
    concept_all, logits_all, probs_all = [], [], []
    P_pred_batches, P_true_batches = [], []

    for xb, yb, ab in ds:
        (
            _,
            concepts_pred,
            (P_pred_used, P_true_batch),
            class_logits,
            class_probs,
        ) = model([xb, yb, ab], training=False)

        y_true_all.append(yb.numpy())
        a_true_all.append(ab.numpy())
        concept_all.append(concepts_pred.numpy())
        logits_all.append(class_logits.numpy())
        probs_all.append(class_probs.numpy())
        P_pred_batches.append(P_pred_used.numpy())
        P_true_batches.append(P_true_batch.numpy())

    y_true = np.concatenate(y_true_all, axis=0).astype("int32")
    a_true = np.concatenate(a_true_all, axis=0).astype("float32")
    concept_probs = np.concatenate(concept_all, axis=0).astype("float32")
    class_logits = np.concatenate(logits_all, axis=0).astype("float32")
    class_probs = np.concatenate(probs_all, axis=0).astype("float32")
    y_pred = np.argmax(class_probs, axis=1).astype("int32")

    extras = {
        "class_count": class_count_np.astype("float32"),
        "class_logits": class_logits,
        "gamma": model.class_layer.gamma.numpy().astype("float32"),
        "bias": model.class_layer.bias.numpy().astype("float32"),
        "anchor_alpha": np.array(model.anchor_alpha.numpy()).astype("float32"),
        "P_pred_ema": model.concept_class_priors.P_pred_ema.numpy().astype("float32"),
        "P_pred_used_batches": np.stack(P_pred_batches, axis=0).astype("float32"),
        "P_true_batch_batches": np.stack(P_true_batches, axis=0).astype("float32"),
    }

    return {
        "load_mode": load_mode,
        "arrays": {
            "y_true": y_true,
            "y_pred": y_pred,
            "a_true": a_true,
            "concept_probs": concept_probs,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "y_transparent": class_probs,
            "test_idx": fold_arrays["test_idx"]
            if fold_arrays["test_idx"] is not None
            else np.array([], dtype="int32"),
        },
        "extras": extras,
    }


# =============================================================================
# Main loop
# =============================================================================

def run_one(
    model_name: str,
    dataset: str,
    backbone: str,
    fold_id: int,
    args,
    data_cache: DatasetCache,
) -> dict:
    trained_root = Path(args.trained_root)

    fold_dir, source = find_fold_dir(
        trained_root,
        model_name,
        dataset,
        backbone,
        fold_id,
    )

    weights_path, weights_kind = weight_path_for_fold_dir(fold_dir)

    if weights_path is None:
        raise FileNotFoundError(f"No weights found in {fold_dir}")

    config_parts = config_parts_from_fold_dir(model_name, fold_dir)
    train_config = config_parts[0]

    epochs, batch_size_from_config, optimizer_name, lr = parse_train_config(train_config)

    batch_size = args.batch_size if args.batch_size is not None else batch_size_from_config

    img_size = args.img_size

    if backbone == "inceptionv3" and args.inception_img_size is not None:
        img_size = args.inception_img_size

    data_obj = data_cache.load(dataset)

    K = int(data_obj["K"])
    M = int(data_obj["M"])

    fold_arrays = prepare_fold_arrays(data_obj, fold_id, img_size)

    backbone_weights = None if args.weights == "None" else "imagenet"

    label = f"{model_name} {dataset}/{backbone} fold{fold_id}"

    if model_name == "cnn":
        result = recompute_cnn(
            fold_arrays,
            K,
            backbone,
            img_size,
            batch_size,
            weights_path,
            backbone_weights,
            args.train_backbone,
        )

    elif model_name == "jointcbm":
        result = recompute_jointcbm(
            fold_arrays,
            M,
            K,
            backbone,
            img_size,
            batch_size,
            weights_path,
            backbone_weights,
            args.train_backbone,
        )

    elif model_name == "klcbm":
        result = recompute_klcbm(
            fold_arrays,
            M,
            K,
            backbone,
            img_size,
            batch_size,
            weights_path,
            backbone_weights,
            args.train_backbone,
        )

    elif model_name in ["pacbm", "pacbm_2"]:
        result = recompute_pacbm(
            fold_arrays,
            M,
            K,
            backbone,
            img_size,
            batch_size,
            weights_path,
            dataset,
            backbone_weights,
            args.train_backbone,
        )

    else:
        raise ValueError(model_name)

    arrays = result["arrays"]
    extras = result["extras"]

    y_true = arrays["y_true"]
    y_pred = arrays["y_pred"]

    acc = float(np.mean(y_true == y_pred))
    bal_acc = balanced_accuracy(y_true, y_pred, K)

    out_dir = (
        Path(args.out_dir)
        / model_name
        / dataset
        / backbone
        / ("__".join(config_parts))
        / f"fold{fold_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_payload = {}
    npz_payload.update(arrays)
    npz_payload.update(extras)

    np.savez_compressed(out_dir / "test_vectors.npz", **npz_payload)

    meta = {
        "model": model_name,
        "dataset": dataset,
        "backbone": backbone,
        "fold": fold_id,
        "K": K,
        "M": M,
        "n_test": int(len(y_true)),
        "config_parts": config_parts,
        "train_config": train_config,
        "epochs_from_config": epochs,
        "batch_size_from_config": batch_size_from_config,
        "batch_size_used": batch_size,
        "optimizer_from_config": optimizer_name,
        "lr_from_config": lr,
        "source": source,
        "fold_dir": str(fold_dir),
        "weights_path": str(weights_path),
        "weights_kind": weights_kind,
        "load_mode": result["load_mode"],
        "test_accuracy_from_vectors": acc,
        "test_balanced_accuracy_from_vectors": bal_acc,
        "saved_npz": str(out_dir / "test_vectors.npz"),
    }

    save_json(out_dir / "metadata.json", meta)

    print(
        f"[OK] {label:<35} source={source:<35} "
        f"weights={weights_kind:<16} load={result['load_mode']:<16} "
        f"acc={acc:.4f} bal={bal_acc:.4f}"
    )

    del result
    tf.keras.backend.clear_session()
    gc.collect()

    return meta


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", nargs="+", default=["pacbm"], choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)

    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--only_fold", type=int, default=None)

    parser.add_argument("--trained_root", type=str, default="trained_models")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="recomputed_test_vectors_native/")

    parser.add_argument("--scratch_dir", type=str, default="_recompute_scratch/2/")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--inception_img_size", type=int, default=None)

    parser.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "None"])

    parser.add_argument("--train_backbone", action="store_true", default=True)
    parser.add_argument("--no_train_backbone", dest="train_backbone", action="store_false")

    args = parser.parse_args()

    setup_gpu()
    set_all_seeds(SEED)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)

    folds = [args.only_fold] if args.only_fold is not None else list(range(1, args.runs + 1))

    data_cache = DatasetCache(
        data_dir=args.data_dir,
        scratch_dir=args.scratch_dir,
        runs=args.runs,
        val_split=args.val_split,
        batch_size=args.batch_size or 64,
    )

    summary = []

    for model_name in args.models:
        for dataset in args.datasets:
            for backbone in args.backbones:
                for fold_id in folds:
                    try:
                        meta = run_one(
                            model_name=model_name,
                            dataset=dataset,
                            backbone=backbone,
                            fold_id=fold_id,
                            args=args,
                            data_cache=data_cache,
                        )
                        summary.append(meta)

                    except Exception as e:
                        err = {
                            "model": model_name,
                            "dataset": dataset,
                            "backbone": backbone,
                            "fold": fold_id,
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }

                        summary.append(err)

                        print(
                            f"[ERROR] {model_name} {dataset}/{backbone} fold{fold_id}: "
                            f"{type(e).__name__}: {e}"
                        )

                    tf.keras.backend.clear_session()
                    gc.collect()

    summary_path = Path(args.out_dir) / "recompute_summary.json"
    save_json(summary_path, summary)

    print(f"\n[DONE] Saved vectors under: {args.out_dir}")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()