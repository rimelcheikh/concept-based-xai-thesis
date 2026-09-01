"""
Create a qualitative concept-intervention figure from one test_vectors.npz file.

The intervention replaces exactly one predicted concept value by its
annotation and recomputes class probabilities using the saved concept-to-class
classifier.

Supported classifiers:
  - PACBM: gamma (+ optional bias), using concept log-odds;
  - JointCBM / recomputable KL-CBM: concept_to_class_kernel
    (+ optional concept_to_class_bias), using concept probabilities directly.
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

try:
    from data_loaders.data_loader import get_dataset_loaders
    from data_loaders.configs import dataset_config
    from data_loaders import cub_loader
except Exception:
    get_dataset_loaders = None
    dataset_config = None
    cub_loader = None


PACBM_CL_LOGIT_CLIP = 1e-2
PACBM_CO_LOGIT_CLIP = 1e-6

ACTIVE_PACBM_LOGIT_CLIP = PACBM_CL_LOGIT_CLIP

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Manuscript figure palette. The blue/orange pairing is kept for
# before/after probabilities, while the remaining tones are neutral.
COLOR_BEFORE = "#2F6FA5"
COLOR_AFTER = "#E67E22"
COLOR_TEXT = "#202733"
COLOR_MUTED = "#667085"
COLOR_BORDER = "#C9CED6"
COLOR_CARD = "#F8FAFC"
COLOR_GRID = "#D9DEE5"
COLOR_SUCCESS_BG = "#EDF7EE"
COLOR_SUCCESS_EDGE = "#6A9D75"
COLOR_SUCCESS_TEXT = "#285C34"
COLOR_INFO_BG = "#EEF4FA"
COLOR_INFO_EDGE = "#7398B8"
COLOR_INFO_TEXT = "#315A78"

plt.rcParams.update(
    {
        "font.size": 11.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#7A828C",
        "axes.linewidth": 0.8,
        "savefig.dpi": 350,
    }
)

# ---------------------------------------------------------------------
# Softer manuscript palette
# ---------------------------------------------------------------------

TEXT_COLOR = "#30343B"
MUTED_TEXT_COLOR = "#727984"
SUBTLE_LINE_COLOR = "#D9DDE2"
GRID_COLOR = "#E5E8EB"

# Muted slate and terracotta instead of saturated blue/orange.
BEFORE_COLOR = "#667C91"
AFTER_COLOR = "#B9856B"

CORRECT_COLOR = "#58735E"
CORRECT_BACKGROUND = "#F1F5F1"

INCREASE_COLOR = "#60778D"
INCREASE_BACKGROUND = "#F1F4F7"

IMAGE_BORDER_COLOR = "#D5D9DE"

SELECTED_BACKBONES = {
    "AwA2": "efficientnetb0",
    "aPY": "efficientnetb0",
    "CUB": "inceptionv3",
}

MODEL_ALIASES = {
    "pacbm": "pacbm",
    "pacbm-cl": "pacbm",
    "pacbm_cl": "pacbm",
    "pacbmcl": "pacbm",
    "pacbm_2": "pacbm_2",
    "pacbm-co": "pacbm_2",
    "pacbm_co": "pacbm_2",
    "pacbmco": "pacbm_2",
    "jointcbm": "jointcbm",
    "joint_cbm": "jointcbm",
    "klcbm": "klcbm",
    "kl-cbm": "klcbm",
}

MODEL_DISPLAY_NAMES = {
    "pacbm": "PACBM-Cl",
    "pacbm_2": "PACBM-Co",
    "jointcbm": "JointCBM",
    "klcbm": "KL-CBM",
}


DEFAULT_DATASETS = ["AwA2", "aPY", "CUB"]
DEFAULT_VECTOR_ROOTS = [
    Path("recomputed_test_vectors_native_20260810"),
    Path("recomputed_test_vectors/2"),
]


class DatasetInfo:
    def __init__(
        self,
        dataset: str,
        images_all: Optional[np.ndarray],
        labels_all: Optional[np.ndarray],
        test_images: Optional[np.ndarray],
        test_labels: Optional[np.ndarray],
        class_names: Sequence[str],
        concept_names: Sequence[str],
    ):
        self.dataset = dataset
        self.images_all = images_all
        self.labels_all = labels_all
        self.test_images = test_images
        self.test_labels = test_labels
        self.class_names = [pretty_name(x) for x in class_names]
        self.concept_names = [pretty_name(x) for x in concept_names]

def display_concept_name(name: str) -> str:
    """Replace abbreviated dataset names only for visual display."""
    replacements = {
        "Furn. Back": "Furniture backrest",
        "Furn. Seat": "Furniture seat",
        "Furn. Arm": "Furniture armrest",
        "row window": "row of windows",
    }
    return replacements.get(name, name)


def display_backbone_name(name: str) -> str:
    replacements = {
        "efficientnetb0": "EfficientNetB0",
        "inceptionv3": "InceptionV3",
        "mobilenetv2": "MobileNetV2",
    }
    return replacements.get(name.lower(), name)


def wrap_display_text(value: str, width: int = 28) -> str:
    return textwrap.fill(str(value), width=width)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def loader_image_to_display(value: Any) -> np.ndarray:
    """
    Convert raw or ImageNet-normalized loader output to an RGB image.
    """
    image = to_numpy(value)
    image = np.squeeze(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.ndim != 3:
        raise ValueError(f"Unsupported loader image shape: {image.shape}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = image.astype(np.float32)

    if np.nanmin(image) < -0.05:
        image = image * IMAGENET_STD + IMAGENET_MEAN
    elif np.nanmax(image) > 1.5:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def numpy_from_loader(loader: Any) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    annotations: List[np.ndarray] = []

    for batch in loader:
        if len(batch) == 3:
            image, label, concepts = batch
            annotations.append(to_numpy(concepts))
        elif len(batch) == 2:
            image, label = batch
        else:
            raise ValueError(
                "Expected loader batches containing either "
                "(image, label) or (image, label, concepts)."
            )

        images.append(to_numpy(image))
        labels.append(to_numpy(label))

    x = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0).reshape(-1).astype(int)
    a = (
        np.concatenate(annotations, axis=0).astype(np.float32)
        if annotations
        else None
    )
    return x, y, a


def safe_names(
    values: Any,
    count: int,
    prefix: str,
) -> List[str]:
    if values is None:
        return [f"{prefix} {index}" for index in range(count)]

    if hasattr(values, "values"):
        values = values.values

    names = [pretty_name(value) for value in np.asarray(values).reshape(-1)]

    if len(names) < count:
        names.extend(
            f"{prefix} {index}"
            for index in range(len(names), count)
        )

    return names[:count]


def class_names_from_mapping(
    classes: Any,
    label_to_idx: Any,
    count: int,
) -> List[str]:
    names: List[Optional[str]] = [None] * count

    if isinstance(label_to_idx, dict):
        for class_name, index in label_to_idx.items():
            try:
                index = int(index)
            except Exception:
                continue
            if 0 <= index < count:
                names[index] = pretty_name(class_name)

    fallback = safe_names(classes, count, "Class")

    return [
        name if name is not None else fallback[index]
        for index, name in enumerate(names)
    ]


def cub_concept_names(
    count: int,
    raw_concepts: Any = None,
) -> List[str]:
    if cub_loader is None:
        return [f"Concept {index}" for index in range(count)]

    selected = None

    if raw_concepts is not None:
        try:
            raw = list(raw_concepts)
            if (
                len(raw) == count
                and all(isinstance(x, (int, np.integer)) for x in raw)
            ):
                selected = raw
        except Exception:
            selected = None

    if selected is None and hasattr(cub_loader, "SELECTED_CONCEPTS"):
        selected = list(cub_loader.SELECTED_CONCEPTS)

    semantics = getattr(cub_loader, "CONCEPT_SEMANTICS", None)

    if selected is not None and semantics is not None:
        names: List[str] = []

        for selected_index in selected[:count]:
            selected_index = int(selected_index)
            if 0 <= selected_index < len(semantics):
                names.append(pretty_name(semantics[selected_index]))
            else:
                names.append(f"Concept {selected_index}")

        if len(names) < count:
            names.extend(
                f"Concept {index}"
                for index in range(len(names), count)
            )

        return names[:count]

    return [f"Concept {index}" for index in range(count)]


def load_dataset_info(
    dataset: str,
    data_dir: str,
    scratch_dir: str,
    batch_size: int,
    seed: int,
) -> DatasetInfo:
    """
    Load images and semantic names using the same project loaders used by the
    dataset-example figure. No separate image-path or name files are needed.
    """
    if get_dataset_loaders is None or dataset_config is None:
        raise ImportError(
            "Could not import data_loaders. Run this script from the project "
            "root, where data_loaders/ is importable."
        )

    if dataset not in dataset_config:
        raise KeyError(f"No dataset_config entry found for {dataset}.")

    config = dataset_config[dataset].copy()
    config["data_dir"] = data_dir
    config["seed"] = seed
    config["batch_size"] = batch_size

    save_dir = str(Path(scratch_dir) / dataset)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if dataset in ("AwA2", "aPY"):
        result = get_dataset_loaders(
            f"{dataset}_cv",
            config,
            seed,
            save_dir,
            batch_size,
            data_dir,
        )

        (
            images,
            _,
            labels,
            _,
            annotations,
            _,
            _,
            classes,
            concepts,
            label_to_idx,
            _,
            _,
            _,
        ) = result

        images = np.asarray(images)
        labels = np.asarray(labels).reshape(-1).astype(int)
        annotations = np.asarray(annotations)

        n_classes = int(np.max(labels)) + 1
        n_concepts = int(annotations.shape[1])

        class_names = class_names_from_mapping(
            classes,
            label_to_idx,
            n_classes,
        )
        concept_names = safe_names(
            concepts,
            n_concepts,
            "Concept",
        )

        return DatasetInfo(
            dataset=dataset,
            images_all=images,
            labels_all=labels,
            test_images=None,
            test_labels=None,
            class_names=class_names,
            concept_names=concept_names,
        )

    if dataset == "CUB":
        cub_root = Path(data_dir) / "CUB_200_2011"
        if cub_root.exists():
            config["root_dir"] = str(cub_root)

        result = get_dataset_loaders(
            "CUB",
            config,
            seed,
            save_dir,
            batch_size,
            data_dir,
        )

        (
            _,
            _,
            test_loader,
            _,
            concepts,
            classes,
            _,
            label_to_idx,
        ) = result

        test_images, test_labels, test_annotations = numpy_from_loader(
            test_loader
        )

        n_classes = int(np.max(test_labels)) + 1
        n_concepts = int(test_annotations.shape[1])

        if cub_loader is not None and hasattr(cub_loader, "CLASS_NAMES"):
            class_names = safe_names(
                cub_loader.CLASS_NAMES,
                n_classes,
                "Class",
            )
        else:
            class_names = class_names_from_mapping(
                classes,
                label_to_idx,
                n_classes,
            )

        concept_names = cub_concept_names(n_concepts, concepts)

        return DatasetInfo(
            dataset=dataset,
            images_all=None,
            labels_all=None,
            test_images=test_images,
            test_labels=test_labels,
            class_names=class_names,
            concept_names=concept_names,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def image_from_dataset_info(
    info: DatasetInfo,
    data: np.lib.npyio.NpzFile,
    sample_index: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Recover the exact displayed sample and verify its label.

    AwA2/aPY use test_idx saved in test_vectors.npz.
    CUB uses the official test-loader order.
    """
    y_true = int(data["y_true"][sample_index])

    if info.dataset in ("AwA2", "aPY"):
        test_indices = safe_get(data, "test_idx")

        if test_indices is None or sample_index >= len(test_indices):
            return None, "test_idx is missing from test_vectors.npz"

        original_index = int(test_indices[sample_index])

        if info.images_all is None or info.labels_all is None:
            return None, "full dataset arrays were not loaded"

        if not 0 <= original_index < len(info.images_all):
            return None, f"test_idx {original_index} is outside the dataset"

        loader_label = int(info.labels_all[original_index])

        if loader_label != y_true:
            return (
                None,
                f"label mismatch: loader={loader_label}, npz={y_true}, "
                f"test_idx={original_index}",
            )

        return loader_image_to_display(info.images_all[original_index]), None

    if info.dataset == "CUB":
        if info.test_images is None or info.test_labels is None:
            return None, "CUB test arrays were not loaded"

        if not 0 <= sample_index < len(info.test_images):
            return None, f"sample index {sample_index} is outside CUB test data"

        loader_label = int(info.test_labels[sample_index])

        if loader_label != y_true:
            return (
                None,
                f"label mismatch: loader={loader_label}, npz={y_true}, "
                f"row={sample_index}",
            )

        return loader_image_to_display(info.test_images[sample_index]), None

    return None, f"unsupported dataset: {info.dataset}"


def select_npz_across_default_roots(
    roots: Sequence[Path],
    dataset: str,
    model: str,
    backbone: str,
    config: Optional[str],
    fold: Optional[str],
    csv_paths: Sequence[str],
    minimum_agreement: float,
) -> Tuple[Path, Dict[str, Any]]:
    errors: List[str] = []

    for root in roots:
        try:
            return select_npz_automatically(
                root=root,
                dataset=dataset,
                model=model,
                backbone=backbone,
                config=config,
                fold=fold,
                csv_paths=csv_paths,
                minimum_agreement=minimum_agreement,
            )
        except (FileNotFoundError, RuntimeError) as error:
            errors.append(f"{root}: {error}")

    raise FileNotFoundError(
        "No valid test_vectors.npz was found in the candidate roots:\n"
        + "\n".join(errors)
    )


def normalize_model_key(value: str) -> str:
    key = str(value).strip().lower()
    return MODEL_ALIASES.get(key, key)


def infer_path_metadata(path: Path, root: Path) -> Dict[str, str]:
    """
    Expected layout:
      ROOT/model/dataset/backbone/config/fold/test_vectors.npz

    Extra folders between backbone and fold are joined into the config name.
    """
    rel = path.relative_to(root)
    parts = rel.parts

    metadata = {
        "model": "unknown",
        "dataset": "unknown",
        "backbone": "unknown",
        "config": "unknown",
        "fold": "unknown",
    }

    if len(parts) < 4:
        return metadata

    metadata["model"] = normalize_model_key(parts[0])
    metadata["dataset"] = parts[1]
    metadata["backbone"] = parts[2].lower()

    fold_index = None
    for index, part in enumerate(parts[:-1]):
        lowered = part.lower()
        if lowered.startswith("fold") or lowered.startswith("run"):
            fold_index = index
            break

    if fold_index is not None:
        metadata["fold"] = parts[fold_index]
        config_parts = parts[3:fold_index]
    else:
        config_parts = parts[3:-1]

    if config_parts:
        metadata["config"] = "/".join(config_parts)

    return metadata


def config_names_from_plot_csvs(
    csv_paths: Sequence[str],
    dataset: str,
    backbone: str,
    model: str,
) -> List[str]:
    """
    Recover the configuration names represented in the same aggregated CSVs
    used by the intervention-curve plotting script.
    """
    frames = []

    for raw_path in csv_paths:
        csv_path = Path(raw_path)
        if not csv_path.exists():
            print(f"[warn] configuration CSV not found: {csv_path}")
            continue

        try:
            frame = __import__("pandas").read_csv(csv_path)
        except Exception as exc:
            print(f"[warn] could not read {csv_path}: {exc}")
            continue

        required = {"dataset", "backbone", "model", "config"}
        if not required.issubset(frame.columns):
            print(
                f"[warn] {csv_path} does not contain all of "
                f"{sorted(required)}"
            )
            continue

        frame = frame.copy()
        frame["_model_key"] = frame["model"].map(normalize_model_key)
        frame["_backbone_key"] = frame["backbone"].astype(str).str.lower()

        subset = frame[
            (frame["dataset"].astype(str) == dataset)
            & (frame["_backbone_key"] == backbone.lower())
            & (frame["_model_key"] == model)
        ]

        if "intervention" in subset.columns:
            baseline = subset[
                subset["intervention"].astype(str) == "baseline_recomputed"
            ]
            if not baseline.empty:
                subset = baseline

        frames.append(subset)

    if not frames:
        return []

    combined = __import__("pandas").concat(frames, ignore_index=True)
    if combined.empty:
        return []

    return [
        str(value)
        for value in combined["config"].dropna().astype(str).unique()
    ]


def inspect_candidate_npz(path: Path) -> Dict[str, float]:
    data = np.load(path, allow_pickle=True)

    y_true = safe_get(data, "y_true")
    concepts = safe_get(data, "concept_probs")

    if y_true is None or concepts is None:
        raise ValueError("missing y_true or concept_probs")

    y_true = np.asarray(y_true).astype(int)
    concepts = np.asarray(concepts, dtype=float)

    predictor_type, weights, bias = detect_predictor(data)

    if weights.shape[0] != concepts.shape[1]:
        raise ValueError(
            f"incompatible classifier/concept shapes: "
            f"{weights.shape} and {concepts.shape}"
        )

    logits = recompute_logits(
        concepts=concepts,
        predictor_type=predictor_type,
        weights=weights,
        bias=bias,
    )
    probabilities = softmax(logits, axis=1)
    predictions = np.argmax(probabilities, axis=1)

    saved_predictions, _ = get_saved_predictions(data)
    if saved_predictions is None:
        agreement = 1.0
    else:
        agreement = float(np.mean(saved_predictions == predictions))

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "accuracy": float(np.mean(y_true == predictions)),
        "agreement": agreement,
    }


def select_npz_automatically(
    root: Path,
    dataset: str,
    model: str,
    backbone: str,
    config: Optional[str],
    fold: Optional[str],
    csv_paths: Sequence[str],
    minimum_agreement: float,
) -> Tuple[Path, Dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(f"NPZ root does not exist: {root}")

    model = normalize_model_key(model)
    backbone = backbone.lower()

    candidates: List[Dict[str, Any]] = []

    for npz_path in sorted(root.rglob("test_vectors.npz")):
        metadata = infer_path_metadata(npz_path, root)

        if metadata["model"] != model:
            continue
        if metadata["dataset"].lower() != dataset.lower():
            continue
        if metadata["backbone"] != backbone:
            continue
        if config is not None and metadata["config"] != config:
            continue
        if fold is not None and metadata["fold"].lower() != fold.lower():
            continue

        try:
            metrics = inspect_candidate_npz(npz_path)
        except Exception as exc:
            print(f"[skip] {npz_path}: {exc}")
            continue

        candidates.append(
            {
                "path": npz_path,
                **metadata,
                **metrics,
            }
        )

    if not candidates:
        raise FileNotFoundError(
            "No valid test_vectors.npz matched "
            f"model={model}, dataset={dataset}, backbone={backbone}, "
            f"config={config or 'auto'}, fold={fold or 'auto'} under {root}."
        )

    csv_configs: List[str] = []
    if config is None:
        csv_configs = config_names_from_plot_csvs(
            csv_paths=csv_paths,
            dataset=dataset,
            backbone=backbone,
            model=model,
        )

        matching_csv_configs = [
            candidate
            for candidate in candidates
            if candidate["config"] in csv_configs
        ]

        if matching_csv_configs:
            candidates = matching_csv_configs
            print(
                "[select] restricted candidates to configurations found "
                "in the intervention-curve CSVs: "
                + ", ".join(sorted(set(csv_configs)))
            )
        elif csv_configs:
            print(
                "[warn] configurations from the curve CSVs were not found "
                "under the supplied NPZ root; using discovered configurations."
            )

    valid = [
        candidate
        for candidate in candidates
        if candidate["agreement"] >= minimum_agreement
    ]

    if not valid:
        ranked = sorted(
            candidates,
            key=lambda item: item["agreement"],
            reverse=True,
        )
        summary = "\n".join(
            f"  {item['path']}: agreement={item['agreement']:.4f}"
            for item in ranked[:10]
        )
        raise RuntimeError(
            "No candidate reached the minimum saved/recomputed prediction "
            f"agreement of {minimum_agreement:.4f}.\n{summary}"
        )

    candidates = valid

    # Configuration selection deliberately avoids choosing the configuration
    # with the best test performance. Prefer the configuration represented by
    # the largest number of valid folds/runs; use its name only as a tie-break.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["config"], []).append(candidate)

    selected_config = sorted(
        grouped,
        key=lambda name: (-len(grouped[name]), name),
    )[0]
    config_candidates = grouped[selected_config]

    # Use a representative fold/run: the one whose balanced accuracy is
    # closest to the mean for the selected configuration, rather than the best
    # fold.
    mean_balanced_accuracy = float(
        np.mean(
            [
                candidate["balanced_accuracy"]
                for candidate in config_candidates
            ]
        )
    )

    selected = sorted(
        config_candidates,
        key=lambda candidate: (
            abs(
                candidate["balanced_accuracy"]
                - mean_balanced_accuracy
            ),
            -candidate["agreement"],
            candidate["fold"],
        ),
    )[0]

    report = {
        "root": str(root),
        "dataset": selected["dataset"],
        "model": selected["model"],
        "model_display": MODEL_DISPLAY_NAMES.get(
            selected["model"], selected["model"]
        ),
        "backbone": selected["backbone"],
        "config": selected["config"],
        "fold": selected["fold"],
        "npz": str(selected["path"]),
        "recomputed_balanced_accuracy": selected["balanced_accuracy"],
        "recomputed_accuracy": selected["accuracy"],
        "saved_recomputed_agreement": selected["agreement"],
        "configuration_mean_balanced_accuracy": mean_balanced_accuracy,
        "number_of_valid_files_for_configuration": len(config_candidates),
        "configs_found_in_curve_csvs": csv_configs,
    }

    return selected["path"], report


def safe_get(data: np.lib.npyio.NpzFile, *keys: str) -> Optional[np.ndarray]:
    for key in keys:
        if key in data.files:
            return data[key]
    return None


def logit(
    values: np.ndarray,
    eps: Optional[float] = None,
) -> np.ndarray:
    if eps is None:
        eps = ACTIVE_PACBM_LOGIT_CLIP

    values = np.clip(values, eps, 1.0 - eps)
    return np.log(values / (1.0 - values))


def decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def pretty_name(name: Any) -> str:
    text = decode_scalar(name).strip()
    text = text.replace("+", " ").replace("_", " ")
    text = text.replace("has ", "")
    text = text.replace("::", ": ")
    text = " ".join(text.split())

    replacements = {
        "oldworld": "old world",
        "Black footed Albatross": "Black-footed Albatross",
        "Furn. Back": "Furniture backrest",
        "Furn Back": "Furniture backrest",
        "tvmonitor": "TV monitor",
        "pottedplant": "potted plant",
    }
    return replacements.get(text, text)


def display_backbone_name(name: str) -> str:
    key = str(name).strip().lower()
    return {
        "efficientnetb0": "EfficientNetB0",
        "inceptionv3": "InceptionV3",
        "mobilenetv2": "MobileNetV2",
    }.get(key, str(name))


def wrap_label(text: str, width: int = 22) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False)


def parse_names_file(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Names file not found: {file_path}")

    names: List[str] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            line = parts[1]

        names.append(pretty_name(line))

    return names


def names_from_npz(
    data: np.lib.npyio.NpzFile,
    keys: Sequence[str],
) -> Optional[List[str]]:
    values = safe_get(data, *keys)
    if values is None:
        return None

    values = np.asarray(values)

    if values.shape == () and values.dtype == object:
        obj = values.item()
        if isinstance(obj, dict):
            try:
                ordered = [obj[k] for k in sorted(obj)]
                return [pretty_name(v) for v in ordered]
            except Exception:
                return [pretty_name(v) for v in obj.values()]

    return [pretty_name(v) for v in values.reshape(-1)]


def resolve_names(
    external_path: Optional[str],
    data: np.lib.npyio.NpzFile,
    npz_keys: Sequence[str],
    count: int,
    prefix: str,
    fallback_names: Optional[Sequence[str]] = None,
) -> List[str]:
    names = parse_names_file(external_path)

    if names is None:
        names = names_from_npz(data, npz_keys)

    if names is None and fallback_names is not None:
        names = [pretty_name(value) for value in fallback_names]

    if names is None:
        return [f"{prefix} {i}" for i in range(count)]

    if len(names) < count:
        names = names + [f"{prefix} {i}" for i in range(len(names), count)]

    return names[:count]


def detect_predictor(
    data: np.lib.npyio.NpzFile,
) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    gamma = safe_get(data, "gamma")
    bias = safe_get(data, "bias")

    if gamma is not None:
        return (
            "pacbm",
            np.asarray(gamma, dtype=float),
            None if bias is None else np.asarray(bias, dtype=float),
        )

    kernel = safe_get(data, "concept_to_class_kernel")
    classifier_bias = safe_get(data, "concept_to_class_bias")

    if kernel is not None:
        return (
            "linear_cbm",
            np.asarray(kernel, dtype=float),
            None
            if classifier_bias is None
            else np.asarray(classifier_bias, dtype=float),
        )

    raise ValueError(
        "The NPZ does not contain a recomputable concept classifier. "
        "Expected gamma/bias for PACBM or "
        "concept_to_class_kernel/concept_to_class_bias for a linear CBM."
    )


def concept_evidence(concepts: np.ndarray, predictor_type: str) -> np.ndarray:
    if predictor_type == "pacbm":
        return logit(concepts)
    return concepts


def recompute_logits(
    concepts: np.ndarray,
    predictor_type: str,
    weights: np.ndarray,
    bias: Optional[np.ndarray],
) -> np.ndarray:
    evidence = concept_evidence(concepts, predictor_type)
    logits = evidence @ weights
    if bias is not None:
        logits = logits + bias
    return logits


def get_saved_predictions(
    data: np.lib.npyio.NpzFile,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    y_pred = safe_get(data, "y_pred")
    class_probs = safe_get(data, "class_probs")

    if y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int)

    if class_probs is not None:
        class_probs = np.asarray(class_probs, dtype=float)

    return y_pred, class_probs


def display_image_from_array(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    image = np.squeeze(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = image.astype(np.float32)

    if image.min() < -0.05:
        image = image * IMAGENET_STD + IMAGENET_MEAN

    if image.max() > 1.5:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def read_image_paths_file(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image-path file not found: {file_path}")

    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def get_image_for_sample(
    data: np.lib.npyio.NpzFile,
    sample_index: int,
    image_paths: Optional[Sequence[str]],
    image_root: Optional[str],
    explicit_image: Optional[str],
) -> Optional[np.ndarray]:
    if explicit_image is not None:
        return np.asarray(Image.open(explicit_image).convert("RGB"))

    images = safe_get(data, "images", "x", "inputs", "input_images")
    if images is not None and sample_index < len(images):
        try:
            return display_image_from_array(images[sample_index])
        except Exception:
            pass

    paths: Optional[Sequence[Any]] = image_paths
    if paths is None:
        stored_paths = safe_get(
            data,
            "image_paths",
            "paths",
            "filenames",
            "image_files",
        )
        if stored_paths is not None:
            paths = np.asarray(stored_paths).reshape(-1)

    if paths is None or sample_index >= len(paths):
        return None

    raw_path = decode_scalar(paths[sample_index])
    path = Path(raw_path)

    if not path.is_absolute() and image_root is not None:
        path = Path(image_root) / path

    if not path.exists():
        return None

    return np.asarray(Image.open(path).convert("RGB"))


def single_concept_outcomes(
    sample_index: int,
    concepts: np.ndarray,
    annotations: np.ndarray,
    y_true: np.ndarray,
    base_logits: np.ndarray,
    base_probs: np.ndarray,
    predictor_type: str,
    weights: np.ndarray,
    min_concept_change: float,
) -> Dict[str, np.ndarray]:
    old_values = concepts[sample_index]
    new_values = annotations[sample_index]

    changed = np.abs(new_values - old_values) >= min_concept_change
    concept_indices = np.flatnonzero(changed)

    if len(concept_indices) == 0:
        return {
            "concept_indices": np.array([], dtype=int),
            "new_probs": np.empty((0, base_probs.shape[1])),
            "new_predictions": np.array([], dtype=int),
            "delta_true_probability": np.array([], dtype=float),
        }

    if predictor_type == "pacbm":
        old_evidence = logit(old_values[concept_indices])
        new_evidence = logit(new_values[concept_indices])
    else:
        old_evidence = old_values[concept_indices]
        new_evidence = new_values[concept_indices]

    evidence_delta = new_evidence - old_evidence

    candidate_logits = (
        base_logits[sample_index][None, :]
        + evidence_delta[:, None] * weights[concept_indices, :]
    )
    candidate_probs = softmax(candidate_logits, axis=1)
    candidate_predictions = np.argmax(candidate_probs, axis=1)

    true_class = int(y_true[sample_index])
    delta_true_probability = (
        candidate_probs[:, true_class] - base_probs[sample_index, true_class]
    )

    return {
        "concept_indices": concept_indices,
        "new_probs": candidate_probs,
        "new_predictions": candidate_predictions,
        "delta_true_probability": delta_true_probability,
    }


def case_record(
    sample_index: int,
    concept_index: int,
    new_probs: np.ndarray,
    new_prediction: int,
    concepts: np.ndarray,
    annotations: np.ndarray,
    y_true: np.ndarray,
    base_probs: np.ndarray,
    base_predictions: np.ndarray,
    concept_names: Sequence[str],
    class_names: Sequence[str],
    case_kind: str,
) -> Dict[str, Any]:
    true_class = int(y_true[sample_index])
    before_class = int(base_predictions[sample_index])
    after_class = int(new_prediction)

    delta_true = float(
        new_probs[true_class] - base_probs[sample_index, true_class]
    )

    if before_class != true_class and after_class == true_class:
        outcome = "Prediction corrected"
    elif before_class == true_class and after_class != true_class:
        outcome = "Correct prediction degraded"
    elif delta_true > 0:
        outcome = "True-class confidence increased"
    elif delta_true < 0:
        outcome = "True-class confidence decreased"
    else:
        outcome = "No change in true-class confidence"

    return {
        "kind": case_kind,
        "sample_index": int(sample_index),
        "concept_index": int(concept_index),
        "concept_name": concept_names[concept_index],
        "concept_before": float(concepts[sample_index, concept_index]),
        "concept_after": float(annotations[sample_index, concept_index]),
        "true_class_index": true_class,
        "true_class_name": class_names[true_class],
        "before_class_index": before_class,
        "before_class_name": class_names[before_class],
        "after_class_index": after_class,
        "after_class_name": class_names[after_class],
        "true_probability_before": float(base_probs[sample_index, true_class]),
        "true_probability_after": float(new_probs[true_class]),
        "true_probability_change": delta_true,
        "outcome": outcome,
        "base_probs": base_probs[sample_index],
        "new_probs": new_probs,
    }


def rank_candidate_cases(
    wanted_kind: str,
    concepts: np.ndarray,
    annotations: np.ndarray,
    y_true: np.ndarray,
    base_logits: np.ndarray,
    base_probs: np.ndarray,
    base_predictions: np.ndarray,
    predictor_type: str,
    weights: np.ndarray,
    concept_names: Sequence[str],
    class_names: Sequence[str],
    min_concept_change: float,
    max_samples: Optional[int],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Rank candidate single-concept interventions for qualitative display.

    The ranking first prioritizes corrections of originally incorrect
    predictions. Within the same outcome type, it favors a large increase in
    true-class probability and a substantial concept-value correction. The
    returned list is intended for manual inspection; the first item remains
    the automatic default.
    """
    n_samples = len(y_true)
    search_limit = n_samples if not max_samples else min(max_samples, n_samples)

    ranked: List[Tuple[float, Dict[str, Any]]] = []

    for sample_index in range(search_limit):
        outcomes = single_concept_outcomes(
            sample_index=sample_index,
            concepts=concepts,
            annotations=annotations,
            y_true=y_true,
            base_logits=base_logits,
            base_probs=base_probs,
            predictor_type=predictor_type,
            weights=weights,
            min_concept_change=min_concept_change,
        )

        concept_indices = outcomes["concept_indices"]
        if len(concept_indices) == 0:
            continue

        true_class = int(y_true[sample_index])
        before_class = int(base_predictions[sample_index])
        before_correct = before_class == true_class

        for local_index, concept_index in enumerate(concept_indices):
            after_class = int(outcomes["new_predictions"][local_index])
            after_correct = after_class == true_class
            delta_true = float(outcomes["delta_true_probability"][local_index])

            true_probability_before = float(
                base_probs[sample_index, true_class]
            )
            true_probability_after = float(
                outcomes["new_probs"][local_index, true_class]
            )

            # Avoid saturated or nearly degenerate qualitative examples.
            if true_probability_before < 0.01:
                continue

            if true_probability_after > 0.95:
                continue

            # The person class dominates many visually uninformative aPY corrections.
            if class_names[true_class].strip().lower() == "person":
                continue


            concept_change = float(
                abs(
                    annotations[sample_index, concept_index]
                    - concepts[sample_index, concept_index]
                )
            )

            

            if wanted_kind == "expected":
                if (not before_correct) and after_correct:
                    priority = 3
                elif before_correct and after_correct and delta_true > 0:
                    priority = 2
                elif delta_true > 0:
                    priority = 1
                else:
                    continue

                # Corrections dominate. Probability gain is the main
                # tie-breaker; concept change prevents visually trivial edits.
                score = priority * 100.0 + 10.0 * delta_true + concept_change

            elif wanted_kind == "unexpected":
                if before_correct and (not after_correct):
                    priority = 3
                elif delta_true < 0:
                    priority = 1
                else:
                    continue

                score = priority * 100.0 - 10.0 * delta_true + concept_change

            else:
                raise ValueError(f"Unknown case kind: {wanted_kind}")

            case = case_record(
                sample_index=sample_index,
                concept_index=int(concept_index),
                new_probs=outcomes["new_probs"][local_index],
                new_prediction=after_class,
                concepts=concepts,
                annotations=annotations,
                y_true=y_true,
                base_probs=base_probs,
                base_predictions=base_predictions,
                concept_names=concept_names,
                class_names=class_names,
                case_kind=wanted_kind,
            )
            case["selection_score"] = float(score)
            ranked.append((score, case))

    ranked.sort(
        key=lambda pair: (
            -pair[0],
            pair[1]["sample_index"],
            pair[1]["concept_index"],
        )
    )

    # Avoid showing several concept interventions from the same image in the
    # candidate sheet unless there are not enough distinct samples.
    unique: List[Dict[str, Any]] = []
    seen_samples = set()
    for _, case in ranked:
        if case["sample_index"] in seen_samples:
            continue
        unique.append(case)
        seen_samples.add(case["sample_index"])
        if len(unique) >= top_n:
            break

    if len(unique) < top_n:
        used = {(c["sample_index"], c["concept_index"]) for c in unique}
        for _, case in ranked:
            key = (case["sample_index"], case["concept_index"])
            if key in used:
                continue
            unique.append(case)
            used.add(key)
            if len(unique) >= top_n:
                break

    return unique


def select_best_case(
    wanted_kind: str,
    concepts: np.ndarray,
    annotations: np.ndarray,
    y_true: np.ndarray,
    base_logits: np.ndarray,
    base_probs: np.ndarray,
    base_predictions: np.ndarray,
    predictor_type: str,
    weights: np.ndarray,
    concept_names: Sequence[str],
    class_names: Sequence[str],
    min_concept_change: float,
    max_samples: Optional[int],
) -> Optional[Dict[str, Any]]:
    candidates = rank_candidate_cases(
        wanted_kind=wanted_kind,
        concepts=concepts,
        annotations=annotations,
        y_true=y_true,
        base_logits=base_logits,
        base_probs=base_probs,
        base_predictions=base_predictions,
        predictor_type=predictor_type,
        weights=weights,
        concept_names=concept_names,
        class_names=class_names,
        min_concept_change=min_concept_change,
        max_samples=max_samples,
        top_n=1,
    )
    return candidates[0] if candidates else None


def select_manual_case(
    sample_index: int,
    concept_index: Optional[int],
    concepts: np.ndarray,
    annotations: np.ndarray,
    y_true: np.ndarray,
    base_logits: np.ndarray,
    base_probs: np.ndarray,
    base_predictions: np.ndarray,
    predictor_type: str,
    weights: np.ndarray,
    concept_names: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    outcomes = single_concept_outcomes(
        sample_index=sample_index,
        concepts=concepts,
        annotations=annotations,
        y_true=y_true,
        base_logits=base_logits,
        base_probs=base_probs,
        predictor_type=predictor_type,
        weights=weights,
        min_concept_change=0.0,
    )

    indices = outcomes["concept_indices"]
    if len(indices) == 0:
        raise ValueError("No concepts are available for this sample.")

    if concept_index is None:
        true_class = int(y_true[sample_index])
        changes = np.abs(
            outcomes["new_probs"][:, true_class]
            - base_probs[sample_index, true_class]
        )
        local_index = int(np.argmax(changes))
        concept_index = int(indices[local_index])
    else:
        matches = np.flatnonzero(indices == concept_index)
        if len(matches) == 0:
            raise ValueError(
                f"Concept index {concept_index} is outside the valid range."
            )
        local_index = int(matches[0])

    delta = float(outcomes["delta_true_probability"][local_index])
    kind = "expected" if delta >= 0 else "unexpected"

    return case_record(
        sample_index=sample_index,
        concept_index=concept_index,
        new_probs=outcomes["new_probs"][local_index],
        new_prediction=int(outcomes["new_predictions"][local_index]),
        concepts=concepts,
        annotations=annotations,
        y_true=y_true,
        base_probs=base_probs,
        base_predictions=base_predictions,
        concept_names=concept_names,
        class_names=class_names,
        case_kind=kind,
    )


def probability_classes(
    base_probs: np.ndarray,
    new_probs: np.ndarray,
    true_class: int,
    top_n: int,
) -> np.ndarray:
    combined = np.maximum(base_probs, new_probs)
    ranked = list(np.argsort(-combined))

    selected: List[int] = []
    for index in ranked:
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) >= top_n:
            break

    if true_class not in selected:
        selected[-1] = true_class

    return np.asarray(selected, dtype=int)


def draw_case(
    axes,
    case,
    image,
    class_names,
    top_classes,
    show_legend=False,
):
    image_axis, text_axis, probability_axis = axes

    # -----------------------------
    # Input image
    # -----------------------------
    if image is not None:
        image_axis.imshow(image)
    else:
        image_axis.text(
            0.5, 0.5,
            f"Image unavailable\nsample {case['sample_index']}",
            ha="center", va="center",
            transform=image_axis.transAxes,
            color=MUTED_TEXT_COLOR,
            fontsize=10,
        )

    image_axis.set_xticks([])
    image_axis.set_yticks([])
    image_axis.set_facecolor("white")

    for spine in image_axis.spines.values():
        spine.set_visible(True)
        spine.set_color(IMAGE_BORDER_COLOR)
        spine.set_linewidth(0.8)

    # -----------------------------
    # Intervention text block
    # -----------------------------
    text_axis.set_xlim(0.0, 1.0)
    text_axis.set_ylim(0.0, 1.0)
    text_axis.axis("off")

    concept_name = display_concept_name(case["concept_name"])
    delta = case["true_probability_change"]

    if case["outcome"] == "Prediction corrected":
        outcome_text = "Prediction corrected"
        outcome_color = CORRECT_COLOR
        outcome_background = CORRECT_BACKGROUND
    elif case["outcome"] == "True-class confidence increased":
        outcome_text = "True-class confidence increased"
        outcome_color = INCREASE_COLOR
        outcome_background = INCREASE_BACKGROUND
    else:
        outcome_text = case["outcome"]
        outcome_color = MUTED_TEXT_COLOR
        outcome_background = "#F5F5F4"

    text_axis.text(
        0.02, 0.93,
        "Intervention response",
        ha="left", va="center",
        fontsize=13.0,
        fontweight="semibold",
        color=TEXT_COLOR,
        transform=text_axis.transAxes,
    )

    text_axis.text(
        0.98, 0.93,
        outcome_text,
        ha="right", va="center",
        fontsize=9.8,
        fontweight="semibold",
        color=outcome_color,
        transform=text_axis.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor=outcome_background,
            edgecolor=outcome_color,
            linewidth=0.7,
        ),
    )

    text_axis.plot(
        [0.02, 0.98], [0.855, 0.855],
        color=SUBTLE_LINE_COLOR,
        linewidth=0.9,
        transform=text_axis.transAxes,
        clip_on=False,
    )

    rows = [
        ("Concept", wrap_display_text(concept_name, width=22)),
        ("Concept value", f"{case['concept_before']:.3f}  →  {case['concept_after']:.3f}"),
        (
            "Prediction",
            wrap_display_text(
                f"{case['before_class_name']}  →  {case['after_class_name']}",
                width=28,
            ),
        ),
        (
            "True-class prob.",
            f"{case['true_probability_before']:.3f}  →  "
            f"{case['true_probability_after']:.3f}  ({delta:+.3f})",
        ),
    ]

    row_centers = [0.70, 0.52, 0.34, 0.16]
    separator_positions = [0.61, 0.43, 0.25]

    for (label, value), y in zip(rows, row_centers):
        text_axis.text(
            0.03, y,
            label,
            ha="left", va="center",
            fontsize=10.2,
            fontweight="semibold",
            color=MUTED_TEXT_COLOR,
            transform=text_axis.transAxes,
        )
        text_axis.text(
            0.40, y,
            value,
            ha="left", va="center",
            fontsize=11.6,
            color=TEXT_COLOR,
            linespacing=1.10,
            transform=text_axis.transAxes,
        )

    for y in separator_positions:
        text_axis.plot(
            [0.03, 0.98], [y, y],
            color=SUBTLE_LINE_COLOR,
            linewidth=0.75,
            transform=text_axis.transAxes,
            clip_on=False,
        )

    # -----------------------------
    # Probability chart
    # -----------------------------
    selected = probability_classes(
        base_probs=case["base_probs"],
        new_probs=case["new_probs"],
        true_class=case["true_class_index"],
        top_n=top_classes,
    )

    before = case["base_probs"][selected]
    after = case["new_probs"][selected]
    labels = [class_names[index] for index in selected]

    order = np.argsort(np.maximum(before, after))
    before = before[order]
    after = after[order]
    selected = selected[order]
    labels = [labels[i] for i in order]
    labels = [textwrap.fill(label, width=18) for label in labels]

    y_positions = np.arange(len(labels))

    before_bars = probability_axis.barh(
        y_positions, before,
        height=0.52,
        color=BEFORE_COLOR,
        alpha=0.42,
        edgecolor="none",
        label="Before intervention",
        zorder=2,
    )

    after_bars = probability_axis.barh(
        y_positions, after,
        height=0.26,
        color=AFTER_COLOR,
        alpha=0.95,
        edgecolor="none",
        label="After intervention",
        zorder=3,
    )

    probability_axis.set_yticks(y_positions)
    probability_axis.set_yticklabels(
        labels,
        fontsize=11.0,
        color=TEXT_COLOR,
    )

    probability_axis.set_xlim(0.0, 1.02)
    probability_axis.set_xlabel(
        "Class probability",
        fontsize=11.3,
        color=TEXT_COLOR,
        labelpad=6,
    )
    probability_axis.set_xticks(np.linspace(0.0, 1.0, 6))
    probability_axis.tick_params(
        axis="x",
        labelsize=10.0,
        colors=MUTED_TEXT_COLOR,
        length=3,
        width=0.6,
    )
    probability_axis.tick_params(axis="y", length=0, pad=6)

    probability_axis.grid(
        axis="x",
        color=GRID_COLOR,
        linewidth=0.75,
        linestyle=(0, (2, 3)),
        zorder=0,
    )
    probability_axis.set_axisbelow(True)

    probability_axis.spines["top"].set_visible(False)
    probability_axis.spines["right"].set_visible(False)
    probability_axis.spines["left"].set_visible(False)
    probability_axis.spines["bottom"].set_color("#B9BEC5")
    probability_axis.spines["bottom"].set_linewidth(0.75)

    for tick, class_index in zip(probability_axis.get_yticklabels(), selected):
        if int(class_index) == int(case["true_class_index"]):
            tick.set_fontweight("semibold")
            tick.set_color(TEXT_COLOR)

    # Label only the "after" bars, and only larger "before" values
    for y, value in enumerate(after):
        value = float(value)
        if value < 0.05:
            continue
        if value >= 0.92:
            x = value - 0.015
            ha = "right"
        else:
            x = value + 0.012
            ha = "left"
        probability_axis.text(
            x, y + 0.15,
            f"{value:.2f}",
            ha=ha, va="center",
            fontsize=9.0,
            color=AFTER_COLOR,
            zorder=5,
        )

    for y, value in enumerate(before):
        value = float(value)
        if value < 0.12:
            continue
        x = value + 0.012 if value < 0.90 else value - 0.015
        ha = "left" if value < 0.90 else "right"
        probability_axis.text(
            x, y - 0.15,
            f"{value:.2f}",
            ha=ha, va="center",
            fontsize=8.7,
            color=BEFORE_COLOR,
            zorder=5,
        )

    if show_legend:
        probability_axis.legend(loc="lower right", frameon=False, fontsize=10)

    return before_bars, after_bars

def make_figure(
    cases: Sequence[Dict[str, Any]],
    data: np.lib.npyio.NpzFile,
    class_names: Sequence[str],
    output_path: Path,
    image_paths: Optional[Sequence[str]],
    image_root: Optional[str],
    explicit_image: Optional[str],
    title: Optional[str],
    top_classes: int,
) -> None:
    n_rows = len(cases)
    figure, axes = plt.subplots(
        n_rows,
        3,
        figsize=(15.5, 5.1 * n_rows),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.5]},
        squeeze=False,
        constrained_layout=True,
    )

    for row, case in enumerate(cases):
        image = get_image_for_sample(
            data=data,
            sample_index=case["sample_index"],
            image_paths=image_paths,
            image_root=image_root,
            explicit_image=explicit_image if n_rows == 1 else None,
        )

        draw_case(
            axes=axes[row],
            case=case,
            image=image,
            class_names=class_names,
            top_classes=top_classes,
            show_legend=(row == 0),
        )

        axes[row, 0].text(
            -0.05,
            1.06,
            f"({chr(ord('a') + row)})",
            transform=axes[row, 0].transAxes,
            fontsize=15,
            fontweight="bold",
            va="bottom",
        )

    if title:
        figure.suptitle("??", fontsize=17, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=350, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def serializable_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"base_probs", "new_probs"}
    }




def draw_combined_intervention_figure(
    items,
    output_path,
    top_classes,
    title,
):
    n_rows = len(items)

    figure, axes = plt.subplots(
        n_rows,
        3,
        figsize=(15.8, 4.25 * n_rows + 1.8),
        gridspec_kw={"width_ratios": [1.10, 1.70, 1.90]},
        squeeze=False,
    )

    figure.patch.set_facecolor("white")

    figure.subplots_adjust(
        top=0.92,
        bottom=0.09,
        left=0.055,
        right=0.985,
        hspace=0.58,
        wspace=0.2,
    )

    legend_handles = None

    for row, item in enumerate(items):
        case = item["case"]
        image = item["image"]

        handles = draw_case(
            axes=axes[row],
            case=case,
            image=image,
            class_names=item["class_names"],
            top_classes=top_classes,
            show_legend=False,
        )

        if legend_handles is None:
            legend_handles = [handles[0][0], handles[1][0]]

        backbone_name = display_backbone_name(item["backbone"])

        # Put row headings in figure coordinates, not inside the axes
        pos = axes[row, 0].get_position()
        y_top = pos.y1 + 0.015

        row_heading = (
            f"({chr(ord('a') + row)})  {item['dataset']}  ·  "
            f"{item['model_display']}  ·  {backbone_name}"
        )

        figure.text(
            pos.x0,
            y_top,
            row_heading,
            ha="left",
            va="bottom",
            fontsize=12.0,
            fontweight="semibold",
            color=TEXT_COLOR,
        )

        figure.text(
            pos.x0,
            y_top - 0.01,
            f"True class: {case['true_class_name']}",
            ha="left",
            va="bottom",
            fontsize=11.5,
            fontweight="semibold",
            color=TEXT_COLOR,
        )

    column_titles = ["Input", "Intervention response", "Class probabilities"]
    header_y = 0.935 if title else 0.965

    for j, label in enumerate(column_titles):
        pos = axes[0, j].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        figure.text(
            x_center,
            header_y,
            label,
            ha="center",
            va="center",
            fontsize=13.2,
            fontweight="semibold",
            color=TEXT_COLOR,
        )

    figure.add_artist(
        plt.Line2D(
            [0.055, 0.985],
            [header_y - 0.028, header_y - 0.028],
            transform=figure.transFigure,
            color="#BBC1C8",
            linewidth=0.8,
        )
    )

    for row in range(n_rows - 1):
        upper_pos = axes[row, 0].get_position()
        lower_pos = axes[row + 1, 0].get_position()
        separator_y = (upper_pos.y0 + lower_pos.y1) / 2

        figure.add_artist(
            plt.Line2D(
                [0.055, 0.985],
                [separator_y, separator_y],
                transform=figure.transFigure,
                color=SUBTLE_LINE_COLOR,
                linewidth=0.7,
                linestyle=(0, (2, 3)),
            )
        )

    """if title:
        figure.suptitle(
            title,
            fontsize=17.2,
            fontweight="semibold",
            color=TEXT_COLOR,
            y=0.985,
        )"""

    if legend_handles is not None:
        figure.legend(
            legend_handles,
            ["Before intervention", "After intervention"],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=2,
            frameon=False,
            fontsize=10.8,
            labelcolor=TEXT_COLOR,
            handlelength=1.8,
            columnspacing=2.0,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )

    plt.close(figure)

def draw_candidate_sheet(
    dataset: str,
    model_display: str,
    backbone: str,
    candidates: Sequence[Dict[str, Any]],
    dataset_info: DatasetInfo,
    data: np.lib.npyio.NpzFile,
    output_path: Path,
) -> None:
    """Save a visual shortlist for manual manuscript-example selection."""
    if not candidates:
        return

    n_rows = len(candidates)
    figure, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10.5, 2.8 * n_rows + 0.5),
        gridspec_kw={"width_ratios": [1.0, 2.25], "hspace": 0.34},
        squeeze=False,
    )

    for rank, case in enumerate(candidates, start=1):
        image_axis, text_axis = axes[rank - 1]
        image, warning = image_from_dataset_info(
            info=dataset_info,
            data=data,
            sample_index=case["sample_index"],
        )
        if image is not None:
            image_axis.imshow(image)
        else:
            image_axis.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
        image_axis.axis("off")
        image_axis.set_title(
            f"#{rank} · sample {case['sample_index']}",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )

        text_axis.axis("off")
        warning_text = f"\nWarning: {warning}" if warning else ""
        text = (
            f"Concept: {case['concept_name']} "
            f"(index {case['concept_index']})\n"
            f"Value: {case['concept_before']:.3f} → {case['concept_after']:.3f}\n"
            f"Class: {case['before_class_name']} → {case['after_class_name']} "
            f"(true: {case['true_class_name']})\n"
            f"True-class prob.: {case['true_probability_before']:.3f} → "
            f"{case['true_probability_after']:.3f} "
            f"({case['true_probability_change']:+.3f})\n"
            f"Outcome: {case['outcome']}"
            f"{warning_text}"
        )
        text_axis.text(
            0.0,
            0.95,
            text,
            ha="left",
            va="top",
            fontsize=10.5,
            linespacing=1.3,
        )

    figure.suptitle(
        f"Candidate interventions · {dataset} · {model_display} · {backbone}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.subplots_adjust(top=0.96, bottom=0.02, left=0.04, right=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def prepare_dataset_cases(
    dataset: str,
    model: str,
    vector_roots: Sequence[Path],
    data_dir: str,
    scratch_dir: str,
    batch_size: int,
    seed: int,
    backbone_override: Optional[str],
    config: Optional[str],
    fold: Optional[str],
    csv_paths: Sequence[str],
    minimum_agreement: float,
    mode: str,
    min_concept_change: float,
    max_samples: Optional[int],
    manual_case: Optional[Tuple[int, Optional[int]]],
    candidate_count: int,
    candidate_output_dir: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    backbone = (
        SELECTED_BACKBONES[dataset]
        if backbone_override is None
        else backbone_override.lower()
    )

    npz_path, selection = select_npz_across_default_roots(
        roots=vector_roots,
        dataset=dataset,
        model=model,
        backbone=backbone,
        config=config,
        fold=fold,
        csv_paths=csv_paths,
        minimum_agreement=minimum_agreement,
    )
    selection["selection_mode"] = "automatic"

    print(f"\n[{dataset}] automatically selected")
    print(f"  model:    {selection['model_display']}")
    print(f"  backbone: {selection['backbone']}")
    print(f"  config:   {selection['config']}")
    print(f"  fold/run: {selection['fold']}")
    print(f"  file:     {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    y_true = safe_get(data, "y_true")
    concepts = safe_get(data, "concept_probs")
    annotations = safe_get(data, "a_true")

    missing = [
        name
        for name, value in {
            "y_true": y_true,
            "concept_probs": concepts,
            "a_true": annotations,
        }.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            f"{npz_path} is missing required arrays: "
            + ", ".join(missing)
        )

    y_true = np.asarray(y_true).astype(int)
    concepts = np.asarray(concepts, dtype=float)
    annotations = np.asarray(annotations, dtype=float)

    predictor_type, weights, bias = detect_predictor(data)

    if weights.shape[0] != concepts.shape[1]:
        raise ValueError(
            f"Classifier shape {weights.shape} is incompatible with "
            f"concept vectors {concepts.shape}."
        )

    base_logits = recompute_logits(
        concepts=concepts,
        predictor_type=predictor_type,
        weights=weights,
        bias=bias,
    )
    base_probs = softmax(base_logits, axis=1)
    base_predictions = np.argmax(base_probs, axis=1)

    dataset_info = load_dataset_info(
        dataset=dataset,
        data_dir=data_dir,
        scratch_dir=scratch_dir,
        batch_size=batch_size,
        seed=seed,
    )

    n_classes = base_probs.shape[1]
    n_concepts = concepts.shape[1]

    class_names = resolve_names(
        external_path=None,
        data=data,
        npz_keys=("class_names", "classes"),
        count=n_classes,
        prefix="Class",
        fallback_names=dataset_info.class_names,
    )
    concept_names = resolve_names(
        external_path=None,
        data=data,
        npz_keys=("concept_names", "attribute_names", "attributes"),
        count=n_concepts,
        prefix="Concept",
        fallback_names=dataset_info.concept_names,
    )

    requested_kinds = (
        ["expected", "unexpected"]
        if mode == "both"
        else [mode]
    )

    items: List[Dict[str, Any]] = []
    selection["candidate_shortlists"] = {}

    for kind in requested_kinds:
        candidates = rank_candidate_cases(
            wanted_kind=kind,
            concepts=concepts,
            annotations=annotations,
            y_true=y_true,
            base_logits=base_logits,
            base_probs=base_probs,
            base_predictions=base_predictions,
            predictor_type=predictor_type,
            weights=weights,
            concept_names=concept_names,
            class_names=class_names,
            min_concept_change=min_concept_change,
            max_samples=max_samples,
            top_n=max(1, candidate_count),
        )

        selection["candidate_shortlists"][kind] = [
            serializable_case(candidate) for candidate in candidates
        ]

        if candidates:
            print(f"  top {kind} candidates:")
            for rank, candidate in enumerate(candidates, start=1):
                print(
                    f"    {rank}. sample={candidate['sample_index']}, "
                    f"concept={candidate['concept_index']} "
                    f"({candidate['concept_name']}), "
                    f"{candidate['before_class_name']} -> "
                    f"{candidate['after_class_name']}, "
                    f"delta_true={candidate['true_probability_change']:+.3f}"
                )

            if candidate_output_dir is not None:
                draw_candidate_sheet(
                    dataset=dataset,
                    model_display=MODEL_DISPLAY_NAMES.get(
                        normalize_model_key(model), normalize_model_key(model)
                    ),
                    backbone=backbone,
                    candidates=candidates,
                    dataset_info=dataset_info,
                    data=data,
                    output_path=(
                        candidate_output_dir
                        / f"{normalize_model_key(model)}_{dataset}_{kind}_candidates.png"
                    ),
                )

        if manual_case is not None and kind == "expected":
            sample_index, concept_index = manual_case
            case = select_manual_case(
                sample_index=sample_index,
                concept_index=concept_index,
                concepts=concepts,
                annotations=annotations,
                y_true=y_true,
                base_logits=base_logits,
                base_probs=base_probs,
                base_predictions=base_predictions,
                predictor_type=predictor_type,
                weights=weights,
                concept_names=concept_names,
                class_names=class_names,
            )
            selection["selection_mode"] = "manual"
            selection["manual_sample_index"] = sample_index
            selection["manual_concept_index"] = concept_index
        else:
            case = candidates[0] if candidates else None

        if case is None:
            print(
                f"[warn] No {kind} intervention case found for {dataset}."
            )
            continue

        image, image_warning = image_from_dataset_info(
            info=dataset_info,
            data=data,
            sample_index=case["sample_index"],
        )

        if image_warning:
            print(
                f"[warn] {dataset}, row {case['sample_index']}: "
                f"{image_warning}"
            )

        items.append(
            {
                "dataset": dataset,
                "model": normalize_model_key(model),
                "model_display": MODEL_DISPLAY_NAMES.get(normalize_model_key(model), normalize_model_key(model),),
                "backbone": backbone,
                "case": case,
                "image": image,
                "image_warning": image_warning,
                "class_names": class_names,
                "npz_path": str(npz_path),
                "selection": selection,
            }
        )

    selection["npz"] = str(npz_path)
    selection["predictor_type"] = predictor_type

    return items, selection



def parse_manual_cases(values: Optional[Sequence[str]]) -> Dict[str, Tuple[int, Optional[int]]]:
    """Parse DATASET:SAMPLE[:CONCEPT] overrides from the command line."""
    parsed: Dict[str, Tuple[int, Optional[int]]] = {}
    if not values:
        return parsed

    for value in values:
        parts = value.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                "Manual cases must use DATASET:SAMPLE or "
                "DATASET:SAMPLE:CONCEPT."
            )
        dataset = parts[0]
        if dataset not in DEFAULT_DATASETS:
            raise ValueError(f"Unknown dataset in manual case: {dataset}")
        sample_index = int(parts[1])
        concept_index = int(parts[2]) if len(parts) == 3 else None
        parsed[dataset] = (sample_index, concept_index)

    return parsed

def main() -> None:
    global ACTIVE_PACBM_LOGIT_CLIP

    parser = argparse.ArgumentParser(
        description=(
            "Generate manuscript-ready qualitative intervention examples "
            "with automatic model configuration, semantic names, and images."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        help="Datasets included in the combined figure.",
    )
    parser.add_argument(
        "--model",
        default="pacbm",
        help=(
            "Model to illustrate: pacbm, pacbm_2, jointcbm, or klcbm. "
            "The default is PACBM-Cl."
        ),
    )

    parser.add_argument(
        "--vectors_root",
        default=None,
        help=(
            "Optional single vectors root. When omitted, the script searches "
            "recomputed_test_vectors and recomputed_test_vectors/2."
        ),
    )
    parser.add_argument(
        "--data_dir",
        default="../data",
        help="Dataset root used by the project data loaders.",
    )
    parser.add_argument(
        "--scratch_dir",
        default="_qualitative_intervention_scratch",
        help="Temporary directory used by the dataset loaders.",
    )
    parser.add_argument(
        "--output_dir",
        default="interventions_examples_native",
        help="Directory in which the combined figure and metadata are saved.",
    )

    parser.add_argument(
        "--backbone",
        default="auto",
        help=(
            "Optional backbone override. 'auto' uses EfficientNetB0 for "
            "AwA2/aPY and InceptionV3 for CUB."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional exact configuration folder.",
    )
    parser.add_argument(
        "--fold",
        default=None,
        help="Optional exact fold or run folder.",
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[
            "intervenability_metrics_native/"
            "intervenability_aggregated.csv",
            "intervenability_metrics/final_2/neg_neutralize/2/"
            "intervenability_aggregated.csv",
        ],
        help=(
            "The same aggregated CSV files used by the intervention-curve "
            "plotting script."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["expected", "unexpected", "both"],
        default="expected",
        help=(
            "Type of qualitative response shown for each dataset. "
            "'expected' keeps the default combined figure compact."
        ),
    )
    parser.add_argument(
        "--min_concept_change",
        type=float,
        default=0.25,
        help=(
            "Minimum difference between predicted and oracle concept values "
            "for automatic example selection."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on the number of test samples searched.",
    )
    parser.add_argument(
        "--candidate_count",
        type=int,
        default=8,
        help=(
            "Number of ranked candidate interventions printed and saved "
            "for each dataset. Set to 1 to disable the shortlist."
        ),
    )
    parser.add_argument(
        "--save_candidate_sheets",
        action="store_true",
        help=(
            "Save one PDF shortlist per dataset so examples can be inspected "
            "before choosing manual sample/concept indices."
        ),
    )
    parser.add_argument(
        "--manual_case",
        nargs="*",
        default=None,
        metavar="DATASET:SAMPLE[:CONCEPT]",
        help=(
            "Optional manual overrides, e.g. AwA2:125:7 aPY:812 CUB:44:19. "
            "When the concept index is omitted, the strongest single-concept "
            "effect for that sample is used."
        ),
    )

    parser.add_argument(
        "--top_classes",
        type=int,
        default=4,
        help="Number of class probabilities displayed per example.",
    )
    parser.add_argument(
        "--minimum_recompute_agreement",
        type=float,
        default=0.98,
        help=(
            "Minimum agreement between saved and reconstructed predictions."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--title",
        default=None,#"Single-concept intervention examples",
        help="Combined figure title. Pass an empty string to omit.",
    )

    args = parser.parse_args()
    manual_cases = parse_manual_cases(args.manual_case)

    model = normalize_model_key(args.model)

    if model == "pacbm_2":
        ACTIVE_PACBM_LOGIT_CLIP = PACBM_CO_LOGIT_CLIP
    else:
        ACTIVE_PACBM_LOGIT_CLIP = PACBM_CL_LOGIT_CLIP

    if args.vectors_root is None:
        vector_roots = DEFAULT_VECTOR_ROOTS
    else:
        vector_roots = [Path(args.vectors_root)]

    backbone_override = (
        None if args.backbone.lower() == "auto" else args.backbone
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_output_dir = (
        output_dir / "candidates" if args.save_candidate_sheets else None
    )

    all_items: List[Dict[str, Any]] = []
    all_selections: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        try:
            items, selection = prepare_dataset_cases(
                dataset=dataset,
                model=model,
                vector_roots=vector_roots,
                data_dir=args.data_dir,
                scratch_dir=args.scratch_dir,
                batch_size=args.batch_size,
                seed=args.seed,
                backbone_override=backbone_override,
                config=args.config,
                fold=args.fold,
                csv_paths=args.csv,
                minimum_agreement=args.minimum_recompute_agreement,
                mode=args.mode,
                min_concept_change=args.min_concept_change,
                max_samples=args.max_samples,
                manual_case=manual_cases.get(dataset),
                candidate_count=args.candidate_count,
                candidate_output_dir=candidate_output_dir,
            )
        except Exception as error:
            print(
                f"[skip] {dataset}: {type(error).__name__}: {error}"
            )
            continue

        all_items.extend(items)
        all_selections.append(selection)

    if not all_items:
        raise RuntimeError(
            "No intervention example could be generated for any dataset."
        )

    dataset_tag = "_".join(args.datasets)
    output_stem = (
        f"{model}_{dataset_tag}_{args.mode}_"
        "qualitative_interventions"
    )

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    json_path = output_dir / f"{output_stem}.json"

    draw_combined_intervention_figure(
        items=all_items,
        output_path=png_path,
        top_classes=args.top_classes,
        title=args.title or None,
    )
    draw_combined_intervention_figure(
        items=all_items,
        output_path=pdf_path,
        top_classes=args.top_classes,
        title=args.title or None,
    )

    json_path.write_text(
        json.dumps(
            {
                "model": model,
                "datasets": args.datasets,
                "mode": args.mode,
                "selections": all_selections,
                "cases": [
                    {
                        "dataset": item["dataset"],
                        "model": item["model"],
                        "backbone": item["backbone"],
                        "npz_path": item["npz_path"],
                        **serializable_case(item["case"]),
                    }
                    for item in all_items
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nGenerated files:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()