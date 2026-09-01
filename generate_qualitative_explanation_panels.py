"""
generate_qualitative_explanation_panels.py

Standalone qualitative explanation visualization script for saved concept-based model vectors.

It uses ONLY:
  - recomputed_test_vectors/**/test_vectors.npz
  - dataset images / image paths in the same order as the vectors
  - optional concept/class name files

It does NOT:
  - retrain models
  - reload model weights
  - import TensorFlow/Keras/PyTorch model code

Main outputs:
  manuscript_assets/qualitative_explanations/*.png
  manuscript_assets/qualitative_explanations/qualitative_explanation_examples.csv

Important image-order rule:
  The script can only make valid panels if image paths are available in the exact same
  order as the saved vectors. It first checks the .npz for image_paths/path-like arrays.
  If absent, it looks for explicit path files under --datasets_root. If none are found,
  it writes a clear warning file explaining what metadata is needed.

Example:
  python generate_qualitative_explanation_panels.py \
    --vectors_root recomputed_test_vectors \
    --datasets_root ../data \
    --output_dir manuscript_assets/qualitative_explanations \
    --top_k 5 \
    --num_examples 10

Optional explicit files:
  python generate_qualitative_explanation_panels.py \
    --vectors_root recomputed_test_vectors \
    --datasets_root ../data \
    --output_dir manuscript_assets/qualitative_explanations \
    --image_paths_file metadata/AwA2_mobilenetv2_fold1_test_paths.txt \
    --concept_names_file metadata/AwA2_concepts.txt \
    --class_names_file metadata/AwA2_classes.txt
"""

import argparse
import csv
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODELS = ["pacbm", "jointcbm", "klcbm"]
DATASETS = ["AwA2", "aPY", "CUB"]
BACKBONES = ["mobilenetv2", "inceptionv3", "efficientnetb0"]
PANEL_TYPES = ["correct", "misclassified", "high_confidence", "low_confidence", "random"]

IMAGE_PATH_KEYS = [
    "image_paths", "test_image_paths", "img_paths", "paths", "filenames", "file_paths",
    "image_files", "test_paths", "samples", "sample_paths",
]

CLASS_NAME_KEYS = ["class_names", "classes", "label_names"]
CONCEPT_NAME_KEYS = ["concept_names", "concepts", "attribute_names", "attributes"]


def warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=2)
    print(f"[warn] {msg}")


def info(msg: str) -> None:
    print(f"[info] {msg}")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_name(x: str) -> str:
    return str(x).strip().lower()


def safe_stem(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown"


def decode_array_strings(arr: np.ndarray) -> list[str]:
    out = []
    for v in np.asarray(arr).reshape(-1):
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def read_lines(path: str | Path) -> list[str]:
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(line)
    return values


def read_name_file(path: str | Path) -> list[str]:
    """Read names from .txt/.csv/.json/.npy/.npz.

    For CSV, uses the first non-empty field per line unless columns named name/class/concept exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in [".txt", ".lst", ".names"]:
        return read_lines(path)

    if suffix == ".csv":
        names = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                preferred = ["name", "concept", "concept_name", "attribute", "attribute_name", "class", "class_name"]
                col = next((c for c in preferred if c in reader.fieldnames), None)
                if col is not None:
                    for row in reader:
                        if row.get(col, "").strip():
                            names.append(row[col].strip())
                    return names
            f.seek(0)
            reader2 = csv.reader(f)
            for row in reader2:
                row = [x.strip() for x in row if str(x).strip()]
                if row:
                    names.append(row[-1] if len(row) >= 2 and row[0].isdigit() else row[0])
        return names

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            # If dict maps name->id, invert. If id->name, sort by id.
            items = []
            for k, v in data.items():
                try:
                    items.append((int(k), str(v)))
                except Exception:
                    try:
                        items.append((int(v), str(k)))
                    except Exception:
                        pass
            if items:
                return [name for _, name in sorted(items)]
        raise ValueError(f"Unsupported JSON name structure in {path}")

    if suffix == ".npy":
        return decode_array_strings(np.load(path, allow_pickle=True))

    if suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        for key in [*CONCEPT_NAME_KEYS, *CLASS_NAME_KEYS, "names"]:
            if key in z:
                return decode_array_strings(z[key])
        raise ValueError(f"No name-like key found in {path}")

    raise ValueError(f"Unsupported name file extension: {path}")


def maybe_get_strings_from_npz(z: np.lib.npyio.NpzFile, keys: list[str]) -> list[str] | None:
    for key in keys:
        if key in z:
            try:
                return decode_array_strings(z[key])
            except Exception as e:
                warn(f"Could not decode {key} from npz: {e}")
    return None


def find_candidate_name_files(datasets_root: Path, dataset: str, kind: str) -> list[Path]:
    """Return likely files for concepts/classes without silently requiring one exact convention."""
    ds_dirs = [datasets_root / dataset, datasets_root / dataset.lower(), datasets_root]
    if kind == "concept":
        names = [
            "concept_names.txt", "concepts.txt", "attributes.txt", "attribute_names.txt",
            f"{dataset}_concept_names.txt", f"{dataset}_concepts.txt", f"{dataset}_attributes.txt",
            "concept_names.csv", "concepts.csv", "attributes.csv", "attribute_names.csv",
            f"{dataset}_concept_names.csv", f"{dataset}_attributes.csv",
            "concept_names.json", "attributes.json",
        ]
    else:
        names = [
            "class_names.txt", "classes.txt", "label_names.txt", "labels.txt",
            f"{dataset}_class_names.txt", f"{dataset}_classes.txt",
            "class_names.csv", "classes.csv", "label_names.csv", "labels.csv",
            "class_names.json", "classes.json", "labels.json",
        ]
    out = []
    for d in ds_dirs:
        for n in names:
            p = d / n
            if p.exists():
                out.append(p)
    return out


def load_names(
    z: np.lib.npyio.NpzFile,
    datasets_root: Path,
    dataset: str,
    n_expected: int,
    kind: str,
    explicit_file: str | None = None,
) -> list[str]:
    keys = CONCEPT_NAME_KEYS if kind == "concept" else CLASS_NAME_KEYS

    if explicit_file:
        names = read_name_file(explicit_file)
        if len(names) != n_expected:
            warn(f"{kind} names file has {len(names)} entries, expected {n_expected}: {explicit_file}. Using IDs instead.")
            return [f"{kind}_{i}" for i in range(n_expected)]
        return names

    from_npz = maybe_get_strings_from_npz(z, keys)
    if from_npz is not None and len(from_npz) == n_expected:
        return from_npz
    if from_npz is not None:
        warn(f"Found {kind} names in npz but length={len(from_npz)}, expected={n_expected}. Ignoring.")

    for path in find_candidate_name_files(datasets_root, dataset, kind):
        try:
            names = read_name_file(path)
            if len(names) == n_expected:
                info(f"Using {kind} names from {path}")
                return names
            warn(f"Candidate {kind} name file length mismatch: {path} has {len(names)}, expected {n_expected}.")
        except Exception as e:
            warn(f"Could not read candidate {kind} name file {path}: {e}")

    warn(
        f"Missing valid {kind} names for {dataset}. Saving {kind} IDs instead. "
        f"Provide --{kind}_names_file or place a valid file under --datasets_root/{dataset}."
    )
    return [f"{kind}_{i}" for i in range(n_expected)]


def resolve_path(path_value: str, datasets_root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    p2 = datasets_root / path_value
    if p2.exists():
        return p2
    return p


def read_image_paths_file(path: str | Path, datasets_root: Path) -> list[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    paths: list[str] = []

    if suffix in [".txt", ".lst"]:
        paths = read_lines(path)
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                col = next((c for c in ["image_path", "path", "file_path", "filename", "image"] if c in reader.fieldnames), None)
                if col:
                    paths = [row[col].strip() for row in reader if row.get(col, "").strip()]
                else:
                    f.seek(0)
                    reader2 = csv.reader(f)
                    paths = [row[0].strip() for row in reader2 if row and row[0].strip()]
            else:
                f.seek(0)
                reader2 = csv.reader(f)
                paths = [row[0].strip() for row in reader2 if row and row[0].strip()]
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            paths = [str(x) for x in data]
        elif isinstance(data, dict):
            for key in IMAGE_PATH_KEYS:
                if key in data:
                    paths = [str(x) for x in data[key]]
                    break
        if not paths:
            raise ValueError(f"Could not find image paths in JSON file {path}")
    elif suffix == ".npy":
        paths = decode_array_strings(np.load(path, allow_pickle=True))
    elif suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        arr = maybe_get_strings_from_npz(z, IMAGE_PATH_KEYS)
        if arr is None:
            raise ValueError(f"Could not find path-like array in {path}")
        paths = arr
    else:
        raise ValueError(f"Unsupported image path file extension: {path}")

    return [str(resolve_path(p, datasets_root)) for p in paths]


def candidate_image_path_files(datasets_root: Path, dataset: str, model: str, backbone: str, fold: str) -> list[Path]:
    ds_dirs = [datasets_root / dataset, datasets_root / dataset.lower(), datasets_root]
    fold_variants = [fold, fold.replace("fold", ""), f"fold{fold}" if not fold.startswith("fold") else fold]
    names = []
    for f in fold_variants:
        names.extend([
            f"{dataset}_{model}_{backbone}_{f}_test_image_paths.txt",
            f"{dataset}_{backbone}_{f}_test_image_paths.txt",
            f"{model}_{dataset}_{backbone}_{f}_test_image_paths.txt",
            f"{dataset}_{model}_{backbone}_{f}_test_paths.txt",
            f"{dataset}_{backbone}_{f}_test_paths.txt",
        ])
    names.extend([
        "test_image_paths.txt", "test_paths.txt", "image_paths.txt",
        "test_image_paths.csv", "test_paths.csv", "image_paths.csv",
        f"{dataset}_test_image_paths.txt", f"{dataset}_test_paths.txt",
        f"{dataset}_test_image_paths.csv", f"{dataset}_test_paths.csv",
    ])
    out = []
    for d in ds_dirs:
        for n in names:
            p = d / n
            if p.exists():
                out.append(p)
    return out


def load_image_paths(
    z: np.lib.npyio.NpzFile,
    datasets_root: Path,
    dataset: str,
    model: str,
    backbone: str,
    fold: str,
    n_expected: int,
    explicit_file: str | None = None,
) -> list[str] | None:
    if explicit_file:
        paths = read_image_paths_file(explicit_file, datasets_root)
        if len(paths) != n_expected:
            warn(f"Explicit image path file has {len(paths)} paths but expected {n_expected}: {explicit_file}")
            return None
        return paths

    from_npz = maybe_get_strings_from_npz(z, IMAGE_PATH_KEYS)
    if from_npz is not None:
        paths = [str(resolve_path(p, datasets_root)) for p in from_npz]
        if len(paths) == n_expected:
            return paths
        warn(f"Found image paths in npz but length={len(paths)}, expected={n_expected}. Ignoring.")

    for path in candidate_image_path_files(datasets_root, dataset, model, backbone, fold):
        try:
            paths = read_image_paths_file(path, datasets_root)
            if len(paths) == n_expected:
                info(f"Using image paths from {path}")
                return paths
            warn(f"Candidate image path file length mismatch: {path} has {len(paths)}, expected {n_expected}.")
        except Exception as e:
            warn(f"Could not read candidate image path file {path}: {e}")

    return None


def write_missing_image_metadata_note(out_dir: Path, dataset: str, model: str, backbone: str, fold: str, n_expected: int) -> None:
    ensure_dir(out_dir)
    note = out_dir / f"MISSING_IMAGE_PATHS_{model}_{dataset}_{backbone}_{fold}.txt"
    note.write_text(
        "Image panels were skipped because exact test image paths were not found.\n\n"
        f"Run: model={model}, dataset={dataset}, backbone={backbone}, fold={fold}\n"
        f"Expected number of paths: {n_expected}\n\n"
        "Needed metadata:\n"
        "  A text/CSV/JSON/NPY/NPZ file containing one image path per test sample, "
        "in the exact same order as the arrays in test_vectors.npz.\n\n"
        "Accepted ways to provide it:\n"
        "  1. Save an array in test_vectors.npz named one of: " + ", ".join(IMAGE_PATH_KEYS) + "\n"
        "  2. Pass --image_paths_file /path/to/test_image_paths.txt for this run.\n"
        "  3. Place a file under --datasets_root or --datasets_root/<dataset>/ named e.g.\n"
        "     test_image_paths.txt, test_paths.txt, image_paths.csv, "
        f"{dataset}_{backbone}_{fold}_test_image_paths.txt.\n\n"
        "The script does not infer image order from folders because that can silently mismatch the saved vectors.\n",
        encoding="utf-8",
    )
    warn(f"Missing exact image paths. Wrote requirements to {note}")


def parse_run_info(npz_path: Path) -> dict[str, str] | None:
    parts = [p for p in npz_path.parts]
    low_parts = [p.lower() for p in parts]

    model = next((m for m in MODELS if m in low_parts), None)
    dataset = None
    for d in DATASETS:
        if d.lower() in low_parts:
            dataset = d
            break
    backbone = next((b for b in BACKBONES if b in low_parts), None)

    fold = None
    for p in parts[::-1]:
        m = re.fullmatch(r"fold[_-]?(\d+)", p, flags=re.IGNORECASE)
        if m:
            fold = f"fold{m.group(1)}"
            break
    if fold is None:
        for p in parts[::-1]:
            m = re.search(r"fold[_-]?(\d+)", p, flags=re.IGNORECASE)
            if m:
                fold = f"fold{m.group(1)}"
                break

    if not all([model, dataset, backbone, fold]):
        warn(f"Could not parse model/dataset/backbone/fold from path, skipping: {npz_path}")
        return None
    return {"model": model, "dataset": dataset, "backbone": backbone, "fold": fold}


def find_vector_files(vectors_root: Path) -> list[Path]:
    return sorted(vectors_root.rglob("test_vectors.npz"))


def require_array(z: np.lib.npyio.NpzFile, key: str, npz_path: Path) -> np.ndarray | None:
    if key not in z:
        warn(f"Missing required array '{key}' in {npz_path}. Skipping run.")
        return None
    return np.asarray(z[key])


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


def orient_kernel(kernel: np.ndarray, m_concepts: int, k_classes: int, key: str) -> np.ndarray | None:
    kernel = np.asarray(kernel)
    if kernel.shape == (m_concepts, k_classes):
        return kernel
    if kernel.shape == (k_classes, m_concepts):
        warn(f"{key} appears transposed with shape {kernel.shape}; transposing to (M,K).")
        return kernel.T
    warn(f"Cannot use {key}: shape {kernel.shape}, expected (M,K)=({m_concepts},{k_classes}) or transposed.")
    return None


def compute_support_contributions(
    model: str,
    concept_probs: np.ndarray,
    class_probs: np.ndarray,
    pred: np.ndarray,
    z: np.lib.npyio.NpzFile,
) -> np.ndarray | None:
    n, m = concept_probs.shape
    k = class_probs.shape[1]

    if model == "pacbm":
        if "gamma" not in z:
            warn("PACBM: missing gamma, so top supporting concepts will be empty.")
            return None
        gamma = orient_kernel(np.asarray(z["gamma"]), m, k, "gamma")
        if gamma is None:
            return None
        return logit_np(concept_probs) * gamma[:, pred].T

    if model in ["jointcbm", "klcbm"]:
        if "concept_to_class_kernel" not in z:
            warn(f"{model}: missing concept_to_class_kernel, so top supporting concepts will be empty.")
            return None
        kernel = orient_kernel(np.asarray(z["concept_to_class_kernel"]), m, k, "concept_to_class_kernel")
        if kernel is None:
            return None
        return concept_probs * kernel[:, pred].T

    return None


def top_indices(values: np.ndarray, k: int, positive_only: bool = False) -> list[int]:
    values = np.asarray(values).reshape(-1)
    if positive_only:
        candidates = np.where(values > 0)[0]
        if len(candidates) == 0:
            return []
        return candidates[np.argsort(values[candidates])[::-1][:k]].astype(int).tolist()
    return np.argsort(values)[::-1][:k].astype(int).tolist()


def pack_top_concepts(indices: Iterable[int], names: list[str], values: np.ndarray) -> tuple[str, str]:
    labels = []
    scores = []
    for i in indices:
        name = names[i] if i < len(names) else f"concept_{i}"
        labels.append(str(name))
        scores.append(f"{float(values[i]):.4f}")
    return "; ".join(labels), "; ".join(scores)


def format_concepts_for_panel(names: list[str], scores: np.ndarray, k: int, prefix: str = "") -> str:
    idx = top_indices(scores, k, positive_only=False)
    lines = []
    for i in idx:
        name = names[i] if i < len(names) else f"concept_{i}"
        lines.append(f"{prefix}{name}: {float(scores[i]):.2f}")
    return "\n".join(lines) if lines else "(none)"


def format_support_for_panel(names: list[str], contrib: np.ndarray | None, k: int) -> str:
    if contrib is None:
        return "(not available)"
    idx = top_indices(contrib, k, positive_only=True)
    if not idx:
        idx = top_indices(contrib, k, positive_only=False)
    lines = []
    for i in idx:
        name = names[i] if i < len(names) else f"concept_{i}"
        lines.append(f"{name}: {float(contrib[i]):+.3f}")
    return "\n".join(lines) if lines else "(none)"


def select_examples(
    panel_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    num_examples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(y_true))

    if panel_type == "correct":
        candidates = all_idx[y_true == y_pred]
        order = candidates[np.argsort(confidence[candidates])[::-1]] if len(candidates) else candidates
    elif panel_type == "misclassified":
        candidates = all_idx[y_true != y_pred]
        order = candidates[np.argsort(confidence[candidates])[::-1]] if len(candidates) else candidates
    elif panel_type == "high_confidence":
        order = all_idx[np.argsort(confidence)[::-1]]
    elif panel_type == "low_confidence":
        order = all_idx[np.argsort(confidence)]
    elif panel_type == "random":
        order = all_idx.copy()
        rng.shuffle(order)
    else:
        raise ValueError(panel_type)

    if len(order) == 0:
        return np.array([], dtype=int)
    return np.asarray(order[:num_examples], dtype=int)


def load_image(path: str | Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        warn(f"Could not open image {path}: {e}")
        return None


def class_label(names: list[str], idx: int) -> str:
    return names[idx] if 0 <= idx < len(names) else f"class_{idx}"


def make_panel(
    selected: np.ndarray,
    image_paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    concept_probs: np.ndarray,
    support_contrib: np.ndarray | None,
    concept_names: list[str],
    class_names: list[str],
    out_path: Path,
    title: str,
    top_k: int,
) -> bool:
    if len(selected) == 0:
        warn(f"No selected examples for panel {out_path.name}; skipping.")
        return False

    rows = len(selected)
    fig_h = max(3.0 * rows, 3.8)
    fig, axes = plt.subplots(rows, 2, figsize=(10.5, fig_h), gridspec_kw={"width_ratios": [1.0, 1.75]})
    if rows == 1:
        axes = np.asarray([axes])

    fig.suptitle(title, fontsize=12, y=0.995)

    rendered = 0
    for r, idx in enumerate(selected):
        ax_img, ax_txt = axes[r, 0], axes[r, 1]
        ax_img.axis("off")
        ax_txt.axis("off")

        img = load_image(image_paths[int(idx)])
        if img is not None:
            ax_img.imshow(img)
            rendered += 1
        else:
            ax_img.text(0.5, 0.5, "image\nmissing", ha="center", va="center")

        true_name = class_label(class_names, int(y_true[idx]))
        pred_name = class_label(class_names, int(y_pred[idx]))
        ax_img.set_title(f"idx={idx}\ntrue: {true_name}\npred: {pred_name}\nconf={confidence[idx]:.3f}", fontsize=8)

        top_pred_text = format_concepts_for_panel(concept_names, concept_probs[idx], top_k)
        contrib_vec = support_contrib[idx] if support_contrib is not None else None
        support_text = format_support_for_panel(concept_names, contrib_vec, top_k)

        text = (
            "Top predicted concepts\n"
            "----------------------\n"
            f"{top_pred_text}\n\n"
            "Top supporting concepts\n"
            "-----------------------\n"
            f"{support_text}"
        )
        ax_txt.text(0.0, 0.5, text, ha="left", va="center", fontsize=8, family="monospace", wrap=True)

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved panel: {out_path}")
    return rendered > 0


def process_one_npz(
    npz_path: Path,
    args: argparse.Namespace,
    summary_writer: csv.DictWriter,
) -> int:
    run = parse_run_info(npz_path)
    if run is None:
        return 0

    model, dataset, backbone, fold = run["model"], run["dataset"], run["backbone"], run["fold"]
    if model not in args.models or dataset not in args.datasets or backbone not in args.backbones:
        return 0

    info(f"Processing {model}/{dataset}/{backbone}/{fold}: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)

    y_true = require_array(z, "y_true", npz_path)
    concept_probs = require_array(z, "concept_probs", npz_path)
    class_probs = require_array(z, "class_probs", npz_path)
    if y_true is None or concept_probs is None or class_probs is None:
        return 0

    y_true = np.asarray(y_true).reshape(-1).astype(int)
    concept_probs = np.asarray(concept_probs, dtype=float)
    class_probs = np.asarray(class_probs, dtype=float)

    if class_probs.ndim == 1:
        warn(f"class_probs is 1D in {npz_path}; skipping because class probabilities must be [N,K].")
        return 0
    if concept_probs.ndim != 2:
        warn(f"concept_probs shape is {concept_probs.shape} in {npz_path}; expected [N,M]. Skipping.")
        return 0
    if len(y_true) != len(concept_probs) or len(y_true) != len(class_probs):
        warn(f"Array length mismatch in {npz_path}: y={len(y_true)}, concepts={len(concept_probs)}, class_probs={len(class_probs)}. Skipping.")
        return 0

    if "y_pred" in z:
        y_pred = np.asarray(z["y_pred"]).reshape(-1).astype(int)
        if len(y_pred) != len(y_true):
            warn(f"y_pred length mismatch in {npz_path}; using argmax(class_probs).")
            y_pred = class_probs.argmax(axis=1).astype(int)
    else:
        y_pred = class_probs.argmax(axis=1).astype(int)

    n = len(y_true)
    m_concepts = concept_probs.shape[1]
    k_classes = class_probs.shape[1]
    confidence = class_probs[np.arange(n), y_pred]

    concept_names = load_names(z, Path(args.datasets_root), dataset, m_concepts, "concept", args.concept_names_file)
    class_names = load_names(z, Path(args.datasets_root), dataset, k_classes, "class", args.class_names_file)

    image_paths = load_image_paths(
        z=z,
        datasets_root=Path(args.datasets_root),
        dataset=dataset,
        model=model,
        backbone=backbone,
        fold=fold,
        n_expected=n,
        explicit_file=args.image_paths_file,
    )
    if image_paths is None:
        write_missing_image_metadata_note(Path(args.output_dir), dataset, model, backbone, fold, n)
        return 0

    missing = [p for p in image_paths[: min(20, len(image_paths))] if not Path(p).exists()]
    if missing:
        warn(f"Some image paths do not exist. First missing examples: {missing[:5]}")

    support_contrib = compute_support_contributions(model, concept_probs, class_probs, y_pred, z)

    written = 0
    for panel_type in PANEL_TYPES:
        selected = select_examples(
            panel_type=panel_type,
            y_true=y_true,
            y_pred=y_pred,
            confidence=confidence,
            num_examples=args.num_examples,
            seed=args.seed,
        )
        if len(selected) == 0:
            warn(f"No examples for {model}/{dataset}/{backbone}/{fold}/{panel_type}; skipping panel.")
            continue

        out_name = f"{model}_{dataset}_{backbone}_{fold}_{panel_type}.png"
        out_path = Path(args.output_dir) / out_name
        title = f"{model.upper()} | {dataset} | {backbone} | {fold} | {panel_type.replace('_', ' ')}"
        ok = make_panel(
            selected=selected,
            image_paths=image_paths,
            y_true=y_true,
            y_pred=y_pred,
            confidence=confidence,
            concept_probs=concept_probs,
            support_contrib=support_contrib,
            concept_names=concept_names,
            class_names=class_names,
            out_path=out_path,
            title=title,
            top_k=args.top_k,
        )
        if not ok:
            continue
        written += 1

        for idx in selected:
            idx = int(idx)
            top_pred_idx = top_indices(concept_probs[idx], args.top_k, positive_only=False)
            top_pred_names, top_pred_scores = pack_top_concepts(top_pred_idx, concept_names, concept_probs[idx])

            if support_contrib is not None:
                top_sup_idx = top_indices(support_contrib[idx], args.top_k, positive_only=True)
                if not top_sup_idx:
                    top_sup_idx = top_indices(support_contrib[idx], args.top_k, positive_only=False)
                top_sup_names, top_sup_scores = pack_top_concepts(top_sup_idx, concept_names, support_contrib[idx])
            else:
                top_sup_names, top_sup_scores = "", ""

            summary_writer.writerow({
                "model": model,
                "dataset": dataset,
                "backbone": backbone,
                "fold": fold,
                "sample_index": idx,
                "image_path": image_paths[idx],
                "true_class": class_label(class_names, int(y_true[idx])),
                "predicted_class": class_label(class_names, int(y_pred[idx])),
                "confidence": f"{float(confidence[idx]):.6f}",
                "panel_type": panel_type,
                "top_predicted_concepts": top_pred_names,
                "top_predicted_scores": top_pred_scores,
                "top_supporting_concepts": top_sup_names,
                "top_supporting_scores": top_sup_scores,
                "output_panel_path": str(out_path),
            })

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative concept explanation panels from saved test_vectors.npz files.")
    parser.add_argument("--vectors_root", type=str, default="recomputed_test_vectors")
    parser.add_argument("--datasets_root", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="manuscript_assets/qualitative_explanations")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)

    # Optional explicit files. Useful when metadata is not stored in .npz.
    parser.add_argument("--image_paths_file", type=str, default=None, help="Optional exact test image path file, same order as vectors. Best used for one run at a time.")
    parser.add_argument("--concept_names_file", type=str, default=None, help="Optional concept names file. If omitted, script checks npz and datasets_root.")
    parser.add_argument("--class_names_file", type=str, default=None, help="Optional class names file. If omitted, script checks npz and datasets_root.")

    args = parser.parse_args()
    ensure_dir(args.output_dir)

    vector_files = find_vector_files(Path(args.vectors_root))
    if not vector_files:
        raise FileNotFoundError(f"No test_vectors.npz found under {args.vectors_root}")
    info(f"Found {len(vector_files)} vector files under {args.vectors_root}")

    csv_path = Path(args.output_dir) / "qualitative_explanation_examples.csv"
    columns = [
        "model", "dataset", "backbone", "fold", "sample_index", "image_path",
        "true_class", "predicted_class", "confidence", "panel_type",
        "top_predicted_concepts", "top_predicted_scores",
        "top_supporting_concepts", "top_supporting_scores",
        "output_panel_path",
    ]

    total_panels = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for npz_path in vector_files:
            total_panels += process_one_npz(npz_path, args, writer)

    info(f"Wrote CSV summary: {csv_path}")
    info(f"Generated {total_panels} panels.")
    if total_panels == 0:
        warn("No panels were generated. Check warnings above, especially missing image path metadata.")


if __name__ == "__main__":
    main()
