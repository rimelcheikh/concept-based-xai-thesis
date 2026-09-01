"""
Plot the epoch-wise evolution of PACBM's dynamic anchoring coefficient.

Expected experiment layout
--------------------------
Each selected configuration should contain directories named fold1, ..., fold5
(or a subset of them). A fold directory should contain:

    history.csv
    metrics_per_epoch.csv
    setting.txt

The script searches both CSV files because the alpha trajectory and concept-error
history may be stored in either one. It can also combine runs found under several
roots, which is useful when fold1 is stored in an older tuning directory and
folds2--5 are stored in the final evaluation directory.

Outputs
-------
For each representative dataset--backbone configuration:

    <dataset>_<backbone>_alpha_trajectory.png
    <dataset>_<backbone>_concept_error_trajectory.png   [when available]

It also writes:

    alpha_trajectory_per_run.csv
    alpha_trajectory_summary.csv
    selected_run_paths.txt

No model checkpoints are loaded.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    backbone: str
    directory_name: str
    display_name: str
    config_hint: Optional[str] = None


DEFAULT_EXPERIMENTS = (
    ExperimentSpec(
        dataset="AwA2",
        backbone="efficientnetb0",
        directory_name="AwA2_efficientnetb0",
        display_name="AwA2 -- PACBM-Cl -- EfficientNetB0",
    ),
    ExperimentSpec(
        dataset="aPY",
        backbone="efficientnetb0",
        directory_name="aPY_efficientnetb0",
        display_name="aPY -- PACBM-Cl -- EfficientNetB0",
    ),
    ExperimentSpec(
        dataset="CUB",
        backbone="inceptionv3",
        directory_name="CUB_inceptionv3",
        display_name="CUB -- PACBM-Cl -- InceptionV3",
    ),
)


EPOCH_ALIASES = (
    "epoch",
    "epochs",
    "epoch_index",
    "epoch_number",
)

ALPHA_ALIASES = (
    "alpha",
    "alpha_t",
    "alpha_value",
    "current_alpha",
    "dynamic_alpha",
    "prior_alpha",
    "anchor_alpha",
    "anchoring_alpha",
    "anchoring_coefficient",
    "dynamic_prior_alpha",
)

SMOOTHED_SCORE_ALIASES = (
    "s_t",
    "smoothed_score",
    "smoothed_metric",
    "smoothed_error",
    "smoothed_concept_error",
    "concept_error_smoothed",
    "score_ema",
    "metric_ema",
    "error_ema",
    "running_score",
    "running_metric",
)

RAW_SCORE_ALIASES = (
    "hat_s_t",
    "raw_score",
    "raw_metric",
    "raw_error",
    "concept_error",
    "validation_concept_error",
    "val_concept_error",
    "val_macro_mse",
    "val_macro_bce",
    "macro_mse",
    "macro_bce",
    "val_attr_mse",
    "val_attribute_mse",
    "val_concept_mse",
    "val_concept_bce",
)

START_ALIASES_COMMON = (
    "s_start",
    "start_score",
    "start_error",
    "start_threshold",
    "alpha_start_threshold",
)

END_ALIASES_COMMON = (
    "s_end",
    "end_score",
    "end_error",
    "end_threshold",
    "alpha_end_threshold",
)

RHO_ALIASES = (
    "rho",
    "score_rho",
    "metric_rho",
    "smoothing_rho",
    "score_ema_rho",
    "metric_ema_rho",
)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def natural_key(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def fold_number(path: Path) -> Optional[int]:
    match = re.fullmatch(r"fold[_-]?(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_setting_file(path: Path) -> dict[str, float | str]:
    """Parse common key=value, key:value, and command-line style settings."""
    values: dict[str, float | str] = {}
    if not path.exists():
        return values

    text = path.read_text(encoding="utf-8", errors="replace")

    # key=value or key: value
    for match in re.finditer(
        r"(?im)^\s*([A-Za-z][A-Za-z0-9_.-]*)\s*[:=]\s*([^\s,#;]+)",
        text,
    ):
        key = normalize_name(match.group(1))
        raw = match.group(2).strip()
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw

    # --key value or --key=value
    for match in re.finditer(
        r"--([A-Za-z][A-Za-z0-9_.-]*)\s*(?:=|\s)\s*([^\s]+)",
        text,
    ):
        key = normalize_name(match.group(1))
        raw = match.group(2).strip()
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw

    # Human-readable threshold lines used by the PACBM training scripts.
    readable_patterns = {
        "startmse": r"(?im)^\s*Start\s+MSE\s+value\s+for\s+alpha\s+update\s*:\s*([-+0-9.eE]+)",
        "endmse": r"(?im)^\s*End\s+MSE\s+value\s+for\s+alpha\s+update\s*:\s*([-+0-9.eE]+)",
        "startbce": r"(?im)^\s*Start\s+BCE\s+value\s+for\s+alpha\s+update\s*:\s*([-+0-9.eE]+)",
        "endbce": r"(?im)^\s*End\s+BCE\s+value\s+for\s+alpha\s+update\s*:\s*([-+0-9.eE]+)",
    }
    for key, pattern in readable_patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = float(match.group(1))

    return values


def lookup_setting(
    settings: dict[str, float | str],
    aliases: Iterable[str],
) -> Optional[float]:
    normalized_aliases = [normalize_name(alias) for alias in aliases]

    for alias in normalized_aliases:
        if alias in settings:
            try:
                return float(settings[alias])
            except (TypeError, ValueError):
                pass

    # Conservative suffix match, useful for names such as callback_s_start.
    for key, value in settings.items():
        if any(key.endswith(alias) for alias in normalized_aliases):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


def threshold_aliases(dataset: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if dataset.lower() == "awa2":
        return (
            START_ALIASES_COMMON + ("start_mse", "mse_start"),
            END_ALIASES_COMMON + ("end_mse", "mse_end"),
        )
    return (
        START_ALIASES_COMMON + ("start_bce", "bce_start"),
        END_ALIASES_COMMON + ("end_bce", "bce_end"),
    )


def read_numeric_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc

    # Remove fully empty columns, which often appear after trailing commas.
    frame = frame.dropna(axis=1, how="all")
    return frame


def find_column(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    *,
    kind: str,
) -> Optional[str]:
    if frame.empty:
        return None

    normalized = {column: normalize_name(column) for column in frame.columns}
    alias_norm = [normalize_name(alias) for alias in aliases]

    # Exact normalized match.
    for alias in alias_norm:
        for column, norm in normalized.items():
            if norm == alias:
                return column

    # Conservative semantic fallback.
    candidates: list[tuple[int, str]] = []
    for column, norm in normalized.items():
        score = 0

        if kind == "alpha":
            if "alpha" in norm and "loss" not in norm:
                score += 10
            if "anchor" in norm or "prior" in norm:
                score += 2

        elif kind == "smoothed":
            if any(token in norm for token in ("smooth", "running", "ema")):
                score += 7
            if any(token in norm for token in ("score", "metric", "error", "mse", "bce")):
                score += 3
            if "alpha" in norm or "loss" in norm:
                score -= 8

        elif kind == "raw":
            if any(token in norm for token in ("concept", "attr", "attribute")):
                score += 4
            if any(token in norm for token in ("error", "mse", "bce", "metric", "score")):
                score += 4
            if norm.startswith("val"):
                score += 2
            if any(token in norm for token in ("smooth", "running", "ema", "alpha", "loss")):
                score -= 8

        elif kind == "epoch":
            if "epoch" in norm:
                score += 10

        if score > 0:
            candidates.append((score, column))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], natural_key(item[1])))
    return candidates[0][1]


def merge_history_files(fold_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Merge history.csv and metrics_per_epoch.csv by epoch."""
    candidate_paths = [
        fold_dir / "history.csv",
        fold_dir / "metrics_per_epoch.csv",
    ]
    existing = [path for path in candidate_paths if path.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No history.csv or metrics_per_epoch.csv found in {fold_dir}"
        )

    merged: Optional[pd.DataFrame] = None

    for index, path in enumerate(existing):
        frame = read_numeric_csv(path)
        epoch_col = find_column(frame, EPOCH_ALIASES, kind="epoch")

        if epoch_col is None:
            frame = frame.copy()
            frame["__epoch__"] = np.arange(1, len(frame) + 1, dtype=int)
            epoch_col = "__epoch__"

        frame[epoch_col] = pd.to_numeric(frame[epoch_col], errors="coerce")
        frame = frame.dropna(subset=[epoch_col]).copy()
        frame[epoch_col] = frame[epoch_col].astype(int)
        frame = frame.rename(columns={epoch_col: "epoch"})

        # Avoid column-name collisions when both CSVs contain the same metric.
        if merged is not None:
            renames = {
                column: f"{column}__file{index + 1}"
                for column in frame.columns
                if column != "epoch" and column in merged.columns
            }
            frame = frame.rename(columns=renames)

        merged = frame if merged is None else merged.merge(
            frame,
            on="epoch",
            how="outer",
            sort=True,
        )

    assert merged is not None
    merged = merged.sort_values("epoch").drop_duplicates("epoch", keep="last")
    return merged.reset_index(drop=True), existing


def choose_metric_column(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    *,
    kind: str,
) -> Optional[str]:
    column = find_column(frame, aliases, kind=kind)
    if column is None:
        return None

    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    return column


def ema(values: np.ndarray, rho: float) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if valid_indices.size == 0:
        return result

    first = int(valid_indices[0])
    result[first] = values[first]

    for index in range(first + 1, len(values)):
        if np.isfinite(values[index]):
            previous = result[index - 1]
            if not np.isfinite(previous):
                previous = values[index]
            result[index] = rho * previous + (1.0 - rho) * values[index]
        else:
            result[index] = result[index - 1]

    return result


def reconstruct_schedule_score(
    raw_values: np.ndarray,
    *,
    start: float,
    rho: float,
) -> np.ndarray:
    """
    Reconstruct the score used by the alpha callback.

    The saved alpha at epoch t uses the validation concept error measured at
    epoch t-1. The score is initialized at s_start, so alpha_0 = 0.
    """
    raw = np.asarray(raw_values, dtype=float)
    score = np.full(raw.shape, np.nan, dtype=float)
    if raw.size == 0:
        return score

    score[0] = float(start)
    for index in range(1, len(raw)):
        previous_raw = raw[index - 1]
        if np.isfinite(previous_raw):
            score[index] = (
                float(rho) * score[index - 1]
                + (1.0 - float(rho)) * previous_raw
            )
        else:
            score[index] = score[index - 1]

    return score


def reconstruct_alpha(
    smoothed_score: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    denominator = start - end
    if denominator <= 0:
        raise ValueError(
            f"Expected start > end, received start={start}, end={end}"
        )
    return np.clip((start - smoothed_score) / denominator, 0.0, 1.0)


def find_dataset_directories(root: Path, directory_name: str) -> list[Path]:
    matches: list[Path] = []
    direct = root / directory_name
    if direct.is_dir():
        matches.append(direct)

    if root.is_dir():
        for path in root.rglob(directory_name):
            if path.is_dir() and path not in matches:
                matches.append(path)

    return sorted(matches, key=lambda path: natural_key(str(path)))


def config_groups_under(dataset_dir: Path) -> dict[Path, dict[int, Path]]:
    """
    Return configuration parent -> {fold_number: fold_directory}.

    A configuration parent is the directory directly containing fold1/fold2/etc.
    """
    groups: dict[Path, dict[int, Path]] = {}

    for path in dataset_dir.rglob("*"):
        if not path.is_dir():
            continue
        number = fold_number(path)
        if number is None:
            continue
        groups.setdefault(path.parent, {})[number] = path

    return groups


def path_similarity(reference: Path, candidate: Path) -> float:
    return difflib.SequenceMatcher(
        None,
        normalize_name(str(reference)),
        normalize_name(str(candidate)),
    ).ratio()


def select_primary_group(
    groups: dict[Path, dict[int, Path]],
    *,
    hint: Optional[str],
    dataset: str,
) -> tuple[Path, dict[int, Path]]:
    filtered = groups

    if hint:
        hint_lower = hint.lower()
        filtered = {
            parent: folds
            for parent, folds in groups.items()
            if hint_lower in str(parent).lower()
        }

    if not filtered:
        available = "\n".join(f"  - {path}" for path in groups)
        raise RuntimeError(
            f"No configuration group matched {dataset!r}."
            + (f" Hint: {hint!r}." if hint else "")
            + f"\nAvailable groups:\n{available}"
        )

    ranking = sorted(
        filtered.items(),
        key=lambda item: (
            -len(item[1]),
            natural_key(str(item[0])),
        ),
    )
    best_count = len(ranking[0][1])
    tied = [item for item in ranking if len(item[1]) == best_count]

    if len(tied) > 1 and not hint:
        listing = "\n".join(
            f"  - {parent}  folds={sorted(folds)}"
            for parent, folds in tied
        )
        raise RuntimeError(
            f"Several equally plausible configurations were found for {dataset}.\n"
            f"Use the dataset-specific --*-config-hint option.\n{listing}"
        )

    return ranking[0]


def discover_folds(
    roots: list[Path],
    spec: ExperimentSpec,
    *,
    expected_runs: int,
) -> tuple[dict[int, Path], Path]:
    """
    Select the most complete configuration from the first root, then fill missing
    folds from later roots using path similarity. Every selected path is printed.
    """
    primary_parent: Optional[Path] = None
    selected: dict[int, Path] = {}

    for root_index, root in enumerate(roots):
        dataset_dirs = find_dataset_directories(root, spec.directory_name)
        all_groups: dict[Path, dict[int, Path]] = {}
        for dataset_dir in dataset_dirs:
            all_groups.update(config_groups_under(dataset_dir))

        if not all_groups:
            continue

        if primary_parent is None:
            parent, folds = select_primary_group(
                all_groups,
                hint=spec.config_hint,
                dataset=spec.dataset,
            )
            primary_parent = parent
            selected.update(folds)
            continue

        missing = {
            number
            for number in range(1, expected_runs + 1)
            if number not in selected
        }
        if not missing:
            break

        candidates = []
        for parent, folds in all_groups.items():
            available_missing = missing.intersection(folds)
            if not available_missing:
                continue
            candidates.append(
                (
                    len(available_missing),
                    path_similarity(primary_parent, parent),
                    parent,
                    folds,
                )
            )

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                natural_key(str(item[2])),
            )
        )
        _, similarity, parent, folds = candidates[0]

        print(
            f"[info] {spec.dataset}: supplementing runs from {parent} "
            f"(path similarity={similarity:.3f})"
        )
        for number in sorted(missing.intersection(folds)):
            selected[number] = folds[number]

    if primary_parent is None:
        searched = "\n".join(f"  - {root}" for root in roots)
        raise RuntimeError(
            f"Could not find {spec.directory_name} below:\n{searched}"
        )

    return dict(sorted(selected.items())), primary_parent


def load_run(
    fold_dir: Path,
    *,
    dataset: str,
    fallback_rho: float,
    start_override: Optional[float],
    end_override: Optional[float],
) -> dict[str, object]:
    frame, source_paths = merge_history_files(fold_dir)

    settings = parse_setting_file(fold_dir / "setting.txt")
    start_aliases, end_aliases = threshold_aliases(dataset)

    start = start_override
    if start is None:
        start = lookup_setting(settings, start_aliases)

    end = end_override
    if end is None:
        end = lookup_setting(settings, end_aliases)

    # rho is the smoothing factor of the validation concept-error score.
    # It is distinct from the prior-EMA momentum encoded in some directory names.
    rho = lookup_setting(settings, RHO_ALIASES)
    if rho is None:
        rho = fallback_rho

    alpha_col = choose_metric_column(
        frame,
        ALPHA_ALIASES,
        kind="alpha",
    )

    # Use the exact validation macro metric used by the dynamic schedule.
    preferred_raw = (
        "val_concept_mse_macro"
        if dataset.lower() == "awa2"
        else "val_concept_bce_macro"
    )
    raw_col = next(
        (
            column
            for column in frame.columns
            if normalize_name(column) == normalize_name(preferred_raw)
        ),
        None,
    )

    epochs = frame["epoch"].to_numpy(dtype=int)
    alpha_values: Optional[np.ndarray] = None
    raw_values: Optional[np.ndarray] = None
    smoothed_values: Optional[np.ndarray] = None
    alpha_source = ""
    alpha_reconstruction_max_abs_diff = np.nan

    if alpha_col is not None:
        alpha_values = pd.to_numeric(
            frame[alpha_col],
            errors="coerce",
        ).to_numpy(dtype=float)
        alpha_source = f"saved column {alpha_col!r}"

    if raw_col is not None:
        raw_values = pd.to_numeric(
            frame[raw_col],
            errors="coerce",
        ).to_numpy(dtype=float)

    if start is None or end is None:
        columns = ", ".join(map(str, frame.columns))
        raise RuntimeError(
            f"Could not recover s_start/s_end for {fold_dir}. "
            f"Use the dataset-specific --*-start and --*-end options.\n"
            f"Available columns: {columns}"
        )

    if raw_values is None:
        columns = ", ".join(map(str, frame.columns))
        raise RuntimeError(
            f"Expected schedule metric {preferred_raw!r} in {fold_dir}, "
            f"but it was not found.\nAvailable columns: {columns}"
        )

    smoothed_values = reconstruct_schedule_score(
        raw_values,
        start=float(start),
        rho=float(rho),
    )
    reconstructed_alpha = reconstruct_alpha(
        smoothed_values,
        float(start),
        float(end),
    )

    if alpha_values is None:
        alpha_values = reconstructed_alpha
        alpha_source = (
            f"reconstructed from {raw_col!r} with "
            f"start={start}, end={end}, rho={rho}"
        )
    else:
        finite = np.isfinite(alpha_values) & np.isfinite(reconstructed_alpha)
        if np.any(finite):
            alpha_reconstruction_max_abs_diff = float(
                np.max(
                    np.abs(
                        alpha_values[finite]
                        - reconstructed_alpha[finite]
                    )
                )
            )
            if alpha_reconstruction_max_abs_diff > 5e-5:
                print(
                    f"[warning] {fold_dir}: saved alpha and reconstructed "
                    f"alpha differ by up to "
                    f"{alpha_reconstruction_max_abs_diff:.6g}."
                )

    valid_alpha = np.isfinite(alpha_values)
    if valid_alpha.sum() == 0:
        raise RuntimeError(f"Alpha contains no finite values in {fold_dir}")

    return {
        "fold_dir": fold_dir,
        "epoch": epochs,
        "alpha": alpha_values,
        "smoothed_score": smoothed_values,
        "raw_score": raw_values,
        "start": start,
        "end": end,
        "rho": rho,
        "alpha_source": alpha_source,
        "source_paths": source_paths,
        "alpha_column": alpha_col,
        "smoothed_column": f"EMA({raw_col}, one-epoch lag)",
        "raw_column": raw_col,
        "alpha_reconstruction_max_abs_diff":
            alpha_reconstruction_max_abs_diff,
    }

def aligned_matrix(
    runs: list[dict[str, object]],
    key: str,
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    epoch_sets: list[set[int]] = []
    series_by_run: list[dict[int, float]] = []

    for run in runs:
        values = run.get(key)
        if values is None:
            continue

        epochs = np.asarray(run["epoch"], dtype=int)
        values_array = np.asarray(values, dtype=float)
        finite = np.isfinite(values_array)
        mapping = {
            int(epoch): float(value)
            for epoch, value in zip(epochs[finite], values_array[finite])
        }
        if not mapping:
            continue

        epoch_sets.append(set(mapping))
        series_by_run.append(mapping)

    if not series_by_run:
        raise ValueError(f"No usable values for {key}")

    if policy == "common":
        selected_epochs = set.intersection(*epoch_sets)
    elif policy == "available":
        selected_epochs = set.union(*epoch_sets)
    else:
        raise ValueError(f"Unknown epoch policy: {policy}")

    if not selected_epochs:
        raise ValueError(
            f"No aligned epochs remain for {key} under policy={policy!r}"
        )

    epochs = np.array(sorted(selected_epochs), dtype=int)
    matrix = np.full((len(series_by_run), len(epochs)), np.nan, dtype=float)

    for row, mapping in enumerate(series_by_run):
        for column, epoch in enumerate(epochs):
            if epoch in mapping:
                matrix[row, column] = mapping[epoch]

    return epochs, matrix


def summarize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    return mean, std


def save_figure(fig: plt.Figure, output_stem: Path, dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_alpha(
    spec: ExperimentSpec,
    runs: list[dict[str, object]],
    *,
    output_dir: Path,
    epoch_policy: str,
    show_folds: bool,
    dpi: int,
) -> None:
    epochs, matrix = aligned_matrix(runs, "alpha", policy=epoch_policy)
    mean, std = summarize_matrix(matrix)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    if show_folds:
        for row in matrix:
            ax.plot(epochs, row, linewidth=0.8, alpha=0.28)

    mean_line, = ax.plot(
        epochs,
        mean,
        linewidth=2.2,
        label="Mean anchoring coefficient",
    )
    ax.fill_between(
        epochs,
        mean - std,
        mean + std,
        alpha=0.18,
        color=mean_line.get_color(),
        label="Mean ± one standard deviation",
    )

    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.axhline(1.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"Anchoring coefficient $\alpha_t$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(spec.display_name)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    stem = output_dir / f"{spec.dataset}_{spec.backbone}_alpha_trajectory"
    save_figure(fig, stem, dpi)


def plot_smoothed_score(
    spec: ExperimentSpec,
    runs: list[dict[str, object]],
    *,
    output_dir: Path,
    epoch_policy: str,
    show_folds: bool,
    dpi: int,
) -> bool:
    usable = [run for run in runs if run.get("smoothed_score") is not None]
    if not usable:
        return False

    epochs, matrix = aligned_matrix(
        usable,
        "smoothed_score",
        policy=epoch_policy,
    )
    mean, std = summarize_matrix(matrix)

    starts = [
        float(run["start"])
        for run in usable
        if run.get("start") is not None
    ]
    ends = [
        float(run["end"])
        for run in usable
        if run.get("end") is not None
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    if show_folds:
        for row in matrix:
            ax.plot(epochs, row, linewidth=0.8, alpha=0.28)

    mean_line, = ax.plot(
        epochs,
        mean,
        linewidth=2.2,
        label=r"Mean smoothed concept error $s_t$",
    )
    ax.fill_between(
        epochs,
        mean - std,
        mean + std,
        alpha=0.18,
        color=mean_line.get_color(),
        label="Mean ± one standard deviation",
    )

    if starts:
        start = float(np.median(starts))
        ax.axhline(
            start,
            linewidth=1.2,
            linestyle="--",
            label=rf"$s_{{\mathrm{{start}}}}={start:g}$",
        )

    if ends:
        end = float(np.median(ends))
        ax.axhline(
            end,
            linewidth=1.2,
            linestyle=":",
            label=rf"$s_{{\mathrm{{end}}}}={end:g}$",
        )

    ax.set_xlabel("Training epoch")
    if spec.dataset.lower() == "awa2":
        ax.set_ylabel("Smoothed macro concept MSE")
    else:
        ax.set_ylabel("Smoothed macro concept BCE")
    ax.set_title(spec.display_name)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    stem = output_dir / f"{spec.dataset}_{spec.backbone}_concept_error_trajectory"
    save_figure(fig, stem, dpi)
    return True


def first_epoch(
    epochs: np.ndarray,
    values: np.ndarray,
    predicate,
) -> float:
    valid = np.isfinite(values)
    indices = np.flatnonzero(valid & predicate(values))
    if indices.size == 0:
        return np.nan
    return float(epochs[int(indices[0])])


def run_summary(
    spec: ExperimentSpec,
    fold: int,
    run: dict[str, object],
) -> dict[str, object]:
    epochs = np.asarray(run["epoch"], dtype=int)
    alpha_values = np.asarray(run["alpha"], dtype=float)
    valid = np.isfinite(alpha_values)

    if valid.sum() == 0:
        raise RuntimeError(f"No finite alpha values for {run['fold_dir']}")

    valid_epochs = epochs[valid]
    valid_alpha = alpha_values[valid]

    onset = first_epoch(
        valid_epochs,
        valid_alpha,
        lambda values: values > 1e-8,
    )
    full_anchor = first_epoch(
        valid_epochs,
        valid_alpha,
        lambda values: values >= 1.0 - 1e-8,
    )
    mixed_fraction = float(
        np.mean((valid_alpha > 1e-8) & (valid_alpha < 1.0 - 1e-8))
    )

    return {
        "dataset": spec.dataset,
        "backbone": spec.backbone,
        "fold": fold,
        "run_directory": str(run["fold_dir"]),
        "n_epochs": int(len(valid_epochs)),
        "first_epoch": int(valid_epochs[0]),
        "last_epoch": int(valid_epochs[-1]),
        "transition_onset_epoch": onset,
        "full_prediction_anchor_epoch": full_anchor,
        "final_alpha": float(valid_alpha[-1]),
        "minimum_alpha": float(np.nanmin(valid_alpha)),
        "maximum_alpha": float(np.nanmax(valid_alpha)),
        "mixed_regime_fraction": mixed_fraction,
        "s_start": run.get("start"),
        "s_end": run.get("end"),
        "rho": run.get("rho"),
        "alpha_source": run.get("alpha_source"),
        "alpha_column": run.get("alpha_column"),
        "smoothed_score_column": run.get("smoothed_column"),
        "raw_score_column": run.get("raw_column"),
        "alpha_reconstruction_max_abs_diff":
            run.get("alpha_reconstruction_max_abs_diff"),
    }


def aggregate_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "transition_onset_epoch",
        "full_prediction_anchor_epoch",
        "final_alpha",
        "mixed_regime_fraction",
    )

    rows: list[dict[str, object]] = []
    for (dataset, backbone), group in per_run.groupby(
        ["dataset", "backbone"],
        sort=False,
    ):
        row: dict[str, object] = {
            "dataset": dataset,
            "backbone": backbone,
            "n_runs": int(len(group)),
        }
        for column in metric_columns:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(numeric.mean())
            row[f"{column}_std"] = float(numeric.std(ddof=0))
            row[f"{column}_n"] = int(numeric.notna().sum())
        rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PACBM dynamic-alpha trajectories for the representative "
            "AwA2, aPY, and CUB configurations."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path("trained_models/pacbm_cl_native_20260804/pacbm_cl"),
            Path("trained_models/pacbm_2/pacbm/pacbm_final_5fold_selected_old_dynamic"),
        ],
        help=(
            "Experiment roots searched in order. The first root supplies the "
            "primary configuration; later roots may supply missing folds."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("alpha_trajectory_plots_native"),
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--epoch-policy",
        choices=("common", "available"),
        default="common",
        help=(
            "'common' plots only epochs present in every run; 'available' "
            "uses all epochs and averages over the runs still available."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.9,
        help=(
            "Fallback smoothing coefficient used only when rho is absent from "
            "setting.txt and a smoothed score must be reconstructed."
        ),
    )
    parser.add_argument(
        "--show-folds",
        action="store_true",
        help="Draw individual fold trajectories behind the mean curve.",
    )
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--awa2-config-hint", default=None)
    parser.add_argument("--apy-config-hint", default=None)
    parser.add_argument("--cub-config-hint", default=None)

    parser.add_argument("--awa2-start", type=float, default=None)
    parser.add_argument("--awa2-end", type=float, default=None)
    parser.add_argument("--apy-start", type=float, default=None)
    parser.add_argument("--apy-end", type=float, default=None)
    parser.add_argument("--cub-start", type=float, default=None)
    parser.add_argument("--cub-end", type=float, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [path.expanduser().resolve() for path in args.roots]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hint_by_dataset = {
        "AwA2": args.awa2_config_hint,
        "aPY": args.apy_config_hint,
        "CUB": args.cub_config_hint,
    }
    start_override = {
        "AwA2": args.awa2_start,
        "aPY": args.apy_start,
        "CUB": args.cub_start,
    }
    end_override = {
        "AwA2": args.awa2_end,
        "aPY": args.apy_end,
        "CUB": args.cub_end,
    }

    specs = [
        ExperimentSpec(
            dataset=spec.dataset,
            backbone=spec.backbone,
            directory_name=spec.directory_name,
            display_name=spec.display_name,
            config_hint=hint_by_dataset[spec.dataset],
        )
        for spec in DEFAULT_EXPERIMENTS
    ]

    all_summaries: list[dict[str, object]] = []
    selected_path_lines: list[str] = []

    for spec in specs:
        print(f"\n=== {spec.display_name} ===")

        fold_dirs, primary_parent = discover_folds(
            roots,
            spec,
            expected_runs=args.expected_runs,
        )

        print(f"[info] Primary configuration: {primary_parent}")
        if len(fold_dirs) < args.expected_runs:
            print(
                f"[warning] {spec.dataset}: found {len(fold_dirs)} run(s), "
                f"expected {args.expected_runs}. Plotting the available runs."
            )

        runs: list[dict[str, object]] = []

        selected_path_lines.append(f"[{spec.dataset}_{spec.backbone}]")
        selected_path_lines.append(f"primary_config={primary_parent}")

        for fold, fold_dir in sorted(fold_dirs.items()):
            print(f"[info] fold{fold}: {fold_dir}")
            selected_path_lines.append(f"fold{fold}={fold_dir}")

            run = load_run(
                fold_dir,
                dataset=spec.dataset,
                fallback_rho=args.rho,
                start_override=start_override[spec.dataset],
                end_override=end_override[spec.dataset],
            )
            print(
                f"       alpha: {run['alpha_source']}; "
                f"smoothed={run['smoothed_column']!r}; "
                f"raw={run['raw_column']!r}; "
                f"start={run['start']}; end={run['end']}; rho={run['rho']}"
            )

            runs.append(run)
            all_summaries.append(run_summary(spec, fold, run))

        selected_path_lines.append("")

        if not runs:
            raise RuntimeError(f"No usable runs found for {spec.dataset}")

        plot_alpha(
            spec,
            runs,
            output_dir=output_dir,
            epoch_policy=args.epoch_policy,
            show_folds=args.show_folds,
            dpi=args.dpi,
        )

        score_written = plot_smoothed_score(
            spec,
            runs,
            output_dir=output_dir,
            epoch_policy=args.epoch_policy,
            show_folds=args.show_folds,
            dpi=args.dpi,
        )
        if not score_written:
            print(
                f"[warning] {spec.dataset}: no concept-error trajectory was "
                f"available, so only the alpha plot was written."
            )

    per_run = pd.DataFrame(all_summaries)
    per_run.to_csv(
        output_dir / "alpha_trajectory_per_run.csv",
        index=False,
    )

    aggregate = aggregate_summary(per_run)
    aggregate.to_csv(
        output_dir / "alpha_trajectory_summary.csv",
        index=False,
    )

    (output_dir / "selected_run_paths.txt").write_text(
        "\n".join(selected_path_lines),
        encoding="utf-8",
    )

    print(f"\nDone. Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        raise
