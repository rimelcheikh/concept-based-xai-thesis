from pathlib import Path
import argparse
import math
from typing import Any

import pandas as pd


UNUSED = "--"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", type=str, required=True)
    parser.add_argument("--sheet", type=str, default="Tuning_Runs")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _is_unused(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if str(value).strip() == UNUSED:
        return True
    return False


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not _is_unused(v)}


def load_rows(tracker_path: str, sheet_name: str) -> list[dict[str, Any]]:
    df = pd.read_excel(tracker_path, sheet_name=sheet_name, header=1)
    return df.to_dict(orient="records")


def filter_rows(
    rows: list[dict[str, Any]],
    model: str | None = None,
    stage: str | None = None,
    dataset: str | None = None,
    backbone: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    out = rows
    if model:
        out = [r for r in out if str(r.get("model")) == model]
    if stage:
        out = [r for r in out if str(r.get("stage")) == stage]
    if dataset:
        out = [r for r in out if str(r.get("dataset")) == dataset]
    if backbone:
        out = [r for r in out if str(r.get("backbone")) == backbone]
    if run_id:
        out = [r for r in out if str(r.get("run_id")) == run_id]
    return out


def fold_ids_from_scope(fold_scope: str, run_from_one=False) -> list[int]:
    if run_from_one:
        mapping = {
        "1fold": [1],
        "2fold": [1, 2],
        "5fold": [1, 2, 3, 4, 5],
    }
    else:
        mapping = {
        "1fold": [1],
        "2fold": [1, 2],
        "5fold": [2, 3, 4, 5],
    }
    if fold_scope not in mapping:
        raise ValueError(f"Unknown fold_scope: {fold_scope}")
    return mapping[fold_scope]