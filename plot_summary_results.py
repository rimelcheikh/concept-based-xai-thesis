import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D

from matplotlib.ticker import MaxNLocator



plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 12,
})

MODEL_LABELS = {
    "jointcbm": "JointCBM",
    "joint_cbm": "JointCBM",
    "klcbm": "KL-CBM",
    "kl-cbm": "KL-CBM",
    "pacbm": "PACBM-Cl",
    "pacbm_2": "PACBM-Co",
}

MODEL_ORDER = ["JointCBM", "KL-CBM", "PACBM-Cl", "PACBM-Co"]

BACKBONE_ORDER = ["mobilenetv2", "inceptionv3", "efficientnetb0"]

DATASET_ORDER = ["AwA2", "aPY", "CUB"]


# ---------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------

def load_csvs(paths, name):
    dfs = []

    for p in paths:
        p = Path(p)

        if not p.exists():
            print(f"[warn] missing {name} CSV, skipping: {p}")
            continue

        df = pd.read_csv(p)
        df["_source_csv"] = str(p)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No valid {name} CSV files found.")

    return pd.concat(dfs, ignore_index=True)


def normalize_model_names(df):
    df = df.copy()
    df["model_raw"] = df["model"].astype(str)
    df["model_key"] = df["model_raw"].str.lower()
    df["model_label"] = df["model_key"].map(MODEL_LABELS).fillna(df["model_raw"])
    return df


def ordered_items(values, preferred_order):
    values = list(values)
    ordered = [v for v in preferred_order if v in values]
    ordered += sorted([v for v in values if v not in ordered])
    return ordered


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def collapse_configs(df, metric_cols):
    """
    Your aggregated CSV may still have one row per config.
    This collapses configs to one row per model/dataset/backbone by averaging.
    If you only have one selected config, this changes nothing.
    """
    group_cols = ["model_label", "dataset", "backbone"]

    existing = [c for c in metric_cols if c in df.columns]

    if not existing:
        return pd.DataFrame(columns=group_cols)

    out = (
        df.groupby(group_cols, as_index=False)[existing]
        .mean(numeric_only=True)
    )

    return out


# ---------------------------------------------------------------------
# Metric selection
# ---------------------------------------------------------------------

def get_base_metric_specs(dataset):
    """
    First figure:
    - all datasets: accuracy, balanced accuracy
    - AwA2: concept MSE
    - aPY/CUB: micro/macro F1 and micro/macro AUC
    """
    specs = [
        {
            "key": "accuracy",
            "label": "Accuracy ↑",
            "candidates": ["accuracy_mean", "accuracy"],
        },
        {
            "key": "balanced_accuracy",
            "label": "Balanced accuracy ↑",
            "candidates": ["balanced_accuracy_mean", "balanced_accuracy"],
        },
    ]

    if dataset == "AwA2":
        specs += [
            {
                "key": "concept_mse",
                "label": "Concept MSE ↓",
                "candidates": [
                    "concept_micro_mse_mean",
                    "concept_macro_mse_mean",
                    "concept_micro_mse",
                    "concept_macro_mse",
                ],
            }
        ]
    else:
        specs += [
            {
                "key": "concept_micro_f1",
                "label": "Micro F1 ↑",
                "candidates": ["concept_micro_f1_mean", "concept_micro_f1"],
            },
            {
                "key": "concept_macro_f1",
                "label": "Macro F1 ↑",
                "candidates": ["concept_macro_f1_mean", "concept_macro_f1"],
            },
            {
                "key": "concept_micro_auc",
                "label": "Micro AUC ↑",
                "candidates": ["concept_micro_auc_mean", "concept_micro_auc"],
            },
            {
                "key": "concept_macro_auc",
                "label": "Macro AUC ↑",
                "candidates": ["concept_macro_auc_mean", "concept_macro_auc"],
            },
        ]

    return specs


def get_intervention_metric_specs():
    """
    Extra metrics for the second figure.
    These come from intervention == oracle_all.
    """
    return [
        {
            "key": "oracle_accuracy",
            "label": "Full oracle accuracy ↑",
            "candidates": ["intervened_accuracy_mean", "intervened_accuracy"],
        },
        {
            "key": "accuracy_change",
            "label": "Accuracy change ↑",
            "candidates": ["accuracy_change_mean", "accuracy_change"],
        },
        {
            "key": "correction_rate",
            "label": "Correction rate ↑",
            "candidates": ["correction_rate_mean", "correction_rate"],
        },
        {
            "key": "degradation_rate",
            "label": "Degradation rate ↓",
            "candidates": ["degradation_rate_mean", "degradation_rate"],
        },
    ]


# ---------------------------------------------------------------------
# Reshaping
# ---------------------------------------------------------------------

def prepare_faithfulness_summary(faithfulness_df):
    faithfulness_df = normalize_model_names(faithfulness_df)

    all_possible_cols = [
        "accuracy_mean", "accuracy",
        "balanced_accuracy_mean", "balanced_accuracy",
        "concept_micro_mse_mean", "concept_micro_mse",
        "concept_macro_mse_mean", "concept_macro_mse",
        "concept_micro_f1_mean", "concept_micro_f1",
        "concept_macro_f1_mean", "concept_macro_f1",
        "concept_micro_auc_mean", "concept_micro_auc",
        "concept_macro_auc_mean", "concept_macro_auc",
    ]

    return collapse_configs(faithfulness_df, all_possible_cols)


def prepare_oracle_summary(intervention_df):
    intervention_df = normalize_model_names(intervention_df)

    if "intervention" not in intervention_df.columns:
        raise ValueError("Intervention CSV must contain an 'intervention' column.")

    oracle = intervention_df[intervention_df["intervention"] == "oracle_all"].copy()

    all_possible_cols = [
        "intervened_accuracy_mean", "intervened_accuracy",
        "accuracy_change_mean", "accuracy_change",
        "correction_rate_mean", "correction_rate",
        "degradation_rate_mean", "degradation_rate",
    ]

    return collapse_configs(oracle, all_possible_cols)


def build_plot_table(df, dataset, backbone, metric_specs):
    sub = df[
        (df["dataset"] == dataset)
        & (df["backbone"] == backbone)
    ].copy()

    rows = []

    for _, r in sub.iterrows():
        model = r["model_label"]

        for spec in metric_specs:
            col = find_col(sub, spec["candidates"])

            if col is None:
                continue

            value = r[col]

            if pd.isna(value):
                continue

            rows.append({
                "model": model,
                "metric": spec["label"],
                "value": float(value),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_grouped_metric_bars(plot_df, title, out_path):
    if plot_df.empty:
        print(f"[warn] empty plot, skipping: {out_path}")
        return

    metrics = list(plot_df["metric"].drop_duplicates())
    models = ordered_items(plot_df["model"].drop_duplicates(), MODEL_ORDER)

    y = np.arange(len(metrics))
    n_models = len(models)

    bar_height = 0.8 / max(n_models, 1)

    plt.figure(figsize=(11, max(4.5, 0.6 * len(metrics) + 2)))

    values_all = []

    for i, model in enumerate(models):
        mdf = plot_df[plot_df["model"] == model]

        vals = []
        for metric in metrics:
            row = mdf[mdf["metric"] == metric]
            vals.append(row["value"].iloc[0] if not row.empty else np.nan)

        vals = np.asarray(vals, dtype=float)
        values_all.extend(vals[~np.isnan(vals)].tolist())

        offset = (i - (n_models - 1) / 2) * bar_height
        plt.barh(y + offset, vals, height=bar_height, label=model)

        for yy, v in zip(y + offset, vals):
            if not np.isnan(v):
                plt.text(
                    v,
                    yy,
                    f" {v:.3f}",
                    va="center",
                )

    plt.yticks(y, metrics)
    plt.xlabel("Metric value")
    plt.title(title)
    plt.axvline(0, linewidth=0.8)
    plt.grid(axis="x", alpha=0.25)
    plt.legend(loc="best")

    if values_all:
        min_v = min(values_all)
        max_v = max(values_all)

        # Most metrics are in [0, 1], but accuracy_change can be negative.
        left = min(0.0, min_v - 0.05)
        right = max(1.0, max_v + 0.08)

        plt.xlim(left, right)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_path}")


def make_base_figures(faithfulness_summary, out_dir):
    out_dir = Path(out_dir)

    datasets = ordered_items(faithfulness_summary["dataset"].unique(), DATASET_ORDER)

    for dataset in datasets:
        backbones = ordered_items(
            faithfulness_summary[faithfulness_summary["dataset"] == dataset]["backbone"].unique(),
            BACKBONE_ORDER,
        )

        metric_specs = get_base_metric_specs(dataset)

        for backbone in backbones:
            plot_df = build_plot_table(
                faithfulness_summary,
                dataset=dataset,
                backbone=backbone,
                metric_specs=metric_specs,
            )

            title = f"{dataset} / {backbone}: classification and concept quality"
            out_path = out_dir / "base_summary" / f"{dataset}_{backbone}_base_summary.png"

            plot_grouped_metric_bars(plot_df, title, out_path)


def make_oracle_figures(faithfulness_summary, oracle_summary, out_dir):
    out_dir = Path(out_dir)

    merged = faithfulness_summary.merge(
        oracle_summary,
        on=["model_label", "dataset", "backbone"],
        how="left",
        suffixes=("", "_oracle"),
    )

    datasets = ordered_items(merged["dataset"].unique(), DATASET_ORDER)

    for dataset in datasets:
        backbones = ordered_items(
            merged[merged["dataset"] == dataset]["backbone"].unique(),
            BACKBONE_ORDER,
        )

        metric_specs = get_base_metric_specs(dataset) + get_intervention_metric_specs()

        for backbone in backbones:
            plot_df = build_plot_table(
                merged,
                dataset=dataset,
                backbone=backbone,
                metric_specs=metric_specs,
            )

            title = f"{dataset} / {backbone}: quality + full oracle intervention"
            out_path = out_dir / "oracle_summary" / f"{dataset}_{backbone}_oracle_summary.png"

            plot_grouped_metric_bars(plot_df, title, out_path)


# ---------------------------------------------------------------------
# 2D scatter plots: balanced accuracy vs other metrics
# ---------------------------------------------------------------------

def get_2d_metric_specs(dataset):
    """
    2D figures:
    x = balanced accuracy
    y = each other metric.
    """
    specs = [
        {
            "key": "accuracy",
            "label": "Accuracy",
            "candidates": ["accuracy_mean", "accuracy"],
        },
    ]

    if dataset == "AwA2":
        specs += [
            {
                "key": "concept_mse",
                "label": "Concept MSE",
                "candidates": [
                    "concept_micro_mse_mean",
                    "concept_macro_mse_mean",
                    "concept_micro_mse",
                    "concept_macro_mse",
                ],
            }
        ]
    else:
        specs += [
            {
                "key": "concept_micro_f1",
                "label": "Concept micro F1",
                "candidates": ["concept_micro_f1_mean", "concept_micro_f1"],
            },
            {
                "key": "concept_macro_f1",
                "label": "Concept macro F1",
                "candidates": ["concept_macro_f1_mean", "concept_macro_f1"],
            },
            {
                "key": "concept_micro_auc",
                "label": "Concept micro AUC",
                "candidates": ["concept_micro_auc_mean", "concept_micro_auc"],
            },
            {
                "key": "concept_macro_auc",
                "label": "Concept macro AUC",
                "candidates": ["concept_macro_auc_mean", "concept_macro_auc"],
            },
        ]

    return specs


def plot_2d_balacc_vs_metric(df, dataset, backbone, metric_spec, out_path):
    sub = df[
        (df["dataset"] == dataset)
        & (df["backbone"] == backbone)
    ].copy()

    if sub.empty:
        return

    bal_col = find_col(sub, ["balanced_accuracy_mean", "balanced_accuracy"])
    y_col = find_col(sub, metric_spec["candidates"])

    if bal_col is None or y_col is None:
        print(
            f"[warn] missing columns for {dataset}/{backbone}: "
            f"bal_col={bal_col}, y_col={y_col}"
        )
        return

    sub = sub.dropna(subset=[bal_col, y_col])

    if sub.empty:
        return

    plt.figure(figsize=(6.2, 5.2))

    for model in ordered_items(sub["model_label"].unique(), MODEL_ORDER):
        m = sub[sub["model_label"] == model]

        if m.empty:
            continue

        x = m[bal_col].iloc[0]
        y = m[y_col].iloc[0]

        plt.scatter(x, y, s=90, label=model)
        plt.text(x, y, f" {model}", va="center")

    plt.xlabel("Balanced accuracy")
    plt.ylabel(metric_spec["label"])
    #plt.title(f"{dataset} / {backbone}: balanced accuracy vs {metric_spec['label']}")

    plt.grid(True, alpha=0.3)

    # Balanced accuracy is normally in [0, 1]
    x_min = max(0.0, sub[bal_col].min() - 0.03)
    x_max = min(1.0, sub[bal_col].max() + 0.03)
    plt.xlim(x_min, x_max)

    y_min = sub[y_col].min()
    y_max = sub[y_col].max()
    y_pad = 0.05 * max(abs(y_max - y_min), 1e-6)

    if metric_spec["key"] == "concept_mse":
        # For MSE, lower is better, but keep natural vertical axis.
        plt.ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    else:
        plt.ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_path}")


def make_2d_figures(faithfulness_summary, out_dir):
    out_dir = Path(out_dir)

    datasets = ordered_items(faithfulness_summary["dataset"].unique(), DATASET_ORDER)

    for dataset in datasets:
        backbones = ordered_items(
            faithfulness_summary[faithfulness_summary["dataset"] == dataset]["backbone"].unique(),
            BACKBONE_ORDER,
        )

        metric_specs = get_2d_metric_specs(dataset)

        for backbone in backbones:
            for metric_spec in metric_specs:
                out_path = (
                    out_dir
                    / "2d_balacc_tradeoffs"
                    / dataset
                    / f"{dataset}_{backbone}_balacc_vs_{metric_spec['key']}.png"
                )

                plot_2d_balacc_vs_metric(
                    df=faithfulness_summary,
                    dataset=dataset,
                    backbone=backbone,
                    metric_spec=metric_spec,
                    out_path=out_path,
                )


# ---------------------------------------------------------------------
# One global 2D figure: all dataset/backbone combinations
# ---------------------------------------------------------------------

BACKBONE_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "inceptionv3": "InceptionV3",
    "efficientnetb0": "EfficientNetB0",
}

MODEL_MARKERS = {
    "JointCBM": "o",
    "KL-CBM": "s",
    "PACBM-Cl": "^",
    "PACBM-Co": "D",
}


def get_backbone_colors(backbones):
    backbones = ordered_items(backbones, BACKBONE_ORDER)
    cmap = plt.get_cmap("tab10")

    return {
        backbone: cmap(i)
        for i, backbone in enumerate(backbones)
    }


def format_backbone(backbone):
    return BACKBONE_LABELS.get(backbone, backbone)


def short_model_label(model):
    return {
        "JointCBM": "J",
        "KL-CBM": "K",
        "PACBM-Cl": "P-Cl",
        "PACBM-Co": "P-Co",
    }.get(model, model)


def make_global_2d_all_combos_figure(faithfulness_summary, out_dir):
    """
    One global figure with all dataset/metric panels.

    Each subplot:
    - one dataset
    - one y-axis metric
    - x-axis = balanced accuracy

    Each point:
    - one model/backbone combination

    Color = backbone
    Marker = model
    """
    out_dir = Path(out_dir)

    bal_col = find_col(
        faithfulness_summary,
        ["balanced_accuracy_mean", "balanced_accuracy"],
    )

    if bal_col is None:
        raise ValueError("Missing balanced accuracy column.")

    panels = []

    for dataset in ordered_items(faithfulness_summary["dataset"].unique(), DATASET_ORDER):
        for spec in get_2d_metric_specs(dataset):
            panels.append({
                "dataset": dataset,
                "metric_key": spec["key"],
                "metric_label": spec["label"],
                "metric_candidates": spec["candidates"],
            })

    n_panels = len(panels)
    n_cols = 4
    n_rows = math.ceil(n_panels / n_cols)

    all_backbones = ordered_items(
        faithfulness_summary["backbone"].unique(),
        BACKBONE_ORDER,
    )
    backbone_colors = get_backbone_colors(all_backbones)
    save_shared_tradeoff_legend(
        backbone_colors=backbone_colors,
        out_dir=out_dir,
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.6 * n_rows),
        squeeze=False,
    )

    axes = axes.ravel()

    single_panel_data = []

    for ax_idx, panel in enumerate(panels):
        ax = axes[ax_idx]

        dataset = panel["dataset"]
        y_col = find_col(faithfulness_summary, panel["metric_candidates"])

        if y_col is None:
            ax.axis("off")
            ax.set_title(f"{dataset}: {panel['metric_label']}\nmissing metric")
            continue

        sub = faithfulness_summary[
            faithfulness_summary["dataset"] == dataset
        ].copy()

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        sub = sub.dropna(subset=[bal_col, y_col])
        single_panel_data.append({
            "panel": panel,
            "sub": sub.copy(),
            "bal_col": bal_col,
            "y_col": y_col,
        })

        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{dataset}: {panel['metric_label']}\nno values")
            continue

        for _, r in sub.iterrows():
            model = r["model_label"]
            backbone = r["backbone"]

            x = r[bal_col]
            y = r[y_col]

            """ax.scatter(
                x,
                y,
                s=95,
                marker=MODEL_MARKERS.get(model, "o"),
                color=backbone_colors.get(backbone, "black"),
                edgecolor="black",
                linewidth=1.2,
                alpha=0.9,
            )"""
            ax.scatter(
                x,
                y,
                s=140,
                marker=MODEL_MARKERS.get(model, "o"),
                color=backbone_colors.get(backbone, "black"),
                edgecolor="black",
                linewidth=1.0,
            )

            ax.text(
                x,
                y,
                f" {short_model_label(model)}",
                va="center",
            )

        ax.set_title(f"{dataset}: BalAcc vs {panel['metric_label']}")
        ax.set_xlabel("Balanced accuracy")
        ax.set_ylabel(panel["metric_label"])
        ax.grid(True, alpha=0.3)

        # x limits
        x_min = sub[bal_col].min()
        x_max = sub[bal_col].max()
        x_pad = 0.05 * max(x_max - x_min, 1e-6)

        ax.set_xlim(
            max(0.0, x_min - x_pad),
            min(1.0, x_max + x_pad),
        )

        # y limits
        y_min = sub[y_col].min()
        y_max = sub[y_col].max()
        y_pad = 0.08 * max(y_max - y_min, 1e-6)

        if panel["metric_key"] == "concept_mse":
            ax.set_ylim(
                max(0.0, y_min - y_pad),
                y_max + y_pad,
            )
            ax.text(
                0.98,
                0.02,
                "best: bottom-right",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                alpha=0.75,
            )
        else:
            ax.set_ylim(
                max(0.0, y_min - y_pad),
                min(1.0, y_max + y_pad),
            )
            ax.text(
                0.98,
                0.02,
                "best: top-right",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                alpha=0.75,
            )

    # Hide unused subplots
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    # Backbone legend
    backbone_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label=format_backbone(backbone),
            markerfacecolor=backbone_colors[backbone],
            markeredgecolor="black",
            markersize=14,
        )
        for backbone in all_backbones
    ]

    # Model legend
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS.get(model, "o"),
            color="black",
            linestyle="None",
            label=model,
            markersize=14,
        )
        for model in MODEL_ORDER
        if model in faithfulness_summary["model_label"].unique()
    ]

    fig.legend(
        handles=backbone_handles,
        title="Backbone",
        loc="upper center",
        bbox_to_anchor=(0.35, 1.02),
        ncol=len(backbone_handles),
        frameon=False,
    )

    fig.legend(
        handles=model_handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.72, 1.02),
        ncol=len(model_handles),
        frameon=False,
    )

    fig.suptitle(
        "Backbone comparison across all datasets: balanced accuracy versus concept/class metrics",
        y=1.06,
    )

    fig.tight_layout()

    out_path = out_dir / "2d_balacc_tradeoffs_all_combos.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also save every subplot as its own separate figure
    for item in single_panel_data:
        save_single_global_panel(
            panel=item["panel"],
            sub=item["sub"],
            bal_col=item["bal_col"],
            y_col=item["y_col"],
            backbone_colors=backbone_colors,
            out_dir=out_dir,
        )

    print(f"[saved] {out_path}")


def save_single_global_panel(
    panel,
    sub,
    bal_col,
    y_col,
    backbone_colors,
    out_dir,
):
    """
    Save one mini-figure corresponding to one subplot
    from the global combo figure.
    """
    dataset = panel["dataset"]
    metric_key = panel["metric_key"]
    metric_label = panel["metric_label"]

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    for _, r in sub.iterrows():
        model = r["model_label"]
        backbone = r["backbone"]

        x = r[bal_col]
        y = r[y_col]

        ax.scatter(
            x,
            y,
            s=95,
            marker=MODEL_MARKERS.get(model, "o"),
            color=backbone_colors.get(backbone, "black"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )

        """if model in {"PACBM-Cl", "PACBM-Co"}:
            ax.annotate(
                short_model_label(model),
                xy=(x, y),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=9.5,
                va="center",
            )"""
        ax.annotate(
            short_model_label(model),
            xy=(x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=13,
            va="center",
        )
        """ax.text(
            x,
            y,
            f" {short_model_label(model)}",
            va="center",
        )"""

    #ax.set_title(f"{dataset}: BalAcc vs {metric_label}")
    ax.set_xlabel("Balanced accuracy")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)

    x_min = sub[bal_col].min()
    x_max = sub[bal_col].max()
    x_pad = 0.11 * max(x_max - x_min, 1e-6)

    ax.set_xlim(
        max(0.0, x_min - x_pad),
        min(1.0, x_max + x_pad),
    )

    y_min = sub[y_col].min()
    y_max = sub[y_col].max()
    y_pad = 0.08 * max(y_max - y_min, 1e-6)

    if metric_key == "concept_mse":
        ax.set_ylim(
            max(0.0, y_min - y_pad),
            y_max + y_pad,
        )
        """ax.text(
            0.98,
            0.02,
            "best: bottom-right",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            alpha=0.75,
        )"""
    else:
        ax.set_ylim(
            max(0.0, y_min - y_pad),
            min(1.0, y_max + y_pad),
        )
        """ax.text(
            0.98,
            0.02,
            "best: top-right",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            alpha=0.75,
        )"""


    # Legends
    used_backbones = ordered_items(sub["backbone"].unique(), BACKBONE_ORDER)
    used_models = ordered_items(sub["model_label"].unique(), MODEL_ORDER)

    backbone_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label=format_backbone(backbone),
            markerfacecolor=backbone_colors[backbone],
            markeredgecolor="black",
            markersize=12,
        )
        for backbone in used_backbones
    ]

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS.get(model, "o"),
            color="black",
            linestyle="None",
            label=model,
            markersize=12,
        )
        for model in used_models
    ]

    """ax.legend(
        handles=backbone_handles + model_handles,
        loc="best",
        frameon=True,
    )"""

    """if dataset == "AwA2" and metric_key == "concept_mse":
        ax.legend(
            handles=backbone_handles + model_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            fontsize=8,
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.4,
        )"""
    fig.tight_layout()

    out_path = (
        Path(out_dir)
        / "2d_balacc_tradeoffs_all_combos_separate"
        / dataset
        / f"{dataset}_all_backbones_balacc_vs_{metric_key}.png"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_path}")


def save_shared_tradeoff_legend(backbone_colors, out_dir):
    used_backbones = [
        backbone
        for backbone in BACKBONE_ORDER
        if backbone in backbone_colors
    ]

    backbone_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            label=format_backbone(backbone),
            markerfacecolor=backbone_colors[backbone],
            markeredgecolor="black",
            markersize=9,
        )
        for backbone in used_backbones
    ]

    model_handles = [
        Line2D(
            [0], [0],
            marker=MODEL_MARKERS[model],
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            color="black",
            label=model,
            markersize=9,
        )
        for model in MODEL_ORDER
    ]

    fig, ax = plt.subplots(figsize=(7.5, 1.15))
    ax.axis("off")

    backbone_legend = ax.legend(
        handles=backbone_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(backbone_handles),
        frameon=False,
        fontsize=9,
        columnspacing=1.3,
        handletextpad=0.4,
    )
    ax.add_artist(backbone_legend)

    ax.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(model_handles),
        frameon=False,
        fontsize=9,
        columnspacing=1.3,
        handletextpad=0.4,
    )

    out_path = (
        Path(out_dir)
        / "2d_balacc_tradeoffs_all_combos_separate"
        / "tradeoff_shared_legend.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    plt.close(fig)

    print(f"[saved] {out_path}")
# ---------------------------------------------------------------------
# One global 2D figure with mean ± std over backbones
# ---------------------------------------------------------------------

def aggregate_over_backbones_for_panel(sub, bal_col, y_col):
    """
    Aggregate each model over backbones:
    x = mean balanced accuracy over backbones
    xerr = std balanced accuracy over backbones
    y = mean metric over backbones
    yerr = std metric over backbones
    """
    rows = []

    for model in ordered_items(sub["model_label"].unique(), MODEL_ORDER):
        m = sub[sub["model_label"] == model].copy()

        if m.empty:
            continue

        x_vals = m[bal_col].dropna().values
        y_vals = m[y_col].dropna().values

        if len(x_vals) == 0 or len(y_vals) == 0:
            continue

        rows.append({
            "model_label": model,
            "x_mean": float(np.mean(x_vals)),
            "x_std": float(np.std(x_vals, ddof=1)) if len(x_vals) > 1 else 0.0,
            "y_mean": float(np.mean(y_vals)),
            "y_std": float(np.std(y_vals, ddof=1)) if len(y_vals) > 1 else 0.0,
            "n_backbones": len(m["backbone"].unique()),
        })

    return pd.DataFrame(rows)


def plot_panel_mean_std_over_backbones(
    ax,
    panel,
    sub,
    bal_col,
    y_col,
):
    """
    Plot one panel with one point per model.
    Error bars show std over backbones.
    """
    dataset = panel["dataset"]
    metric_key = panel["metric_key"]
    metric_label = panel["metric_label"]

    agg = aggregate_over_backbones_for_panel(sub, bal_col, y_col)

    if agg.empty:
        ax.axis("off")
        ax.set_title(f"{dataset}: {metric_label}\nno values")
        return agg

    for _, r in agg.iterrows():
        model = r["model_label"]

        ax.errorbar(
            r["x_mean"],
            r["y_mean"],
            xerr=r["x_std"],
            yerr=r["y_std"],
            fmt=MODEL_MARKERS.get(model, "o"),
            markersize=8,
            capsize=4,
            elinewidth=1.2,
            markeredgecolor="black",
            label=model,
        )

        ax.text(
            r["x_mean"],
            r["y_mean"],
            f" {short_model_label(model)}",
            va="center",
        )

    ax.set_title(f"{dataset}: BalAcc vs {metric_label}")
    ax.set_xlabel("Balanced accuracy")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)

    # x-axis limits with std included
    x_low = (agg["x_mean"] - agg["x_std"]).min()
    x_high = (agg["x_mean"] + agg["x_std"]).max()
    x_pad = 0.06 * max(x_high - x_low, 1e-6)

    ax.set_xlim(
        max(0.0, x_low - x_pad),
        min(1.0, x_high + x_pad),
    )

    # y-axis limits with std included
    y_low = (agg["y_mean"] - agg["y_std"]).min()
    y_high = (agg["y_mean"] + agg["y_std"]).max()
    y_pad = 0.08 * max(y_high - y_low, 1e-6)

    if metric_key == "concept_mse":
        ax.set_ylim(
            max(0.0, y_low - y_pad),
            y_high + y_pad,
        )
        ax.text(
            0.98,
            0.02,
            "best: bottom-right",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            alpha=0.75,
        )
    else:
        ax.set_ylim(
            max(0.0, y_low - y_pad),
            min(1.0, y_high + y_pad),
        )
        ax.text(
            0.98,
            0.02,
            "best: top-right",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            alpha=0.75,
        )

    return agg


def make_global_2d_mean_std_backbones_figure(faithfulness_summary, out_dir):
    """
    One global figure where each point is model mean over backbones.
    Error bars show std over backbones.

    This does NOT replace the existing all-combos figure.
    It saves a new figure with a different filename.
    """
    out_dir = Path(out_dir)

    bal_col = find_col(
        faithfulness_summary,
        ["balanced_accuracy_mean", "balanced_accuracy"],
    )

    if bal_col is None:
        raise ValueError("Missing balanced accuracy column.")

    panels = []

    for dataset in ordered_items(faithfulness_summary["dataset"].unique(), DATASET_ORDER):
        for spec in get_2d_metric_specs(dataset):
            panels.append({
                "dataset": dataset,
                "metric_key": spec["key"],
                "metric_label": spec["label"],
                "metric_candidates": spec["candidates"],
            })

    n_panels = len(panels)
    n_cols = 4
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.6 * n_rows),
        squeeze=False,
    )

    axes = axes.ravel()

    all_agg_rows = []

    for ax_idx, panel in enumerate(panels):
        ax = axes[ax_idx]

        dataset = panel["dataset"]
        y_col = find_col(faithfulness_summary, panel["metric_candidates"])

        if y_col is None:
            ax.axis("off")
            ax.set_title(f"{dataset}: {panel['metric_label']}\nmissing metric")
            continue

        sub = faithfulness_summary[
            faithfulness_summary["dataset"] == dataset
        ].copy()

        sub = sub.dropna(subset=[bal_col, y_col])

        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{dataset}: {panel['metric_label']}\nno values")
            continue

        agg = plot_panel_mean_std_over_backbones(
            ax=ax,
            panel=panel,
            sub=sub,
            bal_col=bal_col,
            y_col=y_col,
        )

        if not agg.empty:
            agg["dataset"] = dataset
            agg["metric_key"] = panel["metric_key"]
            agg["metric_label"] = panel["metric_label"]
            all_agg_rows.append(agg)

        # Save the same panel separately
        save_single_mean_std_backbones_panel(
            panel=panel,
            sub=sub,
            bal_col=bal_col,
            y_col=y_col,
            out_dir=out_dir,
        )

    # Hide unused subplots
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS.get(model, "o"),
            color="black",
            linestyle="None",
            label=model,
            markersize=9,
        )
        for model in MODEL_ORDER
        if model in faithfulness_summary["model_label"].unique()
    ]

    fig.legend(
        handles=model_handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(model_handles),
        frameon=False,
    )

    fig.suptitle(
        "Backbone-averaged trade-offs: mean ± std over backbones",
        y=1.06,
    )

    fig.tight_layout()

    out_path = out_dir / "2d_balacc_tradeoffs_mean_std_over_backbones.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if all_agg_rows:
        agg_df = pd.concat(all_agg_rows, ignore_index=True)
        csv_path = out_dir / "2d_balacc_tradeoffs_mean_std_over_backbones.csv"
        agg_df.to_csv(csv_path, index=False)
        print(f"[saved] {csv_path}")

    print(f"[saved] {out_path}")


def save_single_mean_std_backbones_panel(
    panel,
    sub,
    bal_col,
    y_col,
    out_dir,
):
    """
    Save one mini figure where each model is averaged over backbones.
    Error bars show std over backbones.
    """
    dataset = panel["dataset"]
    metric_key = panel["metric_key"]
    metric_label = panel["metric_label"]

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    plot_panel_mean_std_over_backbones(
        ax=ax,
        panel=panel,
        sub=sub,
        bal_col=bal_col,
        y_col=y_col,
    )

    used_models = ordered_items(sub["model_label"].unique(), MODEL_ORDER)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS.get(model, "o"),
            color="black",
            linestyle="None",
            label=model,
            markersize=8,
        )
        for model in used_models
    ]

    ax.legend(
        handles=model_handles,
        loc="best",
        frameon=True,
    )

    fig.tight_layout()

    out_path = (
        Path(out_dir)
        / "2d_balacc_tradeoffs_mean_std_over_backbones_separate"
        / dataset
        / f"{dataset}_mean_std_backbones_balacc_vs_{metric_key}.png"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--faithfulness_csv",
        nargs="+",
        default=[
            "faithfulness_results_native/faithfulness_aggregated.csv",
            #"faithfulness_results/2/more_k/faithfulness_aggregated.csv",
        ],
        help="One or more faithfulness_aggregated.csv files.",
    )

    parser.add_argument(
        "--intervention_csv",
        nargs="+",
        default=[
            "intervenability_metrics_native/intervenability_aggregated.csv",
            #"intervenability_metrics/2/intervenability_aggregated.csv",
        ],
        help="One or more intervenability_aggregated.csv files.",
    )

    parser.add_argument(
        "--out_dir",
        default="summary_result_figures_native",
        help="Output directory.",
    )

    args = parser.parse_args()

    faithfulness_df = load_csvs(args.faithfulness_csv, name="faithfulness")
    #intervention_df = load_csvs(args.intervention_csv, name="intervention")

    faithfulness_summary = prepare_faithfulness_summary(faithfulness_df)
    #oracle_summary = prepare_oracle_summary(intervention_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faithfulness_summary.to_csv(out_dir / "faithfulness_summary_collapsed.csv", index=False)
    #oracle_summary.to_csv(out_dir / "oracle_summary_collapsed.csv", index=False)

    #make_base_figures(faithfulness_summary, out_dir)
    #make_oracle_figures(faithfulness_summary, oracle_summary, out_dir)
    #make_2d_figures(faithfulness_summary, out_dir)
    make_global_2d_all_combos_figure(faithfulness_summary, out_dir)
    #make_global_2d_mean_std_backbones_figure(faithfulness_summary, out_dir)

    print("\nDone.")
    print(f"Saved figures under: {out_dir}")
    print(f"- {out_dir / 'base_summary'}")
    print(f"- {out_dir / 'oracle_summary'}")
    print(f"- {out_dir / '2d_balacc_tradeoffs'}")
    print(f"- {out_dir / '2d_balacc_tradeoffs_all_combos.png'}")
    print(f"- {out_dir / '2d_balacc_tradeoffs_mean_std_over_backbones.png'}")
    print(f"- {out_dir / '2d_balacc_tradeoffs_mean_std_over_backbones_separate'}")

if __name__ == "__main__":
    main()