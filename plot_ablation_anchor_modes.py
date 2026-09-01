from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASETS = ["AwA2", "aPY", "CUB"]

VARIANT_ORDER = [
    "no_anchor",
    "true_only",
    "pred_only",
    "fixed_half",
    "dynamic_metric",
]

VARIANT_LABELS = {
    "no_anchor": "No anchor",
    "true_only": "True prior",
    "pred_only": "Pred. prior",
    "fixed_half": r"Fixed $\alpha=0.5$",
    "dynamic_metric": "Dynamic",
}

SELECTED_BACKBONE = {
    "AwA2": "efficientnetb0",
    "aPY": "efficientnetb0",
    "CUB": "inceptionv3",
}

BACKBONE_ORDER = ["mobilenetv2", "inceptionv3", "efficientnetb0"]

BACKBONE_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "inceptionv3": "InceptionV3",
    "efficientnetb0": "EfficientNetB0",
}

DATASET_COLORS = {
    "AwA2": "tab:blue",
    "aPY": "tab:orange",
    "CUB": "tab:green",
}

BACKBONE_COLORS = {
    "mobilenetv2": "tab:blue",
    "inceptionv3": "tab:orange",
    "efficientnetb0": "tab:green",
}

# Offsets are (horizontal, vertical), in points. Balanced-accuracy labels are
# moved away from the gap between AwA2 and aPY, while the nearly coincident
# AwA2/aPY Jaccard labels are placed on opposite sides of their curves.
SELECTED_LABEL_OFFSETS = {
    "bal_acc_mean": {
        "AwA2": (-7, -15),
        "aPY": (7, 10),
        "CUB": (7, 10),
    },
    "pos_jacc10_mean": {
        "AwA2": (-7, 10),
        "aPY": (7, -15),
        "CUB": (7, 10),
    },
}

BACKBONE_LABEL_OFFSETS = {
    "mobilenetv2": (-10, 9),
    "inceptionv3": (0, -14),
    "efficientnetb0": (10, 9),
}

METRICS_MAIN = [
    ("bal_acc_mean", "bal_acc_std", "Bal. acc."),
    ("pos_jacc10_mean", "pos_jacc10_std", "Pos. top-10 Jac."),
]

METRICS_APPENDIX = [
    ("spearman_mean", "spearman_std", "Global Spearman correlation"),
    ("neg_jacc10_mean", "neg_jacc10_std", "Negative top-10 Jaccard"),
]

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 15,
    }
)


def annotate_values(
    ax,
    x,
    y,
    color: str,
    offset: tuple[int, int],
    decimals: int = 2,
) -> None:
    """Write the mean value beside every finite point in one curve."""
    for xi, yi in zip(x, y):
        if not np.isfinite(yi):
            continue

        ax.annotate(
            f"{yi:.{decimals}f}",
            (xi, yi),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=8,
            color=color,
        )


def get_ordered_rows(
    df: pd.DataFrame,
    backbone: str,
) -> pd.DataFrame:
    """Return one row per variant in the required plotting order."""
    sub = df[df["backbone"] == backbone].copy()

    duplicates = sub["variant"].astype(str).duplicated(keep=False)
    if duplicates.any():
        duplicate_names = sorted(sub.loc[duplicates, "variant"].astype(str).unique())
        raise ValueError(
            f"Duplicate rows for backbone={backbone}: {duplicate_names}"
        )

    sub["variant"] = sub["variant"].astype(str)
    sub = sub.set_index("variant").reindex(VARIANT_ORDER)

    missing_variants = sub.index[sub.isna().all(axis=1)].tolist()
    if missing_variants:
        raise ValueError(
            f"Missing variants for backbone={backbone}: {missing_variants}"
        )

    return sub


def require_columns(df: pd.DataFrame, columns, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {context}: {missing}")


def plot_selected_backbones_main(
    all_dfs: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Plot balanced accuracy and PosJacc@10 side by side."""
    x = np.arange(len(VARIANT_ORDER))
    x_labels = [VARIANT_LABELS[variant] for variant in VARIANT_ORDER]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.8),
        sharex=True,
    )

    for ax, (mean_col, std_col, ylabel) in zip(axes, METRICS_MAIN):
        for dataset in DATASETS:
            df = all_dfs[dataset]
            backbone = SELECTED_BACKBONE[dataset]
            require_columns(
                df,
                (mean_col, std_col),
                f"{dataset}/{backbone}",
            )
            sub = get_ordered_rows(df, backbone)

            y = sub[mean_col].astype(float).to_numpy()
            yerr = sub[std_col].astype(float).to_numpy()
            color = DATASET_COLORS[dataset]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=color,
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=f"{dataset} ({BACKBONE_LABELS[backbone]})",
            )

            """annotate_values(
                ax,
                x,
                y,
                color=color,
                offset=SELECTED_LABEL_OFFSETS[mean_col][dataset],
            )"""

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.margins(x=0.08, y=0.16)

    axes[0].set_title("(a) Balanced accuracy")
    axes[1].set_title(r"(b) Positive top-$10$ Jaccard")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset(
    df: pd.DataFrame,
    dataset: str,
    metrics,
    out_path: Path,
    title_suffix: str,
) -> None:
    """Plot two metrics for the backbones available for one dataset."""
    x = np.arange(len(VARIANT_ORDER))
    x_labels = [VARIANT_LABELS[variant] for variant in VARIANT_ORDER]

    available_backbones = [
        backbone
        for backbone in BACKBONE_ORDER
        if (df["backbone"].astype(str) == backbone).any()
    ]
    if not available_backbones:
        raise ValueError(f"No supported backbone rows found for dataset={dataset}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)

    for ax, (mean_col, std_col, ylabel) in zip(axes, metrics):
        require_columns(df, (mean_col, std_col), dataset)

        for backbone in available_backbones:
            sub = get_ordered_rows(df, backbone)
            y = sub[mean_col].astype(float).to_numpy()
            yerr = sub[std_col].astype(float).to_numpy()
            color = BACKBONE_COLORS[backbone]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=color,
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=BACKBONE_LABELS[backbone],
            )

            """annotate_values(
                ax,
                x,
                y,
                color=color,
                offset=BACKBONE_LABEL_OFFSETS[backbone],
            )"""

        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.margins(x=0.08, y=0.16)

    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle(f"{dataset} anchor-mode ablation: {title_suffix}", y=1.03)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    require_columns(df, ("status", "variant", "backbone"), str(csv_path))

    df = df[df["status"] == "OK"].copy()
    df = df[df["variant"].isin(VARIANT_ORDER)].copy()
    df = df[df["backbone"].isin(BACKBONE_ORDER)].copy()

    df["variant"] = pd.Categorical(
        df["variant"],
        categories=VARIANT_ORDER,
        ordered=True,
    )
    df["backbone"] = pd.Categorical(
        df["backbone"],
        categories=BACKBONE_ORDER,
        ordered=True,
    )

    return df.sort_values(["variant", "backbone"])


def save_shared_legend(out_dir: Path) -> None:
    """Save a standalone legend matching the combined selected-backbone plot."""
    out_dir.mkdir(parents=True, exist_ok=True)

    handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[dataset],
            marker="o",
            linewidth=2,
            label=(
                f"{dataset} "
                f"({BACKBONE_LABELS[SELECTED_BACKBONE[dataset]]})"
            ),
        )
        for dataset in DATASETS
    ]

    fig, ax = plt.subplots(figsize=(8.5, 0.7))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )

    fig.savefig(
        out_dir / "shared_legend.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def main() -> None:
    input_dir = Path("ablation_results_native")
    output_dir = input_dir / "ablation_figures"

    all_dfs: dict[str, pd.DataFrame] = {}

    for dataset in DATASETS:
        csv_path = input_dir / dataset / f"{dataset}_ablation_metrics_long.csv"

        if not csv_path.exists():
            print(f"[skip] Missing {csv_path}")
            continue

        df = prepare_df(csv_path)
        all_dfs[dataset] = df

        main_out = output_dir / f"{dataset}_anchor_ablation_main"
        appendix_out = output_dir / f"{dataset}_anchor_ablation_appendix"

        plot_dataset(
            df,
            dataset,
            METRICS_MAIN,
            main_out,
            "balanced accuracy and positive stability",
        )
        plot_dataset(
            df,
            dataset,
            METRICS_APPENDIX,
            appendix_out,
            "complementary stability metrics",
        )

        print(f"[ok] Wrote {main_out}.png")
        print(f"[ok] Wrote {appendix_out}.png")

    if set(DATASETS).issubset(all_dfs):
        selected_out = output_dir / "all_datasets_selected_backbones_main"
        plot_selected_backbones_main(all_dfs, selected_out)
        print(f"[ok] Wrote {selected_out}.png")
    else:
        missing = sorted(set(DATASETS) - set(all_dfs))
        print(
            "[skip] Combined selected-backbone figure; "
            f"missing datasets: {missing}"
        )

    save_shared_legend(output_dir)
    print(f"[ok] Wrote {output_dir / 'shared_legend.png'}")


if __name__ == "__main__":
    main()