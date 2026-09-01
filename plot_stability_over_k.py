import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.size": 14+5,
    "axes.labelsize": 16+5,
    "axes.titlesize": 16+5,
    "xtick.labelsize": 14+5,
    "ytick.labelsize": 14+5,
    "legend.fontsize": 14+5,
})

MODEL_LABELS = {
    "jointcbm": "JointCBM",
    "klcbm": "KL-CBM",
    "pacbm": "PACBM-Cl",
    "pacbm_2": "PACBM-Co",
}

MODEL_ORDER = ["JointCBM", "KL-CBM", "PACBM-Cl", "PACBM-Co"]

BACKBONE_ORDER = ["mobilenetv2", "inceptionv3", "efficientnetb0"]

METRICS = {
    "positive_jaccard": {
        "title": "Top-q positive Jaccard",
        "column": "top{k}_positive_jaccard",
        "ylabel": "Positive Jaccard",
        "start_k": 1,
    },
    "positive_spearman": {
        "title": "Top-q positive Spearman",
        "column": "top{k}_positive_spearman",
        "ylabel": "Positive Spearman",
        "start_k": 2,
    },
    "negative_jaccard": {
        "title": "Top-q negative Jaccard",
        "column": "top{k}_negative_jaccard",
        "ylabel": "Negative Jaccard",
        "start_k": 1,
    },
    "negative_spearman": {
        "title": "Top-q negative Spearman",
        "column": "top{k}_negative_spearman",
        "ylabel": "Negative Spearman",
        "start_k": 2,
    },
    "absolute_spearman": {
        "title": "Top-k absolute Spearman",
        "column": "spearman_at{k}",
        "ylabel": "Absolute Spearman",
        "start_k": 2,
    },
}


def load_csvs(paths):
    dfs = []

    for p in paths:
        p = Path(p)

        if not p.exists():
            print(f"[warn] missing CSV, skipping: {p}")
            continue

        df = pd.read_csv(p)
        df["_source_csv"] = str(p)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No valid CSV files were found.")

    return pd.concat(dfs, ignore_index=True)


def filter_correct_weight_rows(df):
    """
    Avoid duplicate rows:
    - JointCBM / KL-CBM use concept_to_class_kernel
    - PACBM-Cl / PACBM-Co use gamma
    """
    df = df.copy()
    df["model"] = df["model"].astype(str).str.lower()

    keep = (
        (df["model"].isin(["jointcbm", "klcbm"]) & (df["weight_type"] == "concept_to_class_kernel"))
        |
        (df["model"].isin(["pacbm", "pacbm_2"]) & (df["weight_type"] == "gamma"))
    )

    df = df[keep].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df = df.dropna(subset=["model_label"])

    return df


def build_long_df(df, max_k=10):
    rows = []

    for _, r in df.iterrows():
        for metric_key, spec in METRICS.items():
            for k in range(spec["start_k"], max_k + 1):
                col = spec["column"].format(k=k)

                if col not in df.columns:
                    continue

                value = r[col]

                if pd.isna(value):
                    continue

                rows.append({
                    "model": r["model_label"],
                    "dataset": r["dataset"],
                    "backbone": r["backbone"],
                    "k": k,
                    "metric": metric_key,
                    "value": float(value),
                })

    return pd.DataFrame(rows)


def ordered_items(values, preferred_order):
    values = list(values)
    ordered = [v for v in preferred_order if v in values]
    ordered += sorted([v for v in values if v not in ordered])
    return ordered


def plot_dataset_avg_over_backbones(long_df, out_dir):
    """
    For each dataset and metric:
    one curve per model, averaged over backbones.
    This keeps the original mean-only figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(long_df["dataset"].unique()):
        for metric_key, spec in METRICS.items():
            sub = long_df[
                (long_df["dataset"] == dataset)
                & (long_df["metric"] == metric_key)
            ]

            if sub.empty:
                continue

            avg = (
                sub.groupby(["model", "k"], as_index=False)["value"]
                .mean()
            )

            plt.figure(figsize=(8.2, 5.2))

            for model in ordered_items(avg["model"].unique(), MODEL_ORDER):
                m = avg[avg["model"] == model].sort_values("k")

                if not m.empty:
                    plt.plot(m["k"], m["value"], marker="o", markersize=4, linewidth=1.8, label=model)

            #plt.title(f"{dataset}: {spec['title']} averaged over backbones")
            plt.xlabel("q (number of concepts)")
            plt.ylabel(spec["ylabel"])
            ax = plt.gca()
            ax.set_xlim(left=1)
            right = ax.get_xlim()[1]
            ticks = MaxNLocator(nbins=6, integer=True).tick_values(1, right)
            ticks = [int(t) for t in ticks if 1 <= t <= right]

            ax.set_xticks(sorted(set([1, *ticks])))

            plt.grid(True, alpha=0.3)
            #plt.legend()
            plt.tight_layout()

            fname = f"{dataset}_{metric_key}_avg_backbones.png"
            plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
            plt.close()


def plot_dataset_avg_over_backbones_with_std(long_df, out_dir):
    """
    For each dataset and metric:
    one curve per model, averaged over backbones.
    Shaded region = std over backbones.
    Saved with a different filename.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(long_df["dataset"].unique()):
        for metric_key, spec in METRICS.items():
            sub = long_df[
                (long_df["dataset"] == dataset)
                & (long_df["metric"] == metric_key)
            ]

            if sub.empty:
                continue

            stats = (
                sub.groupby(["model", "k"], as_index=False)["value"]
                .agg(["mean", "std"])
                .reset_index()
            )

            stats["std"] = stats["std"].fillna(0.0)

            plt.figure(figsize=(8.2, 5.2))

            for model in ordered_items(stats["model"].unique(), MODEL_ORDER):
                m = stats[stats["model"] == model].sort_values("k")

                if m.empty:
                    continue

                x = m["k"].values
                y = m["mean"].values
                s = m["std"].values

                line, = plt.plot(
                    x,
                    y,
                    marker="o",
                    label=model,
                )

                color = line.get_color()

                plt.fill_between(
                    x,
                    y - s,
                    y + s,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )

            #plt.title(f"{dataset}: {spec['title']} averaged over backbones ± std")
            plt.xlabel("q (number of concepts)")
            plt.ylabel(spec["ylabel"])
            ax = plt.gca()
            ax.set_xlim(left=1)
            right = ax.get_xlim()[1]
            ticks = MaxNLocator(nbins=6, integer=True).tick_values(1, right)
            ticks = [int(t) for t in ticks if 1 <= t <= right]

            ax.set_xticks(sorted(set([1, *ticks])))
            plt.grid(True, alpha=0.3)
            #plt.legend()
            plt.tight_layout()

            fname = f"{dataset}_{metric_key}_avg_backbones_std_backbones.png"
            plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"[saved] {out_dir / fname}")


def plot_dataset_one_curve_per_backbone(long_df, out_dir):
    """
    For each dataset, model, and metric:
    one curve per backbone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(long_df["dataset"].unique()):
        for model in ordered_items(long_df["model"].unique(), MODEL_ORDER):
            for metric_key, spec in METRICS.items():
                sub = long_df[
                    (long_df["dataset"] == dataset)
                    & (long_df["model"] == model)
                    & (long_df["metric"] == metric_key)
                ]

                if sub.empty:
                    continue

                plt.figure(figsize=(8.2, 5.2))

                for backbone in ordered_items(sub["backbone"].unique(), BACKBONE_ORDER):
                    b = sub[sub["backbone"] == backbone].sort_values("k")

                    if not b.empty:
                        plt.plot(b["k"], b["value"], marker="o", label=backbone)

                #plt.title(f"{dataset} - {model}: {spec['title']} by backbone")
                plt.xlabel("q (number of concepts)")
                plt.ylabel(spec["ylabel"])
                ax = plt.gca()
                ax.set_xlim(left=1)
                right = ax.get_xlim()[1]
                ticks = MaxNLocator(nbins=6, integer=True).tick_values(1, right)
                ticks = [int(t) for t in ticks if 1 <= t <= right]

                ax.set_xticks(sorted(set([1, *ticks])))

                #plt.xticks(sorted(sub["k"].unique()))
                plt.grid(True, alpha=0.3)
                #plt.legend()
                plt.tight_layout()

                clean_model = model.replace("-", "").replace(" ", "_")
                fname = f"{dataset}_{clean_model}_{metric_key}_by_backbone.png"
                plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
                plt.close()


def plot_dataset_backbone_all_models(long_df, out_dir):
    """
    For each dataset, backbone, and metric:
    one curve per model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(long_df["dataset"].unique()):
        for backbone in ordered_items(long_df["backbone"].unique(), BACKBONE_ORDER):
            for metric_key, spec in METRICS.items():
                sub = long_df[
                    (long_df["dataset"] == dataset)
                    & (long_df["backbone"] == backbone)
                    & (long_df["metric"] == metric_key)
                ]

                if sub.empty:
                    continue

                plt.figure(figsize=(8.2, 5.2))

                for model in ordered_items(sub["model"].unique(), MODEL_ORDER):
                    m = sub[sub["model"] == model].sort_values("k")

                    if not m.empty:
                        plt.plot(m["k"], m["value"], marker="o", markersize=4, linewidth=1.8, label=model)

                #plt.title(f"{dataset} - {backbone}: {spec['title']} by model")
                plt.xlabel("q (number of concepts)")
                plt.ylabel(spec["ylabel"])
                ax = plt.gca()
                ax.set_xlim(left=1)
                right = ax.get_xlim()[1]
                ticks = MaxNLocator(nbins=6, integer=True).tick_values(1, right)
                ticks = [int(t) for t in ticks if 1 <= t <= right]

                ax.set_xticks(sorted(set([1, *ticks])))

                plt.grid(True, alpha=0.3)
                #plt.legend(loc="lower right")
                plt.tight_layout()

                fname = f"{dataset}_{backbone}_{metric_key}_all_models.png"
                plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
                plt.close()


from matplotlib.lines import Line2D

def save_shared_legend(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_handles = [
        # Models (colors)
        Line2D([0], [0], color="tab:blue",  lw=2, label="JointCBM"),
        Line2D([0], [0], color="tab:orange", lw=2, label="KL-CBM"),
        Line2D([0], [0], color="tab:green", lw=2, label="PACBM-Cl"),
        Line2D([0], [0], color="tab:red",   lw=2, label="PACBM-Co"),
    ]


    fig, ax = plt.subplots(figsize=(14, 1.4))
    ax.axis("off")

    model_legend = ax.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )

    ax.add_artist(model_legend)


    """fig.legend(
        handles=handles,
        loc="center",
        ncol=6,
        frameon=False,
        fontsize=10,
    )"""

    fig.savefig(
        out_dir / "shared_legend.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        nargs="+",
        default=["faithfulness_results_native/concept_weight_stability.csv"],
        help=(
            "One or more concept_weight_stability.csv files. "
            "Pass both main and /2 PACBM-Co CSVs."
        ),
    )

    parser.add_argument(
        "--out_dir",
        default="stability_k_plots_by_dataset_native/",
        help="Output directory for plots.",
    )

    parser.add_argument(
        "--max_k",
        type=int,
        default=112,
        help="Maximum k value to plot.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    avg_dir = out_dir / "dataset_avg_over_backbones"
    backbone_dir = out_dir / "dataset_one_curve_per_backbone"
    all_models_dir = out_dir / "dataset_backbone_all_models"

    df = load_csvs(args.csv)
    df = filter_correct_weight_rows(df)
    long_df = build_long_df(df, max_k=args.max_k)

    if long_df.empty:
        raise RuntimeError(
            "No valid stability metrics found. Check CSV columns and weight_type filtering."
        )

    #plot_dataset_avg_over_backbones(long_df, avg_dir)
    plot_dataset_avg_over_backbones_with_std(long_df, avg_dir)
    plot_dataset_one_curve_per_backbone(long_df, backbone_dir)
    plot_dataset_backbone_all_models(long_df, all_models_dir)

    long_df.to_csv(out_dir / "stability_over_k_long.csv", index=False)

    print("Models found:", sorted(long_df["model"].unique()))
    print("Saved:")
    print(f"- {avg_dir}")
    print(f"- {backbone_dir}")
    print(f"- {all_models_dir}")
    print(f"- {out_dir / 'stability_over_k_long.csv'}")

    save_shared_legend(out_dir)


if __name__ == "__main__":
    main()