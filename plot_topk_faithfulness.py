"""
Plot top-k concept neutralization faithfulness curves.

It combines:
- faithfulness_results/faithfulness_aggregated.csv
  for JointCBM, KL-CBM, PACBM-Cl
- faithfulness_results/2/faithfulness_aggregated.csv
  for PACBM-Co

Curves:
- JointCBM
- KL-CBM
- PACBM-Cl
- PACBM-Co
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 14+5,
    "axes.labelsize": 16+5,
    "axes.titlesize": 16+5,
    "xtick.labelsize": 14+5,
    "ytick.labelsize": 14+5,
    "legend.fontsize": 14+3,
})

MODELS = [
    {
        "source": "main",
        "row_model": "jointcbm",
        "prefix": "jointcbm",
        "label": "JointCBM",
    },
    {
        "source": "main",
        "row_model": "klcbm",
        "prefix": "klcbm_dense",
        "label": "KL-CBM",
    },
    {
        "source": "main",
        "row_model": "pacbm",
        "prefix": "pacbm",
        "label": "PACBM-Cl",
    },
    {
        "source": "co",
        "row_model": "pacbm_2",
        "prefix": "pacbm",
        "label": "PACBM-Co",
    },
]

MODEL_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(MODELS)]

METRICS = {
    "pred_class_probability_drop": "Pred.-class prob. drop",
    "accuracy_drop": "Accuracy drop",
}


def normalize_name(x):
    return str(x).lower().replace("-", "_").replace(" ", "_")


def load_csv(path, source_name):
    df = pd.read_csv(path).copy()

    extra = pd.DataFrame({
        "_source": source_name,
        "_model_norm": df["model"].apply(normalize_name),
    })

    return pd.concat([df, extra], axis=1)


def collect_curve(
    df,
    model_info,
    dataset,
    metric_key,
    max_k,
    backbone=None,
    return_std=False,
    random_order=False,
):
    source = model_info["source"]
    row_model = normalize_name(model_info["row_model"])
    prefix = model_info["prefix"]

    sub = df[
        (df["_source"] == source)
        & (df["_model_norm"] == row_model)
        & (df["dataset"] == dataset)
    ]

    if backbone is not None:
        sub = sub[sub["backbone"] == backbone]

    if sub.empty:
        if return_std:
            return [], [], []
        return [], []

    x_vals = []
    y_vals = []
    std_vals = []

    for k in range(max_k + 1):
        if random_order:
            column_prefix = (
                f"{prefix}_random_top{k}_neutralize"
            )
        else:
            column_prefix = (
                f"{prefix}_top{k}_neutralize"
            )

        col = f"{column_prefix}_{metric_key}_mean"

        if col not in sub.columns:
            continue

        vals = pd.to_numeric(
            sub[col],
            errors="coerce",
        ).dropna()

        if vals.empty:
            continue

        x_vals.append(k)
        y_vals.append(vals.mean())
        std_vals.append(
            vals.std(ddof=1) if len(vals) > 1 else 0.0
        )

    if return_std:
        return x_vals, y_vals, std_vals

    return x_vals, y_vals

def plot_dataset_curve(
    df,
    dataset,
    metric_key,
    output_dir,
    max_k=50,
    backbone=None,
    dpi=300,
):
    metric_label = METRICS[metric_key]

    plt.figure(figsize=(8.2, 5.2))

    found_any = False

    for model_index, model_info in enumerate(MODELS):
        color = MODEL_COLORS[model_index]
        x, y = collect_curve(
            df=df,
            model_info=model_info,
            dataset=dataset,
            metric_key=metric_key,
            max_k=max_k,
            backbone=backbone,
        )

        if len(x) == 0:
            print(
                f"[warn] No curve for {model_info['label']} "
                f"on {dataset}, backbone={backbone}"
            )
            continue

        found_any = True

        line, = plt.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"{model_info['label']} ranked",
        )

        random_x, random_y = collect_curve(
            df=df,
            model_info=model_info,
            dataset=dataset,
            metric_key=metric_key,
            max_k=max_k,
            backbone=backbone,
            random_order=True,
        )

        if len(random_x) > 0:
            plt.plot(
                random_x,
                random_y,
                linestyle="--",
                linewidth=1.8,
                color=color,
                alpha=0.75,
                label=f"{model_info['label']} random",
            )

    if not found_any:
        plt.close()
        print(f"[skip] No data for {dataset}, {metric_key}, backbone={backbone}")
        return None

    title = f"{dataset}: top-k neutralization"
    if backbone is not None:
        title += f" ({backbone})"

    plt.xlabel(
        "q (number of neutralized concepts)",
    )
    plt.ylabel(
        metric_label,
    )
    plt.xticks()
    plt.yticks()
    #plt.legend(loc='lower right')
    #plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    backbone_tag = "avg_backbones" if backbone is None else backbone
    filename = f"{dataset}_{backbone_tag}_topk{max_k}_{metric_key}.png"
    out_path = output_dir / filename

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_path}")
    return out_path


def plot_dataset_curve_with_std(
    df,
    dataset,
    metric_key,
    output_dir,
    max_k=50,
    backbone=None,
    dpi=300,
):
    """
    Same as plot_dataset_curve, but when backbone=None,
    shaded area shows std over backbones/runs available in the CSV rows.

    Saved under a different filename.
    """
    metric_label = METRICS[metric_key]

    plt.figure(figsize=(8.2, 5.2))

    found_any = False

    for model_index, model_info in enumerate(MODELS):
        color = MODEL_COLORS[model_index]
        x, y, s = collect_curve(
            df=df,
            model_info=model_info,
            dataset=dataset,
            metric_key=metric_key,
            max_k=max_k,
            backbone=backbone,
            return_std=True,
        )

        if len(x) == 0:
            print(
                f"[warn] No std curve for {model_info['label']} "
                f"on {dataset}, backbone={backbone}"
            )
            continue

        found_any = True

        x = np.asarray(x)
        y = np.asarray(y)
        s = np.asarray(s)

        plt.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=color,
            label="_nolegend_",
        )

        plt.fill_between(
            x,
            y - s,
            y + s,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        random_x, random_y = collect_curve(
            df=df,
            model_info=model_info,
            dataset=dataset,
            metric_key=metric_key,
            max_k=max_k,
            backbone=backbone,
            random_order=True,
        )

        if len(random_x) > 0:
            plt.plot(
                random_x,
                random_y,
                linestyle="--",
                linewidth=1.8,
                color=color,
                alpha=0.75,
                label="_nolegend_",
            )
 
    if not found_any:
        plt.close()
        print(f"[skip] No std data for {dataset}, {metric_key}, backbone={backbone}")
        return None

    title = f"{dataset}: top-k neutralization ± std"
    if backbone is not None:
        title += f" ({backbone})"

    plt.xlabel(
        "q (number of neutralized concepts)",
    )
    plt.ylabel(
        metric_label,
    )
    plt.xticks()
    plt.yticks()
    #plt.legend(loc='lower right')
    #plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    backbone_tag = "avg_backbones" if backbone is None else backbone
    filename = f"{dataset}_{backbone_tag}_topk{max_k}_{metric_key}_std_backbones.png"
    out_path = output_dir / filename

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_path}")
    return out_path

def save_shared_legend(output_dir, dpi=300):
    fig, ax = plt.subplots(figsize=(11.6, 1.4))
    ax.axis("off")

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[index],
            linewidth=2.5,
            label=model_info["label"],
        )
        for index, model_info in enumerate(MODELS)
    ]

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.5,
            linestyle="-",
            marker="o",
            markersize=4,
            label="Contribution-ranked",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.5,
            linestyle="--",
            label="Random order",
        ),
    ]

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

    ax.legend(
        handles=style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.5,
    )

    png_path = output_dir / "faithfulness_shared_legend.png"


    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )


    plt.close(fig)

    print(f"[saved] {png_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--main_csv",
        type=str,
        default="faithfulness_results_native/faithfulness_aggregated.csv",
        help="CSV containing JointCBM, KL-CBM, and PACBM-Cl results.",
    )

    parser.add_argument(
        "--co_csv",
        type=str,
        default="faithfulness_results_native/faithfulness_aggregated.csv",
        help="CSV containing PACBM-Co results.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="faithfulness_results_native/plots",
        help="Directory where plots will be saved.",
    )

    parser.add_argument(
        "--max_k",
        type=int,
        default=112,
        help="Maximum k value to plot.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to plot. If omitted, all datasets from both CSVs are used.",
    )

    parser.add_argument(
        "--per_backbone",
        default=False,
        action="store_true",
        help="If set, creates one plot per dataset/backbone instead of averaging across backbones.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_shared_legend(output_dir=output_dir, dpi=args.dpi,)

    main_df = load_csv(args.main_csv, "main")
    co_df = load_csv(args.co_csv, "co")

    df = pd.concat([main_df, co_df], ignore_index=True, sort=False)

    if args.datasets is None:
        datasets = sorted(df["dataset"].dropna().unique().tolist())
    else:
        datasets = args.datasets

    for dataset in datasets:
        ddf = df[df["dataset"] == dataset]

        if ddf.empty:
            print(f"[skip] Dataset not found: {dataset}")
            continue

        if args.per_backbone:
            backbones = sorted(ddf["backbone"].dropna().unique().tolist())
        else:
            backbones = [None]

        for backbone in backbones:
            for metric_key in METRICS:
                plot_dataset_curve(
                    df=df,
                    dataset=dataset,
                    metric_key=metric_key,
                    output_dir=output_dir,
                    max_k=args.max_k,
                    backbone=backbone,
                    dpi=args.dpi,
                )

                # Extra version with std shading.
                # Most useful when backbone=None, i.e. averaged over backbones.
                plot_dataset_curve_with_std(
                    df=df,
                    dataset=dataset,
                    metric_key=metric_key,
                    output_dir=output_dir,
                    max_k=args.max_k,
                    backbone=backbone,
                    dpi=args.dpi,
                )


if __name__ == "__main__":
    main()