import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.size": 19,
    "axes.labelsize": 21,
    "axes.titlesize": 21,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 17,
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

MODEL_COLORS = {
    "JointCBM": "tab:blue",
    "KL-CBM": "tab:orange",
    "PACBM-Cl": "tab:green",
    "PACBM-Co": "tab:red",
}

PACBM_MODEL_ORDER = ["PACBM-Cl", "PACBM-Co"]

PRESENT_ABSENT_INTERVENTIONS = {
    "partial_topk_present_supportive_oracle": {
        "label": "present-supportive",
        "linestyle": "-",
        "marker": "o",
        "alpha": 1.0,
    },
    "partial_topk_absent_supportive_oracle": {
        "label": "absent-supportive",
        "linestyle": "--",
        "marker": None,
        "alpha": 0.8,
    },
}

INTERVENTION_LABELS = {
    "partial_topk_present_supportive_oracle": "present concepts",
    "partial_topk_absent_supportive_oracle": "absent concepts",
}

TOPK_METRICS = {
    "accuracy_change": {
        "ylabel": "Accuracy change",
        "title": "",
        "higher_is_better": True,
    },
    "balanced_accuracy_change": {
        "ylabel": "Balanced accuracy change",
        "title": "",
        "higher_is_better": True,
    },
    "correction_rate": {
        "ylabel": "Correction rate",
        "title": "",
        "higher_is_better": True,
    },
    "degradation_rate": {
        "ylabel": "Degradation rate",
        "title": "",
        "higher_is_better": False,
    },
    "intervened_accuracy": {
        "ylabel": "Accuracy after intervention",
        "title": "",
        "higher_is_better": True,
    },
    "intervened_balanced_accuracy": {
        "ylabel": "Bal. acc. after intervention",
        "title": "",
        "higher_is_better": True,
    },
}


NEGATIVE_METRICS = {
    "accuracy_drop_on_originally_correct": {
        "ylabel": "Accuracy drop",
        "title": "Accuracy drop under targeted concept neutralization",
        "higher_is_better": True,
    },
    "predicted_class_probability_drop": {
        "ylabel": "Predicted-class probability drop",
        "title": "Probability drop under targeted concept neutralization",
        "higher_is_better": True,
    },
}


FULL_ORACLE_METRICS = {
    "original_accuracy": "Original accuracy",
    "intervened_accuracy": "Full oracle accuracy",
    "accuracy_change": "Accuracy change",
    "correction_rate": "Correction rate",
    "degradation_rate": "Degradation rate",
}


def ordered_items(values, preferred_order):
    values = list(values)
    ordered = [v for v in preferred_order if v in values]
    ordered += sorted([v for v in values if v not in ordered])
    return ordered


def normalize_model_name(x):
    x = str(x).lower()
    return MODEL_LABELS.get(x, str(x))


def clean_model_for_filename(model):
    return model.replace("-", "").replace(" ", "_")


def load_intervention_csvs(paths):
    dfs = []

    allowed_by_source = {
        "native": {"KL-CBM", "PACBM-Cl"},
        "legacy_joint": {"JointCBM"},
        "legacy_pacbm_co": {"PACBM-Co"},
    }

    for p in paths:
        p = Path(p)

        if not p.exists():
            print(f"[warn] missing CSV, skipping: {p}")
            continue

        path_string = p.as_posix().lower()

        # Identify the role of each CSV.
        if "intervenability_metrics_native/" in path_string:
            source_group = "native"
        elif "intervenability_metrics/2/" in path_string:
            source_group = "legacy_pacbm_co"
        elif "intervenability_metrics/more_k/" in path_string:
            source_group = "legacy_joint"
        else:
            raise ValueError(f"Unknown CSV source: {p}")

        df = pd.read_csv(p)

        df["model"] = df["model"].astype(str).str.lower()
        df["model_label"] = (
            df["model"]
            .map(MODEL_LABELS)
            .fillna(df["model"])
        )

        allowed_models = allowed_by_source[source_group]
        df = df[df["model_label"].isin(allowed_models)].copy()

        df["_source_group"] = source_group
        df["_source_csv"] = str(p)

        print(
            f"[load] {source_group}: "
            f"{len(df)} rows, "
            f"models={sorted(df['model_label'].unique())}"
        )

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            "No valid intervenability CSV files found."
        )

    df = pd.concat(dfs, ignore_index=True, sort=False)

    expected_models = {
        "JointCBM",
        "KL-CBM",
        "PACBM-Cl",
        "PACBM-Co",
    }
    found_models = set(df["model_label"].unique())
    missing_models = expected_models - found_models

    if missing_models:
        raise ValueError(
            f"Missing required models after source filtering: "
            f"{sorted(missing_models)}"
        )

    # Safety check: one model/configuration cannot come from multiple sources.
    key_columns = [
        c for c in [
            "dataset",
            "backbone",
            "model_label",
            "intervention",
            "k",
        ]
        if c in df.columns
    ]

    source_counts = (
        df.groupby(key_columns, dropna=False)["_source_group"]
        .nunique()
    )

    if (source_counts > 1).any():
        mixed = source_counts[source_counts > 1]
        raise ValueError(
            "Old and native results are still being mixed:\n"
            f"{mixed.head(20)}"
        )

    print("\nRows used by source and model:")
    print(pd.crosstab(df["_source_group"], df["model_label"]))

    return df


def metric_mean_col(metric):
    return f"{metric}_mean"


def metric_std_col(metric):
    return f"{metric}_std"


def get_metric_values(df, metric):
    """
    Accepts either:
    - metric_mean / metric_std from aggregated CSV
    - metric from a raw/per-fold CSV
    """
    mean_col = metric_mean_col(metric)
    std_col = metric_std_col(metric)

    if mean_col in df.columns:
        y = pd.to_numeric(df[mean_col], errors="coerce")
    elif metric in df.columns:
        y = pd.to_numeric(df[metric], errors="coerce")
    else:
        return None, None

    if std_col in df.columns:
        s = pd.to_numeric(df[std_col], errors="coerce").fillna(0.0)
    else:
        s = pd.Series(np.zeros(len(df)), index=df.index)

    return y, s


def aggregate_curve_over_backbones(df, metric):
    """
    Input rows may already be fold-aggregated.
    We average over available backbone/config rows.

    Output:
    model, k, mean, std
    where std is std over backbones/configs.
    """
    y, _ = get_metric_values(df, metric)

    if y is None:
        return pd.DataFrame()

    temp = df.copy()
    temp["_value"] = y
    temp = temp.dropna(subset=["_value"])

    if temp.empty:
        return pd.DataFrame()

    out = (
        temp.groupby(["model_label", "k"], as_index=False)["_value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    out["std"] = out["std"].fillna(0.0)
    return out


def plot_pacbm_present_vs_absent_one_backbone(
    df,
    dataset,
    backbone,
    metric,
    metric_info,
    out_dir,
    max_k=None,
):
    """
    One figure per dataset/backbone/metric.
    Only PACBM-Cl and PACBM-Co are shown.

    Solid lines = present-supportive intervention.
    Dashed lines = absent-supportive intervention.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interventions = list(PRESENT_ABSENT_INTERVENTIONS.keys())

    sub = df[
        (df["dataset"] == dataset)
        & (df["backbone"] == backbone)
        & (df["model_label"].isin(PACBM_MODEL_ORDER))
        & (df["intervention"].isin(interventions))
    ].copy()

    if max_k is not None:
        sub = sub[sub["k"] <= max_k]

    if sub.empty:
        print(f"[skip] no PACBM present/absent data for {dataset}, {backbone}, {metric}")
        return

    plt.figure(figsize=(7.8, 4.8))
    found_any = False

    for model in PACBM_MODEL_ORDER:
        color = MODEL_COLORS.get(model, None)

        for intervention, style in PRESENT_ABSENT_INTERVENTIONS.items():
            msub = sub[
                (sub["model_label"] == model)
                & (sub["intervention"] == intervention)
            ].copy()

            if msub.empty:
                continue

            y, _ = get_metric_values(msub, metric)
            if y is None:
                continue

            msub["_value"] = y
            msub = msub.dropna(subset=["_value"])

            if msub.empty:
                continue

            stats = (
                msub.groupby("k", as_index=False)["_value"]
                .mean()
                .sort_values("k")
            )

            if stats.empty:
                continue

            found_any = True

            plt.plot(
                stats["k"].values,
                stats["_value"].values,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3,
                linewidth=2.0,
                color=color,
                alpha=style["alpha"],
                label=f"{model} {style['label']}",
            )

    if not found_any:
        plt.close()
        print(f"[skip] no valid PACBM present/absent values for {dataset}, {backbone}, {metric}")
        return

    plt.title(f"{dataset} - {backbone}: {metric_info['title']} by evidence type")
    plt.xlabel("q (number of intervened concepts)")
    plt.ylabel(metric_info["ylabel"])

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    hint = "higher is better" if metric_info.get("higher_is_better", True) else "lower is better"
    ax.text(
        0.98,
        0.02,
        #"",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.75,
    )

    plt.grid(True, alpha=0.3)
    #plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    fname = f"{dataset}_{backbone}_{metric}_pacbm_present_vs_absent.png"
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_dir / fname}")


def make_pacbm_present_vs_absent_figures(df, out_dir, max_k=None):
    out_dir = Path(out_dir) / "pacbm_present_vs_absent"

    selected_backbones = {
        "AwA2": "efficientnetb0",
        "aPY": "efficientnetb0",
        "CUB": "inceptionv3",
    }

    for dataset, backbone in selected_backbones.items():
        for metric in ["accuracy_change", "correction_rate", "degradation_rate"]:
            info = TOPK_METRICS[metric]
            plot_pacbm_present_vs_absent_one_backbone(
                df=df,
                dataset=dataset,
                backbone=backbone,
                metric=metric,
                metric_info=info,
                out_dir=out_dir,
                max_k=max_k,
            )


def plot_present_vs_absent_one_backbone_all_models(
    df,
    dataset,
    backbone,
    metric,
    metric_info,
    out_dir,
    max_k=None,
):
    """
    One figure per dataset/backbone/metric.
    Solid lines = present-supportive intervention.
    Dashed lines = absent-supportive intervention.
    One color per model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interventions = [
        "partial_topk_present_supportive_oracle",
        "partial_topk_absent_supportive_oracle",
    ]

    sub = df[
        (df["dataset"] == dataset)
        & (df["backbone"] == backbone)
        & (df["intervention"].isin(interventions))
    ].copy()

    if max_k is not None:
        sub = sub[sub["k"] <= max_k]

    if sub.empty:
        print(f"[skip] no data for {dataset}, {backbone}, {metric}")
        return

    plt.figure(figsize=(7.8, 4.8))
    found_any = False

    for model in ordered_items(sub["model_label"].dropna().unique(), MODEL_ORDER):
        color = MODEL_COLORS.get(model, None)

        for intervention, linestyle in [
            ("partial_topk_present_supportive_oracle", "-"),
            ("partial_topk_absent_supportive_oracle", "--"),
        ]:
            msub = sub[
                (sub["model_label"] == model)
                & (sub["intervention"] == intervention)
            ].copy()

            if msub.empty:
                continue

            y, _ = get_metric_values(msub, metric)
            if y is None:
                continue

            msub["_value"] = y
            msub = msub.dropna(subset=["_value"])

            if msub.empty:
                continue

            stats = (
                msub.groupby("k", as_index=False)["_value"]
                .mean()
                .sort_values("k")
            )

            if stats.empty:
                continue

            found_any = True

            label_suffix = INTERVENTION_LABELS.get(intervention, intervention)

            plt.plot(
                stats["k"].values,
                stats["_value"].values,
                linestyle=linestyle,
                marker="o" if intervention == "partial_topk_present_supportive_oracle" else None,
                markersize=3,
                linewidth=2.0,
                color=color,
                alpha=1.0 if intervention == "partial_topk_present_supportive_oracle" else 0.75,
                label=f"{model} {label_suffix}",
            )

    if not found_any:
        plt.close()
        print(f"[skip] no valid metric values for {dataset}, {backbone}, {metric}")
        return

    plt.title(f"{dataset} - {backbone}: {metric_info['title']} (present vs absent)")
    plt.xlabel("Number of intervened concepts ($k$)")
    plt.ylabel(metric_info["ylabel"])

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    hint = "higher is better" if metric_info.get("higher_is_better", True) else "lower is better"
    ax.text(
        0.98,
        0.02,
        #"",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.75,
    )

    plt.grid(True, alpha=0.3)
    #plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    fname = f"{dataset}_{backbone}_{metric}_present_vs_absent.png"
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_dir / fname}")

def make_present_vs_absent_figures(df, out_dir, max_k=None):
    out_dir = Path(out_dir) / "present_vs_absent_per_backbone_all_models"

    for dataset in ordered_items(df["dataset"].dropna().unique(), DATASET_ORDER):
        dsub = df[df["dataset"] == dataset]

        for backbone in ordered_items(dsub["backbone"].dropna().unique(), BACKBONE_ORDER):
            for metric, info in TOPK_METRICS.items():
                plot_present_vs_absent_one_backbone_all_models(
                    df=df,
                    dataset=dataset,
                    backbone=backbone,
                    metric=metric,
                    metric_info=info,
                    out_dir=out_dir,
                    max_k=max_k,
                )


def plot_curves_avg_backbones(
    df,
    dataset,
    intervention,
    metric,
    metric_info,
    out_dir,
    max_k=None,
    compare_random=False,
):
    """
    One dataset/metric figure.
    One curve per model.
    Shading = std over backbones/config rows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df[
        (df["dataset"] == dataset)
        & (df["intervention"] == intervention)
    ].copy()

    if max_k is not None:
        sub = sub[sub["k"] <= max_k]

    if sub.empty:
        print(f"[skip] no data for {dataset}, {intervention}, {metric}")
        return

    plt.figure(figsize=(7.5, 4.6))

    found_any = False

    for model in ordered_items(sub["model_label"].unique(), MODEL_ORDER):
        msub = sub[sub["model_label"] == model].copy()
        stats = aggregate_curve_over_backbones(msub, metric)

        if stats.empty:
            continue

        found_any = True
        stats = stats.sort_values("k")

        x = stats["k"].values
        y = stats["mean"].values
        s = stats["std"].values

        line, = plt.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=model,
        )

        MODEL_COLORS = {
            "JointCBM": "tab:blue",
            "KL-CBM": "tab:orange",
            "PACBM-Cl": "tab:green",
            "PACBM-Co": "tab:red",
        }

        color = MODEL_COLORS.get(model, None)

        plt.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=2.0,
            color=color,
            label=f"{model} top-k",
        )

    if compare_random:
        rand = df[
            (df["dataset"] == dataset)
            & (df["intervention"] == "random_oracle")
        ].copy()

        if max_k is not None:
            rand = rand[rand["k"] <= max_k]

        for model in ordered_items(rand["model_label"].unique(), MODEL_ORDER):
            rsub = rand[rand["model_label"] == model].copy()
            stats = aggregate_curve_over_backbones(rsub, metric)

            if stats.empty:
                continue

            stats = stats.sort_values("k")

            color = MODEL_COLORS.get(model, None)

            plt.plot(
                stats["k"].values,
                stats["mean"].values,
                linestyle="--",
                marker=None,
                linewidth=2.0,
                color=color,
                alpha=0.65,
                label=f"{model} random",
            )

    if not found_any:
        plt.close()
        print(f"[skip] no valid metric values for {dataset}, {intervention}, {metric}")
        return

    title = ""#f"{dataset}: {metric_info['title']}"
    if compare_random:
        title += ""

    plt.title(title)
    plt.xlabel("q (Number of intervened top concepts)")
    plt.ylabel(metric_info["ylabel"])

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    if metric_info.get("higher_is_better", True):
        hint = "higher is better"
    else:
        hint = "lower is better"

    ax.text(
        0.98,
        0.02,
        #"",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.75,
    )

    plt.grid(True, alpha=0.3)
    #plt.legend(fontsize=8)
    plt.tight_layout()

    suffix = "_with_random" if compare_random else ""
    fname = f"{dataset}_{intervention}_{metric}_avg_backbones_std{suffix}.png"

    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_dir / fname}")


def plot_curves_per_backbone(
    df,
    dataset,
    model,
    intervention,
    metric,
    metric_info,
    out_dir,
    max_k=None,
):
    """
    One figure per dataset/model/metric.
    One curve per backbone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df[
        (df["dataset"] == dataset)
        & (df["model_label"] == model)
        & (df["intervention"] == intervention)
    ].copy()

    if max_k is not None:
        sub = sub[sub["k"] <= max_k]

    if sub.empty:
        return

    plt.figure(figsize=(7.5, 4.6))

    found_any = False

    for backbone in ordered_items(sub["backbone"].unique(), BACKBONE_ORDER):
        bsub = sub[sub["backbone"] == backbone].copy()

        y, _ = get_metric_values(bsub, metric)

        if y is None:
            continue

        bsub["_value"] = y
        bsub = bsub.dropna(subset=["_value"])

        if bsub.empty:
            continue

        # Average over config if there are duplicate rows
        stats = (
            bsub.groupby("k", as_index=False)["_value"]
            .mean()
            .sort_values("k")
        )

        found_any = True

        plt.plot(
            stats["k"],
            stats["_value"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=backbone,
        )

    if not found_any:
        plt.close()
        return

    plt.title(f"{dataset} - {model}: {metric_info['title']} by backbone")
    plt.xlabel("q (Number of intervened top concepts)")
    plt.ylabel(metric_info["ylabel"])

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    plt.grid(True, alpha=0.3)
    #plt.legend()
    plt.tight_layout()

    clean_model = clean_model_for_filename(model)
    fname = f"{dataset}_{clean_model}_{intervention}_{metric}_by_backbone.png"

    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_dir / fname}")


def plot_full_oracle_summary(df, out_dir):
    """
    Bar figures for oracle_all.
    One figure per dataset/metric.
    Bars = models.
    Error bars = std over backbones/config rows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = df[df["intervention"] == "oracle_all"].copy()

    if oracle.empty:
        print("[skip] no oracle_all rows")
        return

    for dataset in ordered_items(oracle["dataset"].unique(), DATASET_ORDER):
        dsub = oracle[oracle["dataset"] == dataset].copy()

        for metric, label in FULL_ORACLE_METRICS.items():
            y, _ = get_metric_values(dsub, metric)

            if y is None:
                continue

            temp = dsub.copy()
            temp["_value"] = y
            temp = temp.dropna(subset=["_value"])

            if temp.empty:
                continue

            stats = (
                temp.groupby("model_label", as_index=False)["_value"]
                .agg(["mean", "std"])
                .reset_index()
            )

            stats["std"] = stats["std"].fillna(0.0)
            stats["model_label"] = pd.Categorical(
                stats["model_label"],
                categories=MODEL_ORDER,
                ordered=True,
            )
            stats = stats.sort_values("model_label")

            plt.figure(figsize=(7.2, 4.5))

            x = np.arange(len(stats))

            plt.bar(
                x,
                stats["mean"],
                yerr=stats["std"],
                capsize=4,
            )

            plt.xticks(x, stats["model_label"], rotation=20, ha="right")
            plt.ylabel(label)
            plt.title(f"{dataset}: full oracle intervention - {label}")
            plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            fname = f"{dataset}_oracle_all_{metric}_bar_std_backbones.png"
            plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"[saved] {out_dir / fname}")


def make_topk_figures(df, out_dir, max_k=None):
    avg_dir = Path(out_dir) / "topk_avg_backbones_std"
    backbone_dir = Path(out_dir) / "topk_one_curve_per_backbone"

    for dataset in ordered_items(df["dataset"].dropna().unique(), DATASET_ORDER):
        for metric, info in TOPK_METRICS.items():
            plot_curves_avg_backbones(
                df=df,
                dataset=dataset,
                intervention="partial_topk_oracle",
                metric=metric,
                metric_info=info,
                out_dir=avg_dir,
                max_k=max_k,
                compare_random=False,
            )

            # Useful comparison for accuracy changes
            if metric in ["accuracy_change", "balanced_accuracy_change"]:
                plot_curves_avg_backbones(
                    df=df,
                    dataset=dataset,
                    intervention="partial_topk_oracle",
                    metric=metric,
                    metric_info=info,
                    out_dir=avg_dir,
                    max_k=max_k,
                    compare_random=True,
                )

        for model in ordered_items(df["model_label"].dropna().unique(), MODEL_ORDER):
            for metric, info in TOPK_METRICS.items():
                plot_curves_per_backbone(
                    df=df,
                    dataset=dataset,
                    model=model,
                    intervention="partial_topk_oracle",
                    metric=metric,
                    metric_info=info,
                    out_dir=backbone_dir,
                    max_k=max_k,
                )


def make_targeted_neutralization_figures(df, out_dir, max_k=None):
    avg_dir = Path(out_dir) / "targeted_negative_avg_backbones_std"
    backbone_dir = Path(out_dir) / "targeted_negative_one_curve_per_backbone"

    interventions = [
        x for x in df["intervention"].dropna().unique()
        if str(x).startswith("targeted_negative")
    ]

    for intervention in interventions:
        for dataset in ordered_items(df["dataset"].dropna().unique(), DATASET_ORDER):
            for metric, info in NEGATIVE_METRICS.items():
                plot_curves_avg_backbones(
                    df=df,
                    dataset=dataset,
                    intervention=intervention,
                    metric=metric,
                    metric_info=info,
                    out_dir=avg_dir,
                    max_k=max_k,
                    compare_random=False,
                )

            for model in ordered_items(df["model_label"].dropna().unique(), MODEL_ORDER):
                for metric, info in NEGATIVE_METRICS.items():
                    plot_curves_per_backbone(
                        df=df,
                        dataset=dataset,
                        model=model,
                        intervention=intervention,
                        metric=metric,
                        metric_info=info,
                        out_dir=backbone_dir,
                        max_k=max_k,
                    )


def plot_curves_one_backbone_all_models_with_random(
    df,
    dataset,
    backbone,
    metric,
    metric_info,
    out_dir,
    max_k=None,
):
    """
    One figure per dataset/backbone/metric.
    Solid lines = targeted top-k oracle intervention.
    Dashed lines = random oracle intervention.
    One color per model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df[
        (df["dataset"] == dataset)
        & (df["backbone"] == backbone)
        & (df["intervention"].isin(["partial_topk_oracle", "random_oracle"]))
    ].copy()

    if max_k is not None:
        sub = sub[sub["k"] <= max_k]

    if sub.empty:
        print(f"[skip] no data for {dataset}, {backbone}, {metric}")
        return

    plt.figure(figsize=(7.8, 4.8))
    found_any = False

    for model in ordered_items(sub["model_label"].dropna().unique(), MODEL_ORDER):
        color = MODEL_COLORS.get(model, None)

        for intervention, linestyle, suffix in [
            ("partial_topk_oracle", "-", "top-k"),
            ("random_oracle", "--", "random"),
        ]:
            msub = sub[
                (sub["model_label"] == model)
                & (sub["intervention"] == intervention)
            ].copy()

            if msub.empty:
                continue

            y, _ = get_metric_values(msub, metric)
            if y is None:
                continue

            msub["_value"] = y
            msub = msub.dropna(subset=["_value"])

            if msub.empty:
                continue

            # Average over config / folds / random repeats if duplicates exist
            stats = (
                msub.groupby("k", as_index=False)["_value"]
                .mean()
                .sort_values("k")
            )

            if stats.empty:
                continue

            found_any = True

            plt.plot(
                stats["k"].values,
                stats["_value"].values,
                linestyle=linestyle,
                marker="o" if intervention == "partial_topk_oracle" else None,
                markersize=3,
                linewidth=2.0,
                color=color,
                alpha=1.0 if intervention == "partial_topk_oracle" else 0.65,
                label=f"{model} {suffix}",
            )

    if not found_any:
        plt.close()
        print(f"[skip] no valid metric values for {dataset}, {backbone}, {metric}")
        return

    #plt.title(f"{dataset} - {backbone}: {metric_info['title']} vs random")
    plt.xlabel("Number of intervened concepts ($k$)")
    plt.ylabel(metric_info["ylabel"])

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    hint = "higher is better" if metric_info.get("higher_is_better", True) else "lower is better"
    ax.text(
        0.98,
        0.02,
        "",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.75,
    )

    plt.grid(True, alpha=0.3)
    #plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    fname = f"{dataset}_{backbone}_{metric}_topk_vs_random.png"
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_dir / fname}")


def make_topk_per_backbone_all_models_with_random(df, out_dir, max_k=None):
    out_dir = Path(out_dir) / "topk_per_backbone_all_models_with_random"

    for dataset in ordered_items(df["dataset"].dropna().unique(), DATASET_ORDER):
        dsub = df[df["dataset"] == dataset]

        for backbone in ordered_items(dsub["backbone"].dropna().unique(), BACKBONE_ORDER):
            for metric, info in TOPK_METRICS.items():
                plot_curves_one_backbone_all_models_with_random(
                    df=df,
                    dataset=dataset,
                    backbone=backbone,
                    metric=metric,
                    metric_info=info,
                    out_dir=out_dir,
                    max_k=max_k,
                )


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

    style_handles = [
        # Intervention styles
        Line2D([0], [0], color="black", lw=2, linestyle="-",
               marker="o", markersize=4,
               label="Targeted"),

        Line2D([0], [0], color="black", lw=2, linestyle="--",
               label="Random"),
    ]

    fig, ax = plt.subplots(figsize=(11.6, 1.4))
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

    ax.legend(
        handles=style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.5,
    )


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
        default=[
            "intervenability_metrics_native/intervenability_aggregated.csv",
            "intervenability_metrics/2/more_k/intervenability_aggregated.csv",
            "intervenability_metrics/more_k/intervenability_aggregated.csv",
        ],
        help="One or more intervenability_aggregated.csv files.",
    )

    parser.add_argument(
        "--out_dir",
        default="intervenability_metrics_native/intervenability_plots",
        help="Output directory.",
    )

    parser.add_argument(
        "--max_k",
        type=int,
        default=112,
        help="Maximum k to plot. If omitted, uses all k values in the CSV.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_intervention_csvs(args.csv)

    pco = df[
        (df["dataset"] == "AwA2")
        & (df["backbone"].str.lower() == "efficientnetb0")
        & (df["model_label"] == "PACBM-Co")
    ]
    print(
        pco[pco["intervention"].isin(["oracle_all", "random_oracle"])]
        .query("intervention == 'oracle_all' or k == 85")
        [["config", "intervention", "k",
        "correction_rate_mean", "degradation_rate_mean"]]
        .to_string(index=False)
    )
    cols = [
        "config",
        "original_balanced_accuracy_mean",
        "original_balanced_accuracy_std",
        "intervened_balanced_accuracy_mean",
        "intervened_balanced_accuracy_std",
        "balanced_accuracy_change_mean",
        "balanced_accuracy_change_std",
        "correction_rate_mean",
        "correction_rate_std",
        "degradation_rate_mean",
        "degradation_rate_std",
    ]
    print(
        pco[pco["intervention"] == "oracle_all"][cols]
        .to_string(index=False)
    )

    # Save normalized version for checking
    df.to_csv(out_dir / "intervenability_plot_input_normalized.csv", index=False)

    #make_topk_figures(df, out_dir, max_k=args.max_k)
    #make_targeted_neutralization_figures(df, out_dir, max_k=args.max_k)
    #plot_full_oracle_summary(df, out_dir / "oracle_all_bars_std_backbones")
    make_topk_per_backbone_all_models_with_random(df, out_dir, max_k=args.max_k)
    #make_pacbm_present_vs_absent_figures(df, out_dir, max_k=args.max_k)

    print("\nDone.")
    print(f"Saved plots under: {out_dir}")
    print(f"- {out_dir / 'topk_avg_backbones_std'}")
    print(f"- {out_dir / 'topk_one_curve_per_backbone'}")
    print(f"- {out_dir / 'targeted_negative_avg_backbones_std'}")
    print(f"- {out_dir / 'targeted_negative_one_curve_per_backbone'}")
    print(f"- {out_dir / 'oracle_all_bars_std_backbones'}")
    print(f"- {out_dir / 'topk_per_backbone_all_models_with_random'}")
    print(f"- {out_dir / 'pacbm_present_vs_absent'}")

    save_shared_legend(out_dir)


if __name__ == "__main__":
    main()