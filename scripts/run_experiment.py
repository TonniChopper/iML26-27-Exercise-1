"""Run all experiments and write the results to ``results/``.

Usage::

    uv run python scripts/run_experiment.py

Outputs:

* ``results/scores.csv``              – accuracy and all metrics for every (dataset, model)
* ``results/threshold.csv``           – most complex model per dataset and family in each zone
* ``results/lambda_sensitivity.csv``  – does the ranking of models change for lambda = 1, 2, 3?
* ``results/metric_correlations.csv`` – how strongly our metric agrees with simple baselines
* ``results/*.png``                   – plots used in the presentation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from iml_interpretability import (
    GLOBAL_GREY_ZONE,
    GLOBAL_INTERPRETABLE,
    LOCAL_INTERPRETABLE,
    default_metrics,
    evaluate,
    get_datasets,
    get_models,
    interpretability_zone,
    model_family,
)

RESULTS = Path("results")
SEED = 0
FAMILY_LABELS = {"dt": "decision tree", "rf": "random forest", "gb": "gradient boosting"}
FAMILY_COLORS = {"dt": "#2a6fdb", "rf": "#1b9e77", "gb": "#d95f02"}


def run_scores() -> pd.DataFrame:
    models = get_models(seed=SEED)
    scores = evaluate(datasets=get_datasets(), models=models, metrics=default_metrics(), seed=SEED)
    scores.insert(2, "family", scores["model"].map(model_family))
    scores["zone"] = scores["global_load"].map(interpretability_zone)
    scores["prediction_easy"] = scores["local_load"] <= LOCAL_INTERPRETABLE
    # Keep the order of get_models() as complexity order inside each family.
    scores["complexity_rank"] = scores["model"].map({name: i for i, name in enumerate(models)})
    return scores


def _last_model(group: pd.DataFrame, mask: pd.Series) -> str:
    """Name of the most complex model in ``group`` (sorted by complexity) for which ``mask`` holds."""
    chosen = group[mask]
    return chosen.iloc[-1]["model"] if len(chosen) else "none"


def threshold_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and family: the most complex model that still reaches each verdict."""
    rows = []
    for (dataset, family), group in scores.groupby(["dataset", "family"], sort=False):
        group = group.sort_values("complexity_rank")
        interpretable = group["global_load"] <= GLOBAL_INTERPRETABLE
        best = group[interpretable].iloc[-1] if interpretable.any() else None
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "last_interpretable": _last_model(group, interpretable),
                "last_grey_zone": _last_model(group, group["global_load"] <= GLOBAL_GREY_ZONE),
                "last_easy_prediction": _last_model(group, group["local_load"] <= LOCAL_INTERPRETABLE),
                "accuracy_last_interpretable": best["accuracy"] if best is not None else np.nan,
                "best_accuracy_any_model": scores.loc[scores["dataset"] == dataset, "accuracy"].max(),
            }
        )
    return pd.DataFrame(rows)


def lambda_sensitivity(scores: pd.DataFrame, lambdas=(1.0, 2.0, 3.0)) -> pd.DataFrame:
    """Spearman correlation between model rankings for different lambdas (computed from the parts)."""
    rows = []
    for dataset, group in scores.groupby("dataset", sort=False):
        loads = {lam: group["cases"] * (group["conditions"] + lam * group["exceptions"]) for lam in lambdas}
        row = {"dataset": dataset}
        for a in lambdas:
            for b in lambdas:
                if a < b:
                    row[f"spearman_lam{a:g}_vs_lam{b:g}"] = spearmanr(loads[a], loads[b]).statistic
        row["interpretable_models_per_lambda"] = ", ".join(
            f"lam{lam:g}: {int((loads[lam] <= GLOBAL_INTERPRETABLE).sum())}" for lam in lambdas
        )
        rows.append(row)
    return pd.DataFrame(rows)


def metric_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of global load with the baselines, over all models of a dataset."""
    rows = []
    for dataset, group in scores.groupby("dataset", sort=False):
        row = {"dataset": dataset}
        for baseline in ["n_leaves", "avg_depth", "simulation_effort"]:
            row[f"global_load_vs_{baseline}"] = spearmanr(group["global_load"], group[baseline]).statistic
        row["simulation_effort_vs_avg_depth"] = spearmanr(group["simulation_effort"], group["avg_depth"]).statistic
        rows.append(row)
    return pd.DataFrame(rows)


def plot_load_per_model(scores: pd.DataFrame) -> None:
    datasets = list(scores["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 2.6 * len(datasets)), sharex=True)
    for ax, dataset in zip(np.atleast_1d(axes), datasets, strict=True):
        group = scores[scores["dataset"] == dataset].sort_values("complexity_rank")
        colors = group["family"].map(FAMILY_COLORS)
        ax.bar(group["model"], group["global_load"], color=colors)
        ax.axhline(GLOBAL_INTERPRETABLE, color="black", linestyle="--", linewidth=1)
        ax.axhline(GLOBAL_GREY_ZONE, color="black", linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylabel("global load")
        ax.set_title(dataset, loc="left", fontsize=10)
    np.atleast_1d(axes)[-1].tick_params(axis="x", rotation=60)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    fig.legend(handles, FAMILY_LABELS.values(), loc="upper right", ncol=3, frameon=False)
    title = (
        f"Global load per model (log scale; dashed = {GLOBAL_INTERPRETABLE:g}, dotted = {GLOBAL_GREY_ZONE:g}, "
        "in between = grey zone)"
    )
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESULTS / "global_load_per_model.png", dpi=150)
    plt.close(fig)


def plot_accuracy_vs_load(scores: pd.DataFrame) -> None:
    datasets = list(scores["dataset"].unique())
    cols = 3
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        group = scores[scores["dataset"] == dataset]
        for family, sub in group.groupby("family"):
            sub = sub.sort_values("complexity_rank")
            ax.plot(sub["global_load"], sub["accuracy"], "o-", color=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
        ax.axvspan(1, GLOBAL_INTERPRETABLE, color="0.85", zorder=0)
        ax.axvspan(GLOBAL_INTERPRETABLE, GLOBAL_GREY_ZONE, color="0.94", zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("global load (log)")
        ax.set_ylabel("test accuracy")
        ax.set_title(dataset, fontsize=10)
    for ax in list(axes.flat)[len(datasets) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Accuracy vs. global load (dark = interpretable, light = grey zone)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_global_load.png", dpi=150)
    plt.close(fig)


def plot_tree_components(scores: pd.DataFrame) -> None:
    """How the three parts grow with tree depth (single decision trees only)."""
    trees = scores[scores["family"] == "dt"].sort_values("complexity_rank")
    parts = ["cases", "conditions", "exceptions"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, part in zip(axes, parts, strict=True):
        for dataset, group in trees.groupby("dataset", sort=False):
            ax.plot(group["model"], group[part], "o-", label=dataset)
        ax.set_title(part)
        ax.tick_params(axis="x", rotation=60)
    axes[0].legend(fontsize=8)
    fig.suptitle("The three parts of the load for single decision trees of growing depth", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "tree_components.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    scores = run_scores()
    scores.drop(columns="complexity_rank").to_csv(RESULTS / "scores.csv", index=False)

    tables = {
        "threshold": threshold_table(scores),
        "lambda_sensitivity": lambda_sensitivity(scores),
        "metric_correlations": metric_correlations(scores),
    }
    for name, table in tables.items():
        table.to_csv(RESULTS / f"{name}.csv", index=False)

    plot_load_per_model(scores)
    plot_accuracy_vs_load(scores)
    plot_tree_components(scores)

    columns = ["dataset", "model", "accuracy", "cases", "conditions", "exceptions", "local_load", "global_load", "zone"]
    print(scores[columns].round(3).to_string(index=False))
    for name, table in tables.items():
        print(f"\n=== {name}")
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
