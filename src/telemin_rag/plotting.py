from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_metric_vs_noise(metrics_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plot_df = metrics_df
    if "seed" in metrics_df.columns and metrics_df["seed"].nunique() > 1:
        plot_df = metrics_df.sort_values(["method", "noise_level"])
    elif "dataset" in metrics_df.columns and metrics_df["dataset"].nunique() > 1:
        plot_df = (
            metrics_df.groupby(["method", "noise_level"], as_index=False)[metric]
            .mean()
            .sort_values(["method", "noise_level"])
        )
    plt.figure(figsize=(8.5, 4.8))
    sns.lineplot(
        data=plot_df,
        x="noise_level",
        y=metric,
        hue="method",
        marker="o",
        linewidth=2,
        errorbar=("ci", 95) if "seed" in plot_df.columns and plot_df["seed"].nunique() > 1 else None,
    )
    plt.xlabel("Random noise level")
    plt.ylabel(metric.upper() if metric == "f1" else metric.title())
    plt.ylim(0, 1.02)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_context_reduction_vs_performance(metrics_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = metrics_df.copy()
    if "seed" in plot_df.columns and plot_df["seed"].nunique() > 1:
        plot_df = (
            plot_df.groupby(["method", "noise_level"], as_index=False)
            .agg(f1=("f1", "mean"), context_reduction=("context_reduction", "mean"))
            .sort_values(["method", "noise_level"])
        )
    plt.figure(figsize=(8.2, 5.0))
    sns.scatterplot(
        data=plot_df,
        x="context_reduction",
        y="f1",
        hue="method",
        size="noise_level",
        sizes=(35, 150),
        alpha=0.85,
    )
    plt.xlabel("Context reduction")
    plt.ylabel("F1-score")
    plt.xlim(-0.02, 1.0)
    plt.ylim(0, 1.02)
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_poison_selection(metrics_df: pd.DataFrame, out_path: Path) -> None:
    if "poison_event_selection_rate" not in metrics_df.columns:
        return
    plt.figure(figsize=(8.6, 4.8))
    sns.lineplot(
        data=metrics_df.sort_values(["method", "noise_level"]),
        x="noise_level",
        y="poison_event_selection_rate",
        hue="method",
        marker="o",
        linewidth=2,
        errorbar=("ci", 95) if "seed" in metrics_df.columns and metrics_df["seed"].nunique() > 1 else None,
    )
    plt.xlabel("Random noise level")
    plt.ylabel("Selected poison events / injected poison events")
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_security_utility_pareto(summary_df: pd.DataFrame, out_path: Path) -> None:
    required = {"mean_f1", "mean_context_reduction", "mean_poison_event_selection_rate"}
    if not required.issubset(summary_df.columns):
        return
    plt.figure(figsize=(8.2, 5.0))
    sns.scatterplot(
        data=summary_df,
        x="mean_poison_event_selection_rate",
        y="mean_f1",
        size="mean_context_reduction",
        hue="method",
        sizes=(60, 260),
        alpha=0.88,
    )
    plt.xlabel("Poison event selection rate")
    plt.ylabel("Mean F1-score")
    plt.xlim(-0.02, min(1.02, max(0.05, summary_df["mean_poison_event_selection_rate"].max() * 1.20)))
    plt.ylim(0, 1.02)
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_ablation(summary_df: pd.DataFrame, out_path: Path) -> None:
    if "Full TeleMin-RAG" not in set(summary_df["method"]):
        return
    ordered = summary_df.sort_values("mean_f1", ascending=False)
    plt.figure(figsize=(8.4, 4.9))
    sns.barplot(data=ordered, x="mean_f1", y="method", color="#0072B2")
    plt.xlabel("Mean F1-score")
    plt.ylabel("")
    plt.xlim(0, min(1.02, max(0.1, ordered["mean_f1"].max() * 1.10)))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_robustness(summary_df: pd.DataFrame, out_path: Path) -> None:
    ordered = summary_df.sort_values("robustness_retention", ascending=False)
    plt.figure(figsize=(8.5, 4.8))
    sns.barplot(data=ordered, x="robustness_retention", y="method", color="#3b82f6")
    plt.xlabel("F1 retention at 50% noise relative to 0% noise")
    plt.ylabel("")
    plt.xlim(0, max(1.05, ordered["robustness_retention"].max() * 1.08))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_plots(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: str | Path = "figures") -> None:
    set_style()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_noise(metrics_df, "f1", figures_dir / "f1_vs_noise.png")
    plot_metric_vs_noise(metrics_df, "auroc", figures_dir / "auroc_vs_noise.png")
    if "auprc" in metrics_df.columns:
        plot_metric_vs_noise(metrics_df, "auprc", figures_dir / "auprc_vs_noise.png")
    plot_context_reduction_vs_performance(metrics_df, figures_dir / "context_reduction_vs_performance.png")
    plot_poison_selection(metrics_df, figures_dir / "poison_selection_vs_noise.png")
    plot_security_utility_pareto(summary_df, figures_dir / "security_utility_pareto.png")
    plot_ablation(summary_df, figures_dir / "ablation_bar_chart.png")
    plot_robustness(summary_df, figures_dir / "robustness_comparison.png")
