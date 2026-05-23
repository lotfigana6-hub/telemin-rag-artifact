from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = sns.color_palette("colorblind")


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def short_method(name: str) -> str:
    mapping = {
        "Mutual Information Feature Selection": "Mutual Info",
        "Embedding Similarity Selection": "Embedding",
        "TF-IDF/BM25 Selection": "BM25",
        "Random-k Selection": "Random-k",
        "Ollama LLM zero-shot Full Context (gpt-oss:20b-cloud)": "LLM zero-shot\nFull",
        "Ollama LLM few-shot Full Context (gpt-oss:20b-cloud)": "LLM few-shot\nFull",
        "Ollama LLM few-shot BM25-selected context (gpt-oss:20b-cloud)": "LLM few-shot\nBM25",
        "Ollama LLM few-shot TeleMin-selected context (gpt-oss:20b-cloud)": "LLM few-shot\nTeleMin",
    }
    return mapping.get(name, name)


def load(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method_short"] = frame["method"].map(short_method)
    return frame


def save_line_metric(frame: pd.DataFrame, metric: str, out: Path, title: str) -> None:
    plt.figure(figsize=(9.5, 5.4))
    sns.lineplot(
        data=frame,
        x="noise_level",
        y=metric,
        hue="method_short",
        marker="o",
        linewidth=2.4,
        errorbar=("ci", 95),
    )
    plt.title(title)
    plt.xlabel("Random noise level")
    plt.ylabel({"f1": "F1-score", "auroc": "AUROC", "auprc": "AUPRC"}.get(metric, metric))
    plt.ylim(0, 1.03)
    plt.legend(title="", loc="lower left", ncols=2, frameon=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def save_pareto(frame: pd.DataFrame, out: Path) -> None:
    summary = (
        frame.groupby(["dataset", "method", "method_short"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            mean_poison_event_selection_rate=("poison_event_selection_rate", "mean"),
            mean_context_reduction=("context_reduction", "mean"),
        )
        .sort_values(["dataset", "mean_f1"], ascending=[True, False])
    )
    methods = list(summary["method_short"].drop_duplicates())
    markers = {
        "TeleMin-RAG": "D",
        "Full Context": "X",
        "Embedding": "s",
        "BM25": "o",
        "Mutual Info": "*",
        "Random-k": "P",
    }
    colors = dict(zip(methods, sns.color_palette("colorblind", n_colors=len(methods))))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (dataset, group) in zip(axes, summary.groupby("dataset")):
        for _, row in group.iterrows():
            method = row["method_short"]
            size = 80 + 260 * float(row["mean_context_reduction"])
            ax.scatter(
                row["mean_poison_event_selection_rate"],
                row["mean_f1"],
                s=size,
                marker=markers.get(method, "o"),
                color=colors[method],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.5,
                label=method,
            )
        labelled = pd.concat(
            [group[group["method_short"].eq("TeleMin-RAG")], group.head(1)]
        ).drop_duplicates(subset=["method_short"])
        for _, row in labelled.iterrows():
            ax.annotate(
                row["method_short"],
                xy=(row["mean_poison_event_selection_rate"], row["mean_f1"]),
                xytext=(7, 6),
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.3"},
            )
        ax.set_title(dataset)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.35)
    for ax in axes[2:]:
        ax.set_xlabel("Poison event selection rate")
    for ax in axes[::2]:
        ax.set_ylabel("Mean F1-score")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        title="Method",
        loc="center right",
        bbox_to_anchor=(1.14, 0.5),
        frameon=True,
        fontsize=9,
    )
    fig.text(0.5, 0.015, "Marker size encodes context reduction.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 0.88, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_context_reduction_by_dataset(frame: pd.DataFrame, out: Path) -> None:
    plot = (
        frame.groupby(["dataset", "method", "method_short"], as_index=False)
        .agg(f1=("f1", "mean"), context_reduction=("context_reduction", "mean"))
        .sort_values(["dataset", "f1"], ascending=[True, False])
    )
    grid = sns.relplot(
        data=plot,
        x="context_reduction",
        y="f1",
        hue="method_short",
        col="dataset",
        col_wrap=2,
        kind="scatter",
        s=120,
        height=4.0,
        aspect=1.25,
        edgecolor="black",
        linewidth=0.4,
    )
    grid.set_axis_labels("Context reduction", "Mean F1-score")
    grid.set(xlim=(-0.04, 1.0), ylim=(0, 1.02))
    grid.set_titles("{col_name}")
    for dataset, ax in grid.axes_dict.items():
        group = plot[plot["dataset"].eq(dataset)].copy()
        labelled = pd.concat(
            [
                group[group["method_short"].eq("TeleMin-RAG")],
                group[group["method_short"].eq("Full Context")],
                group.head(1),
            ]
        ).drop_duplicates(subset=["dataset", "method_short"])
        for _, row in labelled.iterrows():
            ax.text(
                row["context_reduction"] + 0.012,
                row["f1"] + 0.012,
                row["method_short"],
                fontsize=8,
            )
    grid.legend.set_title("")
    sns.move_legend(
        grid,
        "center right",
        bbox_to_anchor=(1.08, 0.5),
        frameon=True,
        title="Method",
    )
    grid.fig.set_size_inches(11.5, 8.2)
    grid.fig.tight_layout(rect=(0, 0, 0.86, 1))
    grid.fig.savefig(out, bbox_inches="tight")
    plt.close(grid.fig)


def save_bar(summary: pd.DataFrame, out: Path, title: str, metric: str = "mean_f1") -> None:
    plot = summary.copy().sort_values(metric, ascending=False)
    plot["method_short"] = plot["method"].map(short_method)
    plt.figure(figsize=(9.6, max(4.8, 0.45 * len(plot))))
    sns.barplot(data=plot, y="method_short", x=metric, color=PALETTE[0])
    plt.title(title)
    plt.xlabel(metric.replace("mean_", "").replace("_", " ").title())
    plt.ylabel("")
    plt.xlim(0, min(1.03, max(0.2, plot[metric].max() * 1.12)))
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def save_llm_table_plot(metrics: pd.DataFrame, out: Path) -> None:
    summary = (
        metrics.groupby("method", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            mean_psr=("poison_event_selection_rate", "mean"),
            mean_latency=("latency_per_context_ms", "mean"),
            mean_context_reduction=("context_reduction", "mean"),
        )
        .sort_values("mean_f1", ascending=False)
    )
    summary["method_short"] = summary["method"].map(short_method)
    plt.figure(figsize=(10, 5.6))
    sns.barplot(data=summary, x="mean_f1", y="method_short", hue="mean_psr", palette="viridis")
    plt.xlabel("Mean F1-score")
    plt.ylabel("")
    plt.title("Bounded multi-dataset LLM study: F1 colored by poison selection")
    plt.xlim(0, 1.02)
    plt.legend(title="Poison sel.", loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    style()
    out = Path("figures/paper_final")
    out.mkdir(parents=True, exist_ok=True)

    native = load("results/multiseed_native_fixed/metrics_by_noise.csv")
    all_metrics = load("results/multiseed_all_fixed/metrics_by_noise.csv")
    llm = load("results/llm_multidataset/metrics_by_noise.csv")
    llm_summary = pd.read_csv("results/llm_multidataset/summary_by_method.csv")

    save_line_metric(native, "f1", out / "native_f1_vs_noise.png", "Native-label datasets")
    save_line_metric(native, "auprc", out / "native_auprc_vs_noise.png", "Native-label datasets")
    save_pareto(all_metrics, out / "security_utility_pareto.png")
    save_context_reduction_by_dataset(all_metrics, out / "context_reduction_by_dataset.png")
    save_bar(pd.read_csv("results/ablation_native_fixed/summary_by_method.csv"), out / "ablation_f1.png", "TeleMin-RAG ablation")
    save_llm_table_plot(llm, out / "llm_multidataset_f1.png")


if __name__ == "__main__":
    main()
