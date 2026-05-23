from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from telemin_rag.evaluation import summarize_by_dataset_method, summarize_results, write_markdown_tables
from telemin_rag.plotting import generate_plots
from telemin_rag.statistics import write_dataset_latex_tables, write_latex_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summaries, plots, LaTeX tables, and tests from metrics CSV.")
    parser.add_argument("--metrics", default="results/metrics_by_noise.csv")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--reference", default="TeleMin-RAG")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    results_dir = Path(args.results_dir) if args.results_dir else metrics_path.parent
    metrics = pd.read_csv(metrics_path)
    summary = summarize_results(metrics)
    dataset_summary = summarize_by_dataset_method(metrics)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "summary_by_method.csv", index=False)
    dataset_summary.to_csv(results_dir / "summary_by_dataset_method.csv", index=False)
    write_markdown_tables(metrics, summary, results_dir)
    write_latex_tables(metrics, results_dir, reference=args.reference)
    write_dataset_latex_tables(metrics, results_dir, reference=args.reference)
    generate_plots(metrics, summary, figures_dir=args.figures_dir)
    print(summary.to_string(index=False))
    print(f"\nWrote analysis tables to {results_dir.resolve()}")


if __name__ == "__main__":
    main()
