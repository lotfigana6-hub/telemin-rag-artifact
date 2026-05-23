from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from telemin_rag.evaluation import run_experiment, summarize_by_dataset_method, summarize_results, write_markdown_tables
from telemin_rag.plotting import generate_plots


def parse_noise_levels(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TeleMin-RAG BGL robustness experiments.")
    parser.add_argument("--datasets", default="BGL", help="Comma-separated datasets: BGL,UNSW_NB15,Apache,HPC")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--noise-levels", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument(
        "--poison-strategy",
        default="instruction",
        choices=["instruction", "obfuscated", "semantic", "none"],
        help="Synthetic contextual poisoning strategy.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--no-shap", action="store_true", help="Skip the SHAP-based baseline.")
    parser.add_argument("--no-auto-tune", action="store_true", help="Disable validation-only TeleMin tuning.")
    parser.add_argument("--selector-suite", default="main", choices=["main", "ablation"])
    parser.add_argument("--downstream-model", default="logistic", choices=["logistic", "random_forest"])
    parser.add_argument("--include-llm", action="store_true", help="Add a zero-shot LLM full-context baseline.")
    parser.add_argument("--llm-provider", default="ollama", choices=["ollama", "qwen-cloud"])
    parser.add_argument("--llm-model", default="qwen2.5:0.5b")
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument(
        "--llm-baselines",
        default="zero-shot-full",
        help="Comma-separated LLM baselines: zero-shot-full,few-shot-full,few-shot-bm25,few-shot-telemin or all.",
    )
    args = parser.parse_args()

    if args.llm_provider == "qwen-cloud" and args.llm_model == "qwen2.5:0.5b":
        args.llm_model = "qwen-turbo"
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    all_metrics = []
    for dataset in datasets:
        dataset_results_dir = Path(args.results_dir) / dataset
        metrics, _ = run_experiment(
            dataset=dataset,
            data_dir=Path(args.data_dir),
            results_dir=dataset_results_dir,
            figures_dir=Path(args.figures_dir) / dataset,
            context_size=args.context_size,
            stride=args.stride,
            k=args.k,
            noise_levels=parse_noise_levels(args.noise_levels),
            poison_rate=args.poison_rate,
            poison_strategy=args.poison_strategy,
            seed=args.seed,
            test_size=args.test_size,
            include_shap=not args.no_shap,
            auto_tune=not args.no_auto_tune,
            max_rows=args.max_rows,
            max_contexts=args.max_contexts,
            include_llm=args.include_llm,
            llm_model=args.llm_model,
            llm_provider=args.llm_provider,
            llm_batch_size=args.llm_batch_size,
            selector_suite=args.selector_suite,
            llm_baselines=parse_list(args.llm_baselines),
            downstream_model=args.downstream_model,
        )
        all_metrics.append(metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize_results(metrics)
    dataset_summary = summarize_by_dataset_method(metrics)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "metrics_by_noise.csv", index=False)
    summary.to_csv(results_dir / "summary_by_method.csv", index=False)
    dataset_summary.to_csv(results_dir / "summary_by_dataset_method.csv", index=False)
    write_markdown_tables(metrics, summary, results_dir)
    generate_plots(metrics, summary, figures_dir=args.figures_dir)
    print(summary.to_string(index=False))
    if len(datasets) > 1:
        print("\nPer-dataset summary:")
        print(dataset_summary.to_string(index=False))
    print(f"\nWrote metrics to {Path(args.results_dir) / 'metrics_by_noise.csv'}")
    print(f"Wrote plots to {Path(args.figures_dir).resolve()}")


if __name__ == "__main__":
    main()
