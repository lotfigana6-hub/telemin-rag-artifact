from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from telemin_rag.evaluation import run_experiment, summarize_by_dataset_method, summarize_results, write_markdown_tables
from telemin_rag.plotting import generate_plots
from telemin_rag.statistics import write_dataset_latex_tables, write_latex_tables


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed TeleMin-RAG experiments.")
    parser.add_argument("--datasets", default="BGL,UNSW_NB15")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--poison-strategies", default="instruction")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--results-dir", default="results/multiseed")
    parser.add_argument("--figures-dir", default="figures/multiseed")
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--noise-levels", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--include-shap", action="store_true")
    parser.add_argument("--no-auto-tune", action="store_true")
    parser.add_argument("--selector-suite", default="main", choices=["main", "ablation"])
    parser.add_argument("--downstream-model", default="logistic", choices=["logistic", "random_forest"])
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--llm-provider", default="ollama", choices=["ollama", "qwen-cloud"])
    parser.add_argument("--llm-model", default="gpt-oss:20b-cloud")
    parser.add_argument("--llm-baselines", default="zero-shot-full")
    parser.add_argument("--llm-batch-size", type=int, default=8)
    args = parser.parse_args()

    all_metrics: list[pd.DataFrame] = []
    results_root = Path(args.results_dir)
    figures_root = Path(args.figures_dir)
    for strategy in parse_list(args.poison_strategies):
        for seed in parse_ints(args.seeds):
            for dataset in parse_list(args.datasets):
                run_dir = results_root / f"{strategy}" / f"seed_{seed}" / dataset
                fig_dir = figures_root / f"{strategy}" / f"seed_{seed}" / dataset
                metrics, _ = run_experiment(
                    dataset=dataset,
                    data_dir=Path(args.data_dir),
                    results_dir=run_dir,
                    figures_dir=fig_dir,
                    context_size=args.context_size,
                    stride=args.stride,
                    k=args.k,
                    noise_levels=parse_floats(args.noise_levels),
                    poison_rate=args.poison_rate,
                    poison_strategy=strategy,
                    seed=seed,
                    test_size=args.test_size,
                    include_shap=args.include_shap,
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

    combined = pd.concat(all_metrics, ignore_index=True)
    results_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(results_root / "metrics_by_noise.csv", index=False)
    summary = summarize_results(combined)
    dataset_summary = summarize_by_dataset_method(combined)
    summary.to_csv(results_root / "summary_by_method.csv", index=False)
    dataset_summary.to_csv(results_root / "summary_by_dataset_method.csv", index=False)
    write_markdown_tables(combined, summary, results_root)
    reference = "Full TeleMin-RAG" if args.selector_suite == "ablation" else "TeleMin-RAG"
    write_latex_tables(combined, results_root, reference=reference)
    write_dataset_latex_tables(combined, results_root, reference=reference)
    generate_plots(combined, summary, figures_dir=figures_root)
    print(summary.to_string(index=False))
    print(f"\nWrote multi-seed metrics to {results_root / 'metrics_by_noise.csv'}")


if __name__ == "__main__":
    main()
