from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from .data import (
    AlertContext,
    DATASET_CONFIGS,
    canonical_dataset_name,
    context_labels,
    corrupt_contexts,
    flatten_events,
    load_contexts,
)
from .llm import OllamaBatchClassifier, QwenCloudBatchClassifier, ollama_model_available, qwen_cloud_available
from .selectors import BaseSelector, TeleMinRAGSelector, make_ablation_selectors, make_selectors


def _few_shot_examples(selector: BaseSelector, train_contexts: list[AlertContext], seed: int, per_class: int = 2) -> list[tuple[str, int]]:
    rng = np.random.default_rng(seed)
    selected_contexts: list[AlertContext] = []
    for label in (0, 1):
        pool = [context for context in train_contexts if context.label == label]
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        selected_contexts.extend(pool[int(i)] for i in idx)
    if not selected_contexts:
        return []
    texts, _, _ = selector.transform(selected_contexts)
    return list(zip(texts, [context.label for context in selected_contexts]))


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if not np.isfinite(y_score).all():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if not np.isfinite(y_score).all():
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0 or not np.isfinite(y_score).all():
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_score >= lo) & (y_score < hi if hi < 1.0 else y_score <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_score[mask].mean()))
    return ece


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auroc = safe_auroc(y_true, y_score)
    brier = float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))) if np.isfinite(y_score).all() else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_macro": float(macro_f1),
        "auroc": auroc,
        "auroc_oriented": float(max(auroc, 1.0 - auroc)) if np.isfinite(auroc) else float("nan"),
        "auprc": safe_auprc(y_true, y_score),
        "brier": brier,
        "ece": expected_calibration_error(y_true, y_score),
    }


class TriageClassifier:
    def __init__(self, seed: int = 13, max_features: int = 12000, model_type: str = "logistic"):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
        )
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        else:
            raise ValueError("model_type must be one of: logistic, random_forest")

    def fit(self, texts: list[str], labels: list[int]) -> "TriageClassifier":
        x = self.vectorizer.fit_transform(texts)
        self.model.fit(x, labels)
        return self

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        x = self.vectorizer.transform(texts)
        scores = self.model.predict_proba(x)[:, 1]
        pred = (scores >= 0.5).astype(int)
        return pred, scores


def aggregate_counts(counts: list[dict[str, int]]) -> dict[str, float]:
    frame = pd.DataFrame(counts)
    result: dict[str, float] = {}
    for column in frame.columns:
        result[f"avg_{column}"] = float(frame[column].mean())
        result[f"sum_{column}"] = float(frame[column].sum())
    selected = max(float(frame["selected_logs"].sum()), 1.0)
    total_poison = max(float(frame["total_poison"].sum()), 1.0)
    total_noise = max(float(frame["total_noise"].sum()), 1.0)
    selected_tokens = max(float(frame["selected_tokens"].sum()), 1.0)
    result["noise_selected_rate"] = float(frame["selected_noise"].sum() / selected)
    result["poison_selected_rate"] = float(frame["selected_poison"].sum() / selected)
    result["poison_event_selection_rate"] = float(frame["selected_poison"].sum() / total_poison)
    result["noise_event_selection_rate"] = float(frame["selected_noise"].sum() / total_noise)
    result["context_reduction"] = float(1.0 - frame["selected_logs"].sum() / max(frame["total_logs"].sum(), 1.0))
    result["token_reduction"] = float(1.0 - frame["selected_tokens"].sum() / max(frame["total_tokens"].sum(), 1.0))
    result["poison_token_selected_rate"] = float(frame["selected_poison_tokens"].sum() / selected_tokens)
    return result


def attack_success_rate(y_true: np.ndarray, y_pred: np.ndarray, counts: list[dict[str, int]]) -> float:
    poisoned = np.asarray([count.get("total_poison", 0) > 0 for count in counts], dtype=bool)
    if not poisoned.any():
        return float("nan")
    return float(np.mean(np.asarray(y_true)[poisoned] != np.asarray(y_pred)[poisoned]))


def fit_method(
    selector: BaseSelector,
    train_contexts: list[AlertContext],
    seed: int,
    downstream_model: str = "logistic",
) -> tuple[TriageClassifier, dict]:
    start = time.perf_counter()
    selector.fit(train_contexts)
    fit_selector_seconds = time.perf_counter() - start
    train_texts, _, train_counts = selector.transform(train_contexts)
    classifier = TriageClassifier(seed=seed, model_type=downstream_model).fit(train_texts, context_labels(train_contexts))
    info = {
        "selector_fit_seconds": fit_selector_seconds,
        "train_avg_selected_logs": float(np.mean([c["selected_logs"] for c in train_counts])),
        "downstream_model": downstream_model,
    }
    if isinstance(selector, TeleMinRAGSelector):
        info.update(
            {
                "telemin_stop_threshold": selector.stop_threshold,
                "telemin_min_k": selector.min_k,
                "telemin_max_k": selector.max_k,
                "telemin_weights": asdict(selector.weights),
                "telemin_tuning_log": selector.tuning_log_,
                "telemin_use_poison_penalty": selector.use_poison_penalty,
                "telemin_use_stability": selector.use_stability,
                "telemin_use_rarity": selector.use_rarity,
                "telemin_use_semantic": selector.use_semantic,
                "telemin_use_redundancy": selector.use_redundancy,
                "telemin_adaptive_k": selector.adaptive_k,
            }
        )
    if hasattr(selector, "used_shap"):
        info["used_shap"] = bool(getattr(selector, "used_shap"))
    return classifier, info


def run_experiment(
    *,
    dataset: str = "BGL",
    data_dir: str | Path = "data/raw",
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    context_size: int = 8,
    stride: int | None = None,
    k: int = 3,
    noise_levels: list[float] | None = None,
    poison_rate: float = 0.10,
    poison_strategy: str = "instruction",
    seed: int = 13,
    test_size: float = 0.30,
    include_shap: bool = True,
    auto_tune: bool = True,
    max_rows: int | None = 2000,
    max_contexts: int | None = None,
    include_llm: bool = False,
    llm_model: str = "qwen2.5:0.5b",
    llm_provider: str = "ollama",
    llm_batch_size: int = 8,
    selector_suite: str = "main",
    llm_baselines: list[str] | None = None,
    downstream_model: str = "logistic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = canonical_dataset_name(dataset)
    noise_levels = noise_levels or [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df, contexts = load_contexts(
        dataset,
        data_dir=data_dir,
        context_size=context_size,
        stride=stride or context_size,
        max_rows=max_rows,
        max_contexts=max_contexts,
        seed=seed,
    )
    if len(contexts) < 20:
        raise ValueError(f"Dataset {dataset} produced too few contexts: {len(contexts)}")
    labels = context_labels(contexts)
    if len(set(labels)) < 2:
        raise ValueError(f"Dataset {dataset} produced a single context class; cannot evaluate binary triage.")
    train_contexts, test_contexts = train_test_split(
        contexts,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    noise_pool = flatten_events(train_contexts, normal_only=True) or flatten_events(train_contexts)

    if selector_suite == "ablation":
        selectors = make_ablation_selectors(k=k, seed=seed, auto_tune=auto_tune)
    elif selector_suite == "main":
        selectors = make_selectors(k=k, seed=seed, include_shap=include_shap, auto_tune=auto_tune)
    else:
        raise ValueError("selector_suite must be either 'main' or 'ablation'")
    rows: list[dict] = []
    method_info: dict[str, dict] = {
        "_dataset": {
            "dataset": dataset,
            "label_source": DATASET_CONFIGS[dataset]["label_source"],
            "context_label_rule": DATASET_CONFIGS[dataset]["context_label_rule"],
            "n_raw_events": int(len(df)),
            "n_contexts": int(len(contexts)),
            "n_train_contexts": int(len(train_contexts)),
            "n_test_contexts": int(len(test_contexts)),
            "context_size": int(context_size),
            "stride": int(stride or context_size),
            "test_size": test_size,
            "poison_rate": poison_rate,
            "poison_strategy": poison_strategy,
            "seed": seed,
            "positive_context_rate": float(np.mean(labels)),
            "selector_suite": selector_suite,
            "downstream_model": downstream_model,
        }
    }

    for selector in selectors:
        classifier, info = fit_method(selector, train_contexts, seed=seed, downstream_model=downstream_model)
        method_info[selector.name] = info
        for noise_level in noise_levels:
            corrupted_test = corrupt_contexts(
                test_contexts,
                noise_level=noise_level,
                noise_pool=noise_pool,
                poison_rate=poison_rate,
                poison_strategy=poison_strategy,
                seed=seed + int(noise_level * 1000),
            )
            eval_start = time.perf_counter()
            test_texts, _, counts = selector.transform(corrupted_test)
            pred, scores = classifier.predict(test_texts)
            eval_seconds = time.perf_counter() - eval_start
            y_true = np.asarray(context_labels(corrupted_test))
            metrics = evaluate_predictions(y_true, pred, scores)
            metrics["attack_success_rate"] = attack_success_rate(y_true, pred, counts)
            count_metrics = aggregate_counts(counts)
            rows.append(
                {
                    "dataset": dataset,
                    "method": selector.name,
                    "noise_level": noise_level,
                    "poison_rate": poison_rate,
                    "poison_strategy": poison_strategy,
                    "seed": seed,
                    "downstream_model": downstream_model,
                    "eval_seconds": eval_seconds,
                    "latency_per_context_ms": float(1000.0 * eval_seconds / max(len(test_texts), 1)),
                    **metrics,
                    **count_metrics,
                    **{k: v for k, v in info.items() if isinstance(v, (int, float, bool, str))},
                }
            )

    if include_llm:
        provider = llm_provider.lower().strip()
        requested_llm_baselines = llm_baselines or ["zero-shot-full"]
        if "all" in requested_llm_baselines:
            requested_llm_baselines = [
                "zero-shot-full",
                "few-shot-full",
                "few-shot-bm25",
                "few-shot-telemin",
            ]
        if provider == "qwen-cloud":
            llm_available = qwen_cloud_available()
            llm_error = "Qwen Cloud API key not available; set DASHSCOPE_API_KEY or QWEN_API_KEY."
        else:
            llm_available = ollama_model_available(llm_model)
            llm_error = f"Ollama model not available; run `ollama pull {llm_model}`."
        selector_by_name = {selector.name: selector for selector in selectors}
        specs = {
            "zero-shot-full": ("zero-shot Full Context", "Full Context", False),
            "few-shot-full": ("few-shot Full Context", "Full Context", True),
            "few-shot-bm25": ("few-shot BM25-selected context", "TF-IDF/BM25 Selection", True),
            "few-shot-telemin": ("few-shot TeleMin-selected context", "TeleMin-RAG", True),
        }
        for baseline_key in requested_llm_baselines:
            if baseline_key not in specs:
                raise ValueError(f"Unknown LLM baseline {baseline_key!r}. Use {sorted(specs)} or 'all'.")
            label, selector_name, use_examples = specs[baseline_key]
            llm_name = f"{provider.title()} LLM {label} ({llm_model})"
            selector = selector_by_name.get(selector_name)
            method_info[llm_name] = {
                "llm_provider": provider,
                "llm_model": llm_model,
                "llm_available": llm_available,
                "llm_baseline": baseline_key,
                "context_selector": selector_name,
                "few_shot": use_examples,
            }
            if selector is None:
                method_info[llm_name]["llm_available"] = False
                method_info[llm_name]["error"] = f"Selector {selector_name!r} not present in selector suite."
                continue
            if not llm_available:
                method_info[llm_name]["error"] = llm_error
                continue
            examples = _few_shot_examples(selector, train_contexts, seed=seed) if use_examples else []
            method_info[llm_name]["few_shot_examples"] = len(examples)
            cache_suffix = baseline_key.replace("-", "_")
            try:
                if provider == "qwen-cloud":
                    llm = QwenCloudBatchClassifier(
                        model=llm_model,
                        cache_path=results_dir / f"qwen_cloud_cache_{cache_suffix}.json",
                        batch_size=llm_batch_size,
                        examples=examples,
                    )
                else:
                    llm = OllamaBatchClassifier(
                        model=llm_model,
                        cache_path=results_dir / f"llm_cache_{cache_suffix}.json",
                        batch_size=llm_batch_size,
                        examples=examples,
                    )
                for noise_level in noise_levels:
                    corrupted_test = corrupt_contexts(
                        test_contexts,
                        noise_level=noise_level,
                        noise_pool=noise_pool,
                        poison_rate=poison_rate,
                        poison_strategy=poison_strategy,
                        seed=seed + int(noise_level * 1000),
                    )
                    test_texts, _, counts = selector.transform(corrupted_test)
                    pred, scores, raw_responses, eval_seconds = llm.classify(test_texts)
                    y_true = np.asarray(context_labels(corrupted_test))
                    metrics = evaluate_predictions(y_true, pred, scores)
                    metrics["attack_success_rate"] = attack_success_rate(y_true, pred, counts)
                    count_metrics = aggregate_counts(counts)
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": llm_name,
                            "noise_level": noise_level,
                            "poison_rate": poison_rate,
                            "poison_strategy": poison_strategy,
                            "seed": seed,
                            "eval_seconds": eval_seconds,
                            "latency_per_context_ms": float(1000.0 * eval_seconds / max(len(test_texts), 1)),
                            "selector_fit_seconds": 0.0,
                            "train_avg_selected_logs": np.nan,
                            "llm_model": llm_model,
                            "llm_provider": provider,
                            "llm_baseline": baseline_key,
                            "llm_batches": int(np.ceil(len(test_texts) / llm_batch_size)),
                            **metrics,
                            **count_metrics,
                        }
                    )
                method_info[llm_name]["cache_path"] = str(llm.cache_path)
            except Exception as exc:
                method_info[llm_name]["llm_available"] = False
                method_info[llm_name]["error"] = str(exc)

    metrics_df = pd.DataFrame(rows)
    baseline = (
        metrics_df[metrics_df["noise_level"] == min(noise_levels)]
        .groupby("method")["f1"]
        .mean()
        .replace(0, np.nan)
        .to_dict()
    )
    metrics_df["relative_f1"] = metrics_df.apply(
        lambda row: float(row["f1"] / baseline.get(row["method"], np.nan)),
        axis=1,
    )
    metrics_df["f1_drop_from_noise0"] = metrics_df.apply(
        lambda row: float(baseline.get(row["method"], np.nan) - row["f1"]),
        axis=1,
    )
    metrics_path = results_dir / "metrics_by_noise.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary_df = summarize_results(metrics_df)
    summary_df.to_csv(results_dir / "summary_by_method.csv", index=False)
    (results_dir / "method_info.json").write_text(json.dumps(method_info, indent=2), encoding="utf-8")
    write_markdown_tables(metrics_df, summary_df, results_dir)
    return metrics_df, summary_df


def summarize_results(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for method, group in metrics_df.groupby("method"):
        min_noise = group["noise_level"].min()
        max_noise = group["noise_level"].max()
        f1_noise0 = group[group["noise_level"] == min_noise]["f1"].mean()
        f1_noise50 = group[group["noise_level"] == max_noise]["f1"].mean()
        rows.append(
            {
                "method": method,
                "mean_f1": float(group["f1"].mean()),
                "mean_auroc": float(group["auroc"].mean()),
                "mean_auroc_oriented": float(group["auroc_oriented"].mean()),
                "mean_auprc": float(group["auprc"].mean()),
                "mean_f1_macro": float(group["f1_macro"].mean()),
                "min_f1": float(group["f1"].min()),
                "f1_noise0": float(f1_noise0),
                "f1_noise50": float(f1_noise50),
                "mean_context_reduction": float(group["context_reduction"].mean()),
                "mean_token_reduction": float(group["token_reduction"].mean()),
                "mean_selected_logs": float(group["avg_selected_logs"].mean()),
                "mean_selected_tokens": float(group["avg_selected_tokens"].mean()),
                "mean_poison_selected_rate": float(group["poison_selected_rate"].mean()),
                "mean_poison_event_selection_rate": float(group["poison_event_selection_rate"].mean()),
                "mean_attack_success_rate": float(group["attack_success_rate"].mean()),
                "mean_ece": float(group["ece"].mean()),
                "mean_brier": float(group["brier"].mean()),
                "mean_latency_per_context_ms": float(group["latency_per_context_ms"].mean()),
                "total_eval_seconds": float(group["eval_seconds"].sum()),
                "robustness_retention": float(f1_noise50 / f1_noise0) if f1_noise0 else np.nan,
            }
        )
    grouped = pd.DataFrame(rows)
    grouped["rank_mean_f1"] = grouped["mean_f1"].rank(ascending=False, method="min")
    grouped["rank_robustness"] = grouped["robustness_retention"].rank(ascending=False, method="min")
    return grouped.sort_values(["rank_mean_f1", "rank_robustness", "method"]).reset_index(drop=True)


def summarize_by_dataset_method(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in metrics_df.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for (dataset, method), group in metrics_df.groupby(["dataset", "method"]):
        min_noise = group["noise_level"].min()
        max_noise = group["noise_level"].max()
        f1_noise0 = group[group["noise_level"] == min_noise]["f1"].mean()
        f1_noise50 = group[group["noise_level"] == max_noise]["f1"].mean()
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "mean_f1": float(group["f1"].mean()),
                "mean_auroc": float(group["auroc"].mean()),
                "mean_auprc": float(group["auprc"].mean()),
                "mean_f1_macro": float(group["f1_macro"].mean()),
                "min_f1": float(group["f1"].min()),
                "f1_noise0": float(f1_noise0),
                "f1_noise50": float(f1_noise50),
                "robustness_retention": float(f1_noise50 / f1_noise0) if f1_noise0 else np.nan,
                "mean_context_reduction": float(group["context_reduction"].mean()),
                "mean_token_reduction": float(group["token_reduction"].mean()),
                "mean_selected_logs": float(group["avg_selected_logs"].mean()),
                "mean_selected_tokens": float(group["avg_selected_tokens"].mean()),
                "mean_poison_selected_rate": float(group["poison_selected_rate"].mean()),
                "mean_poison_event_selection_rate": float(group["poison_event_selection_rate"].mean()),
                "mean_attack_success_rate": float(group["attack_success_rate"].mean()),
                "mean_latency_per_context_ms": float(group["latency_per_context_ms"].mean()),
                "total_eval_seconds": float(group["eval_seconds"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["rank_in_dataset"] = out.groupby("dataset")["mean_f1"].rank(ascending=False, method="min")
    return out.sort_values(["dataset", "rank_in_dataset", "method"]).reset_index(drop=True)


def legacy_summarize_results(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics_df.groupby("method", as_index=False).agg(
        mean_f1=("f1", "mean"),
        mean_auroc=("auroc", "mean"),
        min_f1=("f1", "min"),
        f1_noise0=("f1", lambda s: float(s.iloc[metrics_df.loc[s.index, "noise_level"].argmin()])),
        f1_noise50=("f1", lambda s: float(s.iloc[metrics_df.loc[s.index, "noise_level"].argmax()])),
        mean_context_reduction=("context_reduction", "mean"),
        mean_selected_logs=("avg_selected_logs", "mean"),
        mean_poison_selected_rate=("poison_selected_rate", "mean"),
        total_eval_seconds=("eval_seconds", "sum"),
    )
    grouped["robustness_retention"] = grouped["f1_noise50"] / grouped["f1_noise0"].replace(0, np.nan)
    grouped["rank_mean_f1"] = grouped["mean_f1"].rank(ascending=False, method="min")
    grouped["rank_robustness"] = grouped["robustness_retention"].rank(ascending=False, method="min")
    return grouped.sort_values(["rank_mean_f1", "rank_robustness", "method"]).reset_index(drop=True)


def write_markdown_tables(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, results_dir: Path) -> None:
    cols = [
        "method",
        "mean_f1",
        "mean_auroc",
        "mean_auprc",
        "mean_f1_macro",
        "min_f1",
        "f1_noise0",
        "f1_noise50",
        "robustness_retention",
        "mean_context_reduction",
        "mean_token_reduction",
        "mean_selected_logs",
        "mean_selected_tokens",
        "mean_poison_selected_rate",
        "mean_poison_event_selection_rate",
        "mean_attack_success_rate",
        "mean_latency_per_context_ms",
    ]
    existing_cols = [col for col in cols if col in summary_df.columns]
    summary_md = summary_df[existing_cols].to_markdown(index=False, floatfmt=".4f")
    dataset_summary = summarize_by_dataset_method(metrics_df)
    dataset_md = ""
    if not dataset_summary.empty:
        dataset_cols = [col for col in ["dataset", *cols] if col in dataset_summary.columns]
        dataset_md = dataset_summary[dataset_cols].to_markdown(index=False, floatfmt=".4f")
    pivot_f1 = metrics_df.pivot_table(index="noise_level", columns="method", values="f1", aggfunc="mean").reset_index()
    pivot_auroc = metrics_df.pivot_table(index="noise_level", columns="method", values="auroc", aggfunc="mean").reset_index()
    text = "\n\n".join(
        [
            "# TeleMin-RAG Results",
            "## Summary by Method",
            summary_md,
            "## Summary by Dataset and Method",
            dataset_md,
            "## F1 by Noise Level",
            pivot_f1.to_markdown(index=False, floatfmt=".4f"),
            "## AUROC by Noise Level",
            pivot_auroc.to_markdown(index=False, floatfmt=".4f"),
        ]
    )
    (results_dir / "tables.md").write_text(text, encoding="utf-8")
