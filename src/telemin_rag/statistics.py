from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_PAIR_COLS = ["dataset", "seed", "noise_level", "poison_strategy"]


def mean_std_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(clean))) if len(clean) > 1 else 0.0
    return mean, std, ci95


def summarize_uncertainty(
    metrics_df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or ["method"]
    metrics = metrics or [
        "f1",
        "f1_macro",
        "auroc",
        "auprc",
        "context_reduction",
        "token_reduction",
        "poison_event_selection_rate",
        "attack_success_rate",
        "latency_per_context_ms",
    ]
    rows: list[dict] = []
    for keys, group in metrics_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric not in group.columns:
                continue
            mean, std, ci95 = mean_std_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        rows.append(row)
    out = pd.DataFrame(rows)
    if "f1_mean" in out.columns:
        out["rank_f1"] = out["f1_mean"].rank(ascending=False, method="min")
        out = out.sort_values(["rank_f1", "method"] if "method" in out.columns else group_cols)
    return out.reset_index(drop=True)


def wilcoxon_against(
    metrics_df: pd.DataFrame,
    *,
    reference: str = "TeleMin-RAG",
    metric: str = "f1",
    pair_cols: list[str] | None = None,
) -> pd.DataFrame:
    pair_cols = pair_cols or [col for col in DEFAULT_PAIR_COLS if col in metrics_df.columns]
    if metric not in metrics_df.columns or "method" not in metrics_df.columns:
        return pd.DataFrame()
    pivot = metrics_df.pivot_table(index=pair_cols, columns="method", values=metric, aggfunc="mean")
    if reference not in pivot.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    ref = pivot[reference]
    for method in pivot.columns:
        if method == reference:
            continue
        paired = pd.concat([ref.rename("reference"), pivot[method].rename("baseline")], axis=1).dropna()
        if len(paired) < 2:
            stat = float("nan")
            pvalue = float("nan")
        else:
            diff = paired["reference"] - paired["baseline"]
            if np.allclose(diff, 0.0):
                stat = 0.0
                pvalue = 1.0
            else:
                stat, pvalue = wilcoxon(paired["reference"], paired["baseline"], zero_method="wilcox", alternative="two-sided")
        rows.append(
            {
                "reference": reference,
                "baseline": method,
                "metric": metric,
                "n_pairs": int(len(paired)),
                "mean_delta": float((paired["reference"] - paired["baseline"]).mean()) if len(paired) else float("nan"),
                "wilcoxon_stat": float(stat),
                "p_value": float(pvalue),
                "significant_0_05": bool(pvalue < 0.05) if np.isfinite(pvalue) else False,
            }
        )
    return pd.DataFrame(rows).sort_values(["p_value", "baseline"], na_position="last").reset_index(drop=True)


def latex_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex_tables(metrics_df: pd.DataFrame, out_dir: str | Path, reference: str = "TeleMin-RAG") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = summarize_uncertainty(metrics_df)
    summary.to_csv(out_path / "summary_with_uncertainty.csv", index=False)
    wilcoxon_df = wilcoxon_against(metrics_df, reference=reference)
    wilcoxon_df.to_csv(out_path / "wilcoxon_tests.csv", index=False)

    table_cols = [
        ("f1", "F1"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("context_reduction", "Ctx. red."),
        ("poison_event_selection_rate", "Poison sel."),
        ("latency_per_context_ms", "Latency ms"),
    ]
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & F1 & AUROC & AUPRC & Ctx. red. & Poison sel. & Latency ms \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        cells = [str(row["method"])]
        for metric, _ in table_cols:
            cells.append(latex_mean_std(row.get(f"{metric}_mean", np.nan), row.get(f"{metric}_std", np.nan)))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (out_path / "table_overall_mean_std.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not wilcoxon_df.empty:
        lines = [
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Baseline & $n$ & $\\Delta$ F1 & $p$-value \\\\",
            "\\midrule",
        ]
        for _, row in wilcoxon_df.iterrows():
            marker = "$^*$" if row["significant_0_05"] else ""
            lines.append(
                f"{row['baseline']} & {int(row['n_pairs'])} & {row['mean_delta']:.4f} & {row['p_value']:.4g}{marker} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        (out_path / "table_wilcoxon.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_latex_tables(metrics_df: pd.DataFrame, out_dir: str | Path, reference: str = "TeleMin-RAG") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if "dataset" not in metrics_df.columns:
        return

    summary = summarize_uncertainty(metrics_df, group_cols=["dataset", "method"])
    summary.to_csv(out_path / "summary_by_dataset_with_uncertainty.csv", index=False)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Method & F1 & AUROC & AUPRC & Poison sel. \\\\",
        "\\midrule",
    ]
    for dataset, group in summary.groupby("dataset", sort=True):
        ordered = group.sort_values("f1_mean", ascending=False)
        first = True
        for _, row in ordered.iterrows():
            label = str(dataset) if first else ""
            first = False
            cells = [
                label,
                str(row["method"]),
                latex_mean_std(row.get("f1_mean", np.nan), row.get("f1_std", np.nan)),
                latex_mean_std(row.get("auroc_mean", np.nan), row.get("auroc_std", np.nan)),
                latex_mean_std(row.get("auprc_mean", np.nan), row.get("auprc_std", np.nan)),
                latex_mean_std(
                    row.get("poison_event_selection_rate_mean", np.nan),
                    row.get("poison_event_selection_rate_std", np.nan),
                ),
            ]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (out_path / "table_by_dataset_mean_std.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    wilcoxon_rows = []
    for dataset, group in metrics_df.groupby("dataset", sort=True):
        tests = wilcoxon_against(group, reference=reference)
        if tests.empty:
            continue
        tests.insert(0, "dataset", dataset)
        wilcoxon_rows.append(tests)
    if wilcoxon_rows:
        tests = pd.concat(wilcoxon_rows, ignore_index=True)
        tests.to_csv(out_path / "wilcoxon_by_dataset.csv", index=False)
        lines = [
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "Dataset & Baseline & $n$ & $\\Delta$ F1 & $p$-value \\\\",
            "\\midrule",
        ]
        for _, row in tests.iterrows():
            marker = "$^*$" if row["significant_0_05"] else ""
            lines.append(
                f"{row['dataset']} & {row['baseline']} & {int(row['n_pairs'])} & {row['mean_delta']:.4f} & {row['p_value']:.4g}{marker} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        (out_path / "table_wilcoxon_by_dataset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
