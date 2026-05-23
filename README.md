# TeleMin-RAG

Robust Minimal Telemetry Selection for AI-Assisted SOC Alert Triage under Noise and Poisoning Attacks.

This repository is a reproducible research prototype for selecting the smallest useful subset of telemetry/log events needed to triage a security alert. It supports BGL, UNSW-NB15, Apache, and HPC, creates multi-event alert contexts, injects random telemetry noise and contextual-poisoning events, and compares TeleMin-RAG against retrieval, feature-selection, explanation-based, and LLM baselines.

## Methods

- Full Context
- Random-k Selection
- TF-IDF/BM25 Selection
- Mutual Information Feature Selection
- Embedding Similarity Selection
- SHAP-based Selection
- TeleMin-RAG
- Ollama/Qwen-compatible LLM baselines:
  - zero-shot full context
  - few-shot full context
  - few-shot BM25-selected context
  - few-shot TeleMin-selected context

TeleMin-RAG combines statistical feature importance, semantic similarity to an alert query, rarity, redundancy penalties, and a robustness score based on bootstrap-stable token effects, context coherence, and poisoning signatures. The current default uses fixed weights to avoid validation overfitting; optional validation tuning remains available with `--selector-suite main` and without `--no-auto-tune`.

## Reproduction

For double-blind review, upload this repository to an anonymous artifact service
and replace the placeholder in the paper with the generated link, e.g.
`https://anonymous.4open.science/r/telemin-rag-XXXX/`. The artifact includes
scripts, raw public samples, result CSVs, figures, fixed seeds, `requirements.txt`,
`requirements-lock.txt`, `environment.yml`, and `Dockerfile`. See `ARTIFACT.md`
for the full manifest.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python scripts\run_experiment.py
```

The script downloads missing public CSVs when possible, or uses the files already in `data/raw`.

Main multi-seed experiment used in the revised paper:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python scripts\run_multiseed_experiment.py `
  --datasets BGL,UNSW_NB15,Apache,HPC `
  --seeds 1,2,3,4,5 `
  --max-rows 2000 `
  --no-auto-tune `
  --results-dir results\multiseed_all_fixed `
  --figures-dir figures\multiseed_all_fixed
```

Ablation study:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python scripts\run_multiseed_experiment.py `
  --datasets BGL,UNSW_NB15 `
  --seeds 1,2,3,4,5 `
  --max-rows 2000 `
  --selector-suite ablation `
  --no-auto-tune `
  --results-dir results\ablation_native_fixed `
  --figures-dir figures\ablation_native_fixed
```

Additional poisoning strategies:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python scripts\run_multiseed_experiment.py `
  --datasets BGL,UNSW_NB15 `
  --seeds 1,2 `
  --poison-strategies obfuscated,semantic `
  --max-rows 2000 `
  --no-auto-tune `
  --results-dir results\poisoning_native_fixed `
  --figures-dir figures\poisoning_native_fixed
```

Bounded multi-dataset Ollama Cloud LLM study:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python scripts\run_multiseed_experiment.py `
  --datasets BGL,UNSW_NB15,Apache,HPC `
  --seeds 1,2 `
  --max-rows 600 `
  --noise-levels 0,0.5 `
  --include-llm `
  --llm-provider ollama `
  --llm-model gpt-oss:20b-cloud `
  --llm-baselines all `
  --llm-batch-size 8 `
  --no-auto-tune `
  --results-dir results\llm_multidataset `
  --figures-dir figures\llm_multidataset
```

The LLM baseline uses the cloud model directly; do not run or pull `gpt-oss:20b` locally unless you explicitly want a local baseline.

Outputs:

- `results/metrics_by_noise.csv`
- `results/summary_by_method.csv`
- `results/tables.md`
- `results/table_overall_mean_std.tex`
- `results/table_wilcoxon.tex`
- `results/method_info.json`
- `results/summary_by_dataset_method.csv`
- `results/*/llm_cache_*.json`
- `figures/f1_vs_noise.png`
- `figures/auroc_vs_noise.png`
- `figures/auprc_vs_noise.png`
- `figures/context_reduction_vs_performance.png`
- `figures/poison_selection_vs_noise.png`
- `figures/security_utility_pareto.png`
- `figures/robustness_comparison.png`
- `figures/telemin_architecture.png`
- `figures/experimental_pipeline.png`
- `figures/paper_final/*.png`

## Default Experimental Design

- Datasets: BGL and UNSW-NB15 as native-label primary datasets; Apache and HPC as derived-label stress tests.
- Context construction: non-overlapping chronological windows of 8 log events.
- Target: BGL uses any native alert tag in a window; UNSW-NB15, Apache, and HPC use the center-event label.
- Training: models and selectors fit on clean training contexts only.
- Test attacks: random benign log injection at 0%, 10%, 20%, 30%, 40%, 50%; contextual-poisoning events at 10% of the clean context length.
- Poisoning modes: instruction-like, obfuscated instruction, and semantic false-evidence.
- Metrics: accuracy, precision, recall, positive-class F1, macro-F1, AUROC, AUPRC, Brier score, ECE, selected logs, selected tokens, context/token reduction, poison selection, attack success rate, robustness retention, and execution time.

## Notes

The default BGL_2k run is intentionally lightweight so it can execute locally. BGL and UNSW-NB15 are the native-label primary evidence; Apache and HPC are derived-label stress tests. The bounded LLM study uses two seeds and two noise levels and is not a comprehensive SOTA LLM comparison. For full conference submission, repeat the experiments on larger full-size datasets and report hardware, wall-clock time, anonymous artifact commit hash, and confidence intervals from the generated CSV artifacts.
