# TeleMin-RAG A-Tier Revision Report

## Major Scientific Changes

- Reframed TeleMin-RAG as a robust context-construction layer and security boundary, not as a universal replacement for retrieval or LLM triage.
- Added a formal threat model with attacker capabilities, defender capabilities, random noise, contextual poisoning, poison selection rate, and downstream misclassification under poisoning.
- Separated native-label datasets (BGL, UNSW-NB15) from derived-label stress tests (Apache, HPC).
- Added multi-seed evaluation with seeds `[1, 2, 3, 4, 5]`, mean/std tables, Wilcoxon signed-rank tests, and CI-capable plots.
- Added stronger poisoning modes: instruction-like, obfuscated instruction, and semantic false-evidence poisoning.
- Added ablations: without poisoning penalty, without bootstrap robustness, without rarity, without semantic similarity, without redundancy, fixed-k, and adaptive-k.
- Added additional metrics: macro-F1, AUPRC, Brier score, ECE, token reduction, poison event selection rate, attack success rate, and latency.
- Added bounded multi-dataset/multi-seed LLM baselines through Ollama Cloud `gpt-oss:20b-cloud`: zero-shot full context, few-shot full context, few-shot BM25 context, and few-shot TeleMin context.
- Added random-forest downstream-model support to evaluate TeleMin-RAG as preprocessing.
- Expanded the LaTeX article with Related Work, Threat Model, Ethical Considerations, Ablation Study, Error Analysis, mathematical exposure analysis, and LLM multi-dataset analysis.
- Added and verified recent 2025-2026 SOC/LLM/RAG-poisoning references, including AACT, SOC LLM survey work, human-machine co-teaming for SOCs, AIDR, Joint-GCG, Practical Poisoning RAG, RevPRAG, FilterRAG, RAGForensics, PoisonArena, and CPA-RAG.
- Cleaned BibTeX metadata for the final build; the last compile has no BibTeX warnings.
- Added `ARTIFACT.md`, `requirements-lock.txt`, `environment.yml`, and `Dockerfile` for anonymous artifact release.

## Key Results To Use

- Native-label datasets: TeleMin-RAG F1 = `0.866 +/- 0.045`, context reduction = `0.846 +/- 0.042`, poison selection = `0.000 +/- 0.000`.
- Combined native + derived: TeleMin-RAG F1 = `0.728 +/- 0.183`, context reduction = `0.865 +/- 0.040`, poison selection = `0.000 +/- 0.000`.
- Wilcoxon global F1: TeleMin-RAG beats Full Context with `p = 0.0269`, but on native labels it is statistically tied with Embedding Similarity (`p = 0.9799`).
- Dataset-level nuance: TeleMin-RAG wins on BGL and HPC, trails BM25 on Apache, and trails Embedding Similarity on UNSW-NB15.
- Stronger poisoning: TeleMin-RAG reduces poison exposure but does not eliminate it under obfuscated/semantic attacks.
- Bounded LLM study: few-shot TeleMin-selected LLM context is the strongest LLM variant (`0.451 +/- 0.277` F1), avoids poison exposure, and improves over zero-shot full-context LLM (`0.314 +/- 0.154` F1), but remains slower and below the classical TeleMin pipeline (`0.576 +/- 0.244` F1 in the same reduced study).

## Improved Figures

- `figures/telemin_architecture.png`: architecture of TeleMin-RAG as preprocessing/security boundary.
- `figures/experimental_pipeline.png`: reproducible experimental pipeline.
- `figures/paper_final/native_f1_vs_noise.png`: native-label F1 vs noise with confidence intervals.
- `figures/paper_final/native_auprc_vs_noise.png`: native-label AUPRC vs noise with confidence intervals.
- `figures/paper_final/security_utility_pareto.png`: dataset-colored F1 vs poison selection with context reduction encoded by marker size.
- `figures/paper_final/context_reduction_by_dataset.png`: context reduction vs F1 in separate dataset panels with TeleMin/Full/top-method annotations.
- `figures/paper_final/ablation_f1.png`: ablation comparison.
- `figures/paper_final/llm_multidataset_f1.png`: bounded LLM study.

## Improved LaTeX Tables

- `results/multiseed_all_fixed/table_overall_mean_std.tex`
- `results/multiseed_all_fixed/table_by_dataset_mean_std.tex`
- `results/multiseed_all_fixed/table_wilcoxon.tex`
- `results/multiseed_all_fixed/table_wilcoxon_by_dataset.tex`
- `results/multiseed_native_fixed/table_overall_mean_std.tex`
- `results/multiseed_native_fixed/table_by_dataset_mean_std.tex`
- `results/ablation_native_fixed/table_overall_mean_std.tex`
- `results/poisoning_native_fixed/summary_by_strategy_method.csv`
- `results/llm_multidataset/table_overall_mean_std.tex`
- `results/llm_multidataset/table_by_dataset_mean_std.tex`

## Conference-A Submission Checklist

- [x] Claims are nuanced and consistent with per-dataset results.
- [x] Native-label and derived-label datasets are separated.
- [x] Multi-seed results and statistical tests are included.
- [x] Poisoning attacks include instruction-like, obfuscated, and semantic false-evidence variants.
- [x] LLM zero-shot is no longer the only LLM baseline; full, BM25-selected, and TeleMin-selected few-shot variants are included.
- [x] Bounded multi-dataset/multi-seed LLM baselines are executed.
- [x] TeleMin-RAG is evaluated as preprocessing before classical and LLM downstream models.
- [x] Related Work is expanded into SOC triage, log anomaly detection, RAG, RAG poisoning, feature selection, and LLM security triage.
- [x] Threat Model and Ethical Considerations are included.
- [x] Reproducibility commands, environment, seeds, scripts, tables, and figures are documented.
- [x] Bibliography expanded to 50 references with recent 2025-2026 SOC, LLM, and RAG poisoning work.
- [x] 2025-2026 references checked against arXiv/ACL/industry-report metadata in `paper/reference_verification_2025_2026.md`.
- [x] Anonymous artifact placeholder and artifact manifest included; replace `XXXX` with the generated review URL before submission.
- [x] Final Elsevier-style PDF regenerated from the revised LaTeX source.
- [ ] Add full-scale datasets beyond 2k samples before a real A-tier submission.
- [ ] Add an anonymized GitHub URL and commit hash.
- [ ] Add stronger production-grade baselines such as XGBoost/LightGBM and DeepLog/LogAnomaly if compute permits.
- [ ] Run full five-seed, six-noise-level LLM baselines if cloud budget permits.
- [ ] Add adaptive attackers that know the selector.

## Acceptable Claims

- TeleMin-RAG achieves competitive F1 and the most favorable security-utility trade-off in the main instruction-like poisoning benchmark.
- TeleMin-RAG selects zero synthetic instruction-like poison events in the five-seed main benchmark.
- TeleMin-RAG reduces, but does not eliminate, poison exposure under obfuscated and semantic false-evidence attacks.
- TeleMin-RAG is a competitive and auditable preprocessing layer for SOC classifiers and LLM agents.
- TeleMin-selected context is the strongest LLM setting in the bounded LLM study.
- Full Context remains competitive or superior in AUROC/AUPRC.
- TeleMin-RAG is not the top method on every dataset, metric, or downstream model.

## Claims To Avoid

- Avoid: "TeleMin-RAG outperforms all baselines on all datasets."
- Avoid: "Minimal selection is always more robust."
- Avoid: "TeleMin-RAG eliminates poisoning."
- Avoid: "The LLM baseline proves LLMs are unsuitable for SOC triage."
- Avoid: "Apache/HPC validate SOC triage labels equivalently to BGL/UNSW-NB15."
- Avoid: "Zero poison selected" without the qualifier "in our synthetic instruction-like poisoning benchmark."
- Avoid: "The LLM study is full-scale"; it is bounded by dataset size, two seeds, and two noise levels.
- Avoid: "TeleMin-RAG is a universally superior classifier."
- Avoid: "TeleMin-RAG is better than the literature"; the paper evaluates a security-aware context-construction layer under a specific protocol.

## Experiments Still Needed Before a Real A-Tier Submission

- Full-size BGL, HDFS, CIC-IDS2017/CSE-CIC-IDS2018, and larger UNSW-NB15 splits.
- Full five-seed, six-noise-level LLM baselines across all datasets and attacks.
- Stronger calibrated full-context models: XGBoost, LightGBM, and calibrated SVM.
- DeepLog/LogAnomaly/LogBERT baselines on log datasets.
- Sensitivity grid over `k`, `tau`, and all TeleMin weights with validation-only model selection.
- Adaptive poisoning where the attacker optimizes against the TeleMin score.
- Analyst-facing study measuring investigation time, not only classification metrics.
