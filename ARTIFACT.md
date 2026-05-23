# TeleMin-RAG Artifact Manifest

The reproducibility artifact repository is available at:

`https://github.com/lotfigana6-hub/telemin-rag-artifact`

Record the pushed commit hash in the paper.

## Contents

- `src/telemin_rag/`: TeleMin-RAG implementation, selectors, data builders, LLM wrappers, statistics, and plotting.
- `scripts/`: reproduction entrypoints for single-run, multi-seed, analysis, diagrams, and paper figures.
- `data/raw/`: public sample datasets used in the paper.
- `results/`: generated CSV metrics, uncertainty summaries, Wilcoxon tests, and LaTeX tables.
- `figures/`: generated plots and architecture diagrams.
- `paper/`: Elsevier-style LaTeX manuscript, PDF, BibTeX, revision report, and reference-verification notes.
- `requirements.txt`: lightweight dependency specification.
- `requirements-lock.txt`: exact package versions from the final local environment.
- `environment.yml`: Conda environment bootstrap.
- `Dockerfile`: containerized execution environment.

## Seeds

Main experiments use seeds `1,2,3,4,5`. The bounded LLM study uses seeds `1,2`
and noise levels `0,0.5` because cloud inference latency dominates cost.

## Main Reproduction Commands

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

```powershell
.\.venv\Scripts\python scripts\run_multiseed_experiment.py `
  --datasets BGL,UNSW_NB15,Apache,HPC `
  --seeds 1,2,3,4,5 `
  --max-rows 2000 `
  --no-auto-tune `
  --results-dir results\multiseed_all_fixed `
  --figures-dir figures\multiseed_all_fixed
```

```powershell
.\.venv\Scripts\python scripts\generate_paper_figures.py
```

## Docker

```bash
docker build -t telemin-rag .
docker run --rm -v "$PWD/results:/workspace/results" -v "$PWD/figures:/workspace/figures" telemin-rag
```

## Notes For Reviewers

Apache and HPC are derived-label stress tests. BGL and UNSW-NB15 are the primary
native-label evidence. The LLM study is bounded and should not be interpreted as
a comprehensive state-of-the-art LLM comparison.
