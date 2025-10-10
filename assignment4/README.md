# Assignment 4 — Homogeneous vs Heterogeneous Ensembles (Breast Cancer)

This repository implements a complete, **ready-to-run** experiment comparing a **homogeneous ensemble** (Random Forest)
against a **heterogeneous ensemble** (soft/weighted Voting over diverse base learners) on the provided *breastCancer.csv* dataset.

The code follows a rigorous **nested cross-validation** protocol (5 outer folds × 3 inner folds), includes **calibration**, **statistical testing**,
**explainability** via permutation importance, and generates all figures/tables required for the report.

## Quick Start

```bash
# 1) Create virtual env (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Run
python -m src.main --data data/breastCancer.csv --out results --seed 42
```

Outputs (CSV + PNG) are saved under `results/`.

## Project Structure

```
assignment4/
  data/
    breastCancer.csv
  src/
    dataset.py          # loading, schema detection, summary
    preprocessing.py    # ColumnTransformer pipelines
    models.py           # model builders + param grids
    evaluation.py       # metrics, curves, confusion matrices
    importance.py       # permutation importance aggregation
    stats.py            # paired tests
    experiment.py       # nested CV runner/orchestrator
    plots.py            # plotting utilities
    main.py             # CLI entrypoint
  results/
    figures/            # generated figures
  report/
    (reserved for LaTeX later)
  requirements.txt
  README.md
```

## Notes & Assumptions
- The target is `diagnosis` with labels `{B, M}` mapped to `{0, 1}` respectively.
- Feature `id` is dropped; `gender` is treated as categorical when present.
- All preprocessing steps are performed **inside** cross-validation to prevent leakage.
- Primary metric: **ROC AUC**; also reports Accuracy, Precision, Recall, F1, PR-AUC, Brier.
- Statistical comparison uses **Wilcoxon signed-rank** on outer-fold paired metrics.

## Reproducibility
- Set `--seed` to ensure determinism in splits and certain algorithms.
- Library versions are pinned loosely in `requirements.txt`.

