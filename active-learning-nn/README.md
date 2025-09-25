# Active Learning with Single-Hidden-Layer MLPs (Option 3)

This repository provides a **turnkey skeleton** to complete the Option 3 assignment:
comparing **passive learning (SGD)** vs **active learning via output sensitivity** and **uncertainty sampling**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Download UCI datasets to data/raw/
python scripts/download_data.py

# Run a sample experiment (Digits classification)
python scripts/run_experiment.py --config experiments/configs/cls_digits.yaml

# Aggregate and plot
python scripts/aggregate_results.py
python scripts/plot_results.py
```

## Project layout

```
alnn/
  active_learning/    # AL strategies + loop controller
  config/             # Typed experiment configs
  data/               # Dataset loaders, preprocessing, pool splitters
  metrics/            # Metrics + AUCLC + stats tests
  models/             # Single-hidden-layer MLP
  plots/              # Plot helpers
  training/           # SGD train/eval + early stopping
  utils/              # Seeds, device, logging, timers

experiments/
  configs/            # YAML configs per dataset/setting
  results/            # CSV logs and pickled runs

data/
  raw/                # (Optional) source CSVs
  processed/          # cached/sklearn-processed dumps

report/
  main.tex            # LaTeX skeleton mapped to rubric
  figures/
```

## Datasets (suggested)

**Classification (3):**
- `sklearn.datasets`: `load_breast_cancer`, `load_digits` (8×8), `make_classification` (synthetic, imbalanced)

**Regression (3):**
- UCI Energy Efficiency, Airfoil Self-Noise (place CSVs into `data/raw/`), and a **synthetic piecewise-sine**

You can swap any dataset as long as you meet the spec.

## Reproducible protocol (high level)

- Single hidden layer MLP with **over-provisioned** width (e.g., `H = 4–8× input dim`) and **weight decay**.
- Start with `p0` fraction labeled (e.g., 5–10%), query batches of size `q` until budget `B`.
- Strategies: **Passive**, **Uncertainty (entropy/margin)**, **Sensitivity (∥∂y/∂x∥)**.
- Repeat **R=20 seeds**, **fixed test set**. Report **learning curves** and **AUCLC** with **paired tests**.
