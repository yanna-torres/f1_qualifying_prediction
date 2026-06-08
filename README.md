# F1 Qualifying Grid Position Prediction

Regression study comparing five machine learning models for predicting Formula 1 qualifying session grid positions during the ground-effect regulatory era (2022–2025).

Developed as part of a graduate-level machine learning course paper.

---

## Models compared

| Model | Implementation |
|---|---|
| Linear Regression | `sklearn.linear_model.LinearRegression` |
| Support Vector Regression (SVR) | `sklearn.svm.SVR` |
| Gradient Boosting | `xgboost` / `lightgbm` |
| Multilayer Perceptron (MLP) | `sklearn.neural_network.MLPRegressor` |
| Gaussian Process Regression (GPR) | `sklearn.gaussian_process.GaussianProcessRegressor` |

---

## Repository structure

```f1_qualifying_prediction/
│
├── config.py
├── run_all.py
├── requirements.txt
├── README.md
│
├── utils/
│   ├── __init__.py   ← re-exports everything
│   ├── data.py       ← load_and_split
│   ├── eval.py       ← evaluate, save_results_table, save_enriched_predictions
│   └── plot.py       ← all plot functions
│
├── models/
│   ├── mlp_f1_pipeline.py
│   └── ...
│
├── data/
└── outputs/
    ├── predictions/
    └── figures/
```

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

---

## Usage

### Run all models

```bash
python run_all.py
```

This runs each model pipeline in sequence, saves per-model predictions and figures to `outputs/`, and writes the aggregated metrics table to `outputs/results_table.csv`.

### Run a single model

```bash
python models/mlp_f1_pipeline.py
```

Each pipeline script is self-contained and can be run independently.

---

## Data

The dataset `f1_v3_predictive.csv` covers the 2022–2024 Formula 1 seasons and includes qualifying session results, championship standings, circuit characteristics, driver history, and weather conditions. It is not included in the repository. Place the file at `data/f1_v3_predictive.csv` before running.

**Train / test split:** seasons 2022–2023 are used for training; the 2024 season is the held-out test set. This temporal split avoids data leakage and mirrors real-world deployment conditions.

**Features used (29):** qualifying session reached, best qualifying time, gap to pole position, driver and constructor championship standings, rolling performance metrics, circuit history features, weather variables, and target-encoded identifiers for driver, constructor, and circuit.

---

## Evaluation metrics

All models are evaluated on the same held-out test set using:

| Metric | Description |
|---|---|
| MAE | Mean absolute error in grid positions |
| RMSE | Root mean squared error in grid positions |
| R² | Coefficient of determination |
| Spearman ρ | Rank correlation between predicted and true grid order |
| Top-1 accuracy | Fraction of predictions within ±1 grid position |
| Top-3 accuracy | Fraction of predictions within ±3 grid positions |

---

## Configuration

All shared settings are defined in `config.py`:

```python
TRAIN_SEASONS = [2022, 2023]
TEST_SEASONS  = [2024]
TARGET_COL    = "quali_position"
RANDOM_STATE  = 42
```

Changing seasons, the target column, or the feature exclusion list here propagates automatically to all model scripts.

---

## Outputs

After `run_all.py` completes, the `outputs/` directory contains:

- `results_table.csv` — all metrics for all models in one table
- `figures/comparison_mae.pdf` — bar chart comparing MAE across models
- `figures/<model>_pred_vs_actual.pdf` — predicted vs true grid position
- `figures/<model>_residuals.pdf` — residual plot
- `figures/<model>_search_results.pdf` — hyperparameter search trajectory
- `predictions/<model>_predictions.csv` — per-driver, per-race predictions with metadata