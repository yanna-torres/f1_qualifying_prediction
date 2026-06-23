# F1 Qualifying Grid Position Prediction

Regression study comparing five machine learning models for predicting Formula 1 qualifying session grid positions during the ground-effect regulatory era (2022–2025).

Developed as part of a graduate-level machine learning course paper.

---

## Models compared

| Model | Implementation |
|---|---|
| Linear Regression | `sklearn.linear_model.LinearRegression` |
| Support Vector Regression (SVR) | `sklearn.svm.SVR` |
| Gradient Boosting | `lightgbm` |
| Multilayer Perceptron (MLP) | `sklearn.neural_network.MLPRegressor` |
| Gaussian Process Regression (GPR) | `sklearn.gaussian_process.GaussianProcessRegressor` |

---

## Repository structure

```
f1_qualifying_prediction/
│
├── config.py
├── run_all.py
├── requirements.txt
├── README.md
│
├── utils/
│   └── (helper functions)
│
├── models/
│   ├── lr_pipeline.py
│   ├── svr_pipeline.py
│   ├── gb_pipeline.py
│   ├── mlp_pipeline.py
│   └── gpr_pipeline.py
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
python models/mlp_pipeline.py
```

or

```bash
python run_all.py --mlp
```

Each pipeline script is self-contained and can be run independently.

---

## Data

The dataset is built from FastF1 timing data covering the 2022–2025 Formula 1 seasons (the ground-effect regulatory era). The raw wide-format dataset, `qualifying_dataset_wide_with_fp.csv`, is not included in the repository; place it at `data/qualifying_dataset_wide_with_fp.csv` before running `python data.py`.

**Train / test split:** seasons 2022–2024 are used for training (1,355 observations); the 2025 season is held out entirely as the test set (476 observations). This temporal split avoids leakage from within-season correlation between races and mirrors real-world deployment conditions, where a model must predict a season it has not seen.

**Avoiding target leakage:** any feature generated during or after the qualifying session itself (lap times, sector times, speed-trap measurements, tire state, and which sub-session a driver was eliminated in) is excluded, since these are functions of the very session that produces `GridPosition` and would not be available at the moment a real prediction needs to be made. See `purge_data_leakage()` in `data.py`.

**Features used (22):** session identifiers (`year`, `Round`, `Circuit`, `Driver`, `Team`, label-encoded), ambient and track conditions (air/track temperature, humidity, pressure, wind speed, rainfall, wet flag), Free Practice performance expressed as a relative delta to the session's fastest lap (`FP1_s_Delta_pct`, `FP2_s_Delta_pct`, `FP3_s_Delta_pct`; see `apply_feature_engineering()`), a sprint-weekend flag, and six manually compiled circuit characteristics (`circuit_layout`, `circuit_speed`, `circuit_character`, `track_length_km`, `num_corners`, `elevation_change_m`; see `circuit_metadata.py`). Label encoding mappings for all categorical columns are saved to `data/label_encoders.json` for later decoding (e.g. when analysing saved prediction CSVs).

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
TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASONS  = [2025]
TARGET_COL    = "GridPosition"
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