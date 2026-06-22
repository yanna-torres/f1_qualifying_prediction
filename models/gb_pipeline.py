"""
models/gb_pipeline.py
======================
Gradient Boosting (LightGBM) regression model for F1 qualifying
grid position prediction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.stats import loguniform, randint
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

from config import RANDOM_STATE, CV_FOLDS
from utils import (
    load_and_split,
    evaluate,
    save_enriched_predictions,
    plot_pred_vs_actual,
    plot_residuals,
    plot_search_results,
    save_model,
)

MODEL_NAME = "GradientBoosting"
N_ITER = 40


def build_model():
    return LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def tune(X_train, y_train):
    model = build_model()

    param_dist = {
        "n_estimators": randint(100, 800),
        "max_depth": randint(3, 12),
        "num_leaves": randint(15, 127),
        "learning_rate": loguniform(1e-3, 3e-1),
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": loguniform(1e-4, 1e1),
        "reg_lambda": loguniform(1e-4, 1e1),
        "min_child_samples": randint(5, 50),
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)

    print(f"\n── Best hyperparameters ({MODEL_NAME}) ──────────────────────")
    for k, v in search.best_params_.items():
        print(f"  {k:<40} {v}")
    print(f"  CV MAE ({CV_FOLDS}-fold): {-search.best_score_:.4f} grid positions")

    return search.best_estimator_, search


def main():
    print(f"\n{'=' * 55}\n  {MODEL_NAME} pipeline\n{'=' * 55}")

    X_train, X_test, y_train, y_test, train_df, test_df = load_and_split()

    model, search = tune(X_train, y_train)
    y_pred = model.predict(X_test)

    results = evaluate(MODEL_NAME, y_test, y_pred, label="held-out test season")

    save_model(
        model, MODEL_NAME, best_params=search.best_params_, cv_mae=-search.best_score_
    )

    plot_pred_vs_actual(MODEL_NAME, y_test, y_pred)
    plot_residuals(MODEL_NAME, y_test, y_pred)
    plot_search_results(MODEL_NAME, search)
    save_enriched_predictions(MODEL_NAME, test_df, y_test, y_pred)

    return results


if __name__ == "__main__":
    main()