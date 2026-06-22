"""
models/svr_pipeline.py
========================
Support Vector Regression (SVR) model for F1 qualifying grid
position prediction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.stats import loguniform
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline

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

MODEL_NAME = "SVR"
N_ITER = 40


def build_pipeline():
    svr = SVR()
    return Pipeline([("scaler", StandardScaler()), ("svr", svr)])


def tune(X_train, y_train):
    pipe = build_pipeline()

    param_dist = {
        "svr__kernel": ["rbf", "linear", "poly"],
        "svr__C": loguniform(1e-1, 1e3),
        "svr__epsilon": loguniform(1e-3, 1e0),
        "svr__gamma": loguniform(1e-4, 1e0),
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipe,
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
