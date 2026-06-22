"""
models/svr_pipeline.py
========================
Support Vector Regression (SVR) model for F1 qualifying grid
position prediction.

Supports feature ablation studies via the `ablation` parameter to
main(), using the named column sets defined in config.ABLATIONS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.stats import loguniform
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE, CV_FOLDS, ABLATIONS
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


def main(ablation: str = "full"):
    """
    Parameters
    ----------
    ablation : str
        Key into config.ABLATIONS. "full" (default) uses every
        available feature. Any other key drops the extra columns
        defined for that ablation before training.
    """
    if ablation not in ABLATIONS:
        raise ValueError(
            f"Unknown ablation '{ablation}'. Available: {list(ABLATIONS.keys())}"
        )

    model_tag = MODEL_NAME if ablation == "full" else f"{MODEL_NAME}_{ablation}"

    print(f"\n{'=' * 55}\n  {model_tag} pipeline\n{'=' * 55}")

    X_train, X_test, y_train, y_test, train_df, test_df = load_and_split(
        extra_drop_cols=ABLATIONS[ablation]
    )

    model, search = tune(X_train, y_train)
    y_pred = model.predict(X_test)

    results = evaluate(model_tag, y_test, y_pred, label="held-out test season")

    save_model(
        model, model_tag, best_params=search.best_params_, cv_mae=-search.best_score_
    )

    plot_pred_vs_actual(model_tag, y_test, y_pred)
    plot_residuals(model_tag, y_test, y_pred)
    plot_search_results(model_tag, search)
    save_enriched_predictions(model_tag, test_df, y_test, y_pred)

    return results


if __name__ == "__main__":
    main()
