"""
lr_pipeline.py
==============
Baseline Linear Regression pipeline for F1 qualifying position prediction.

Supports feature ablation studies via the `ablation` parameter to
main(), using the named column sets defined in config.ABLATIONS.

Usage
-----
    python models/lr_pipeline.py          # standalone
    python run_all.py                     # via orchestrator
"""

import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    load_and_split,
    evaluate,
    save_enriched_predictions,
    plot_pred_vs_actual,
    plot_residuals,
)
from config import OUT_PREDS, ABLATIONS

MODEL_NAME = "LinearRegression"


def main(ablation: str = "full") -> dict:
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

    print("\n" + "=" * 55)
    print(f"  BASELINE: {model_tag}")
    print("=" * 55)

    # 1. Load preprocessed data (with optional ablation columns dropped)
    X_train, X_test, y_train, y_test, train_df, test_df = load_and_split(
        extra_drop_cols=ABLATIONS[ablation]
    )

    # 2. Build pipeline: scale features, then fit OLS
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    print("  Model trained.")

    # 3. Inference on test set
    y_pred = model.predict(X_test)

    # 4. Standard metrics (MAE, RMSE, R2, Spearman, Top-1/Top-3)
    result = evaluate(model_tag, y_test, y_pred)

    # 5. Error statistics
    residuals = y_test - y_pred
    print("\n-- Error Statistics --------------------------------------")
    print(f"  Mean residual   : {residuals.mean():.4f}  (bias)")
    print(f"  Std  residual   : {residuals.std():.4f}")
    print(f"  Median residual : {np.median(residuals):.4f}")
    print(f"  Min  residual   : {residuals.min():.4f}")
    print(f"  Max  residual   : {residuals.max():.4f}")

    # 6. Predicted vs actual sample (first 20 rows for quick visual check)
    print("\n-- Predicted vs Actual (first 20 test rows) --------------")
    print(f"  {'True':>6}  {'Pred':>6}  {'Error':>7}")
    for true, pred in zip(y_test[:20], y_pred[:20]):
        print(f"  {true:>6.1f}  {pred:>6.2f}  {true - pred:>+7.2f}")

    # 7. Save enriched predictions CSV
    save_enriched_predictions(model_tag, test_df, y_test, y_pred)

    # 8. Save figures
    plot_pred_vs_actual(model_tag, y_test, y_pred)
    plot_residuals(model_tag, y_test, y_pred)

    # 9. Persist trained model
    model_path = OUT_PREDS.parent / "models" / f"{model_tag.lower()}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"  Model saved -> {model_path}")

    # 10. Qualitative baseline analysis
    mae = result["MAE"]
    r2 = result["R2"]
    top3 = result["Top3_acc"] * 100

    print("\n-- Baseline Analysis -------------------------------------")
    if mae < 2.0:
        quality = "strong"
    elif mae < 3.5:
        quality = "adequate"
    else:
        quality = "weak"

    print(f"  Quality: {quality}  (MAE = {mae:.2f} grid positions on test season)")

    if r2 > 0.80:
        print(f"  R2 = {r2:.3f}: model explains most variance in qualifying positions.")
    elif r2 > 0.50:
        print(f"  R2 = {r2:.3f}: model captures a moderate fraction of variance.")
    else:
        print(
            f"  R2 = {r2:.3f}: model struggles to explain position variance --\n"
            f"  likely non-linear relationships or unexploited feature interactions."
        )

    print(
        f"  Top-3 accuracy: {top3:.1f}% of predictions fall within +-3 grid positions.\n"
        f"  This establishes the performance floor for all subsequent models."
    )

    return result


if __name__ == "__main__":
    main()
