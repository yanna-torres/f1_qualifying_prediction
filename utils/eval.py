"""
eval.py
=============
Model evaluation: metrics, results table, and enriched prediction output.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import joblib
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ROOT, TARGET_COL, META_COLS, OUT_PREDS, OUT_RESULTS

MODELS_DIR = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, model_name: str, best_params: dict = None, cv_mae: float = None):
    # Save model
    path = MODELS_DIR / f"{model_name.lower()}_model.pkl"
    joblib.dump(model, path)
    print(f"  Model saved → {path}")

    # Save hyperparameters and CV score as JSON
    if best_params is not None:
        meta = {
            "model": model_name,
            "best_params": {k: str(v) for k, v in best_params.items()},
            "cv_mae": round(cv_mae, 4) if cv_mae is not None else None,
        }
        meta_path = MODELS_DIR / f"{model_name.lower()}_params.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Params saved → {meta_path}")


def load_model(model_name: str):
    path = MODELS_DIR / f"{model_name.lower()}_model.pkl"
    return joblib.load(path)


def evaluate(
    model_name: str, y_test: np.ndarray, y_pred: np.ndarray, label: str = "test"
) -> dict:
    """
    Compute and print MAE, RMSE, R², Spearman ρ, and Top-1 / Top-3
    position accuracy.

    Returns
    -------
    dict with all metric values and y_pred
    """
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    rho, pval = spearmanr(y_test, y_pred)

    top1 = np.mean(np.abs(y_test - np.round(y_pred)) <= 1)
    top3 = np.mean(np.abs(y_test - np.round(y_pred)) <= 3)

    print(f"\n-- {model_name}  |  {label} " + "-" * 33)
    print(f"  MAE        : {mae:.4f}  grid positions")
    print(f"  RMSE       : {rmse:.4f} grid positions")
    print(f"  R2         : {r2:.4f}")
    print(f"  Spearman r : {rho:.4f}  (p = {pval:.4e})")
    print(f"  Top-1 acc. : {top1 * 100:.1f}%  (within +-1 rank)")
    print(f"  Top-3 acc. : {top3 * 100:.1f}%  (within +-3 ranks)")

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman_rho": rho,
        "Spearman_p": pval,
        "Top1_acc": top1,
        "Top3_acc": top3,
        "y_pred": y_pred,
    }


def save_results_table(results_list: list, path=OUT_RESULTS):
    """
    Aggregate a list of result dicts (one per model) into a comparison
    table and save to CSV.
    """
    rows = []
    for r in results_list:
        rows.append(
            {
                "Model": r["model"],
                "MAE": round(r["MAE"], 4),
                "RMSE": round(r["RMSE"], 4),
                "R2": round(r["R2"], 4),
                "Spearman_rho": round(r["Spearman_rho"], 4),
                "Top1_acc_%": round(r["Top1_acc"] * 100, 1),
                "Top3_acc_%": round(r["Top3_acc"] * 100, 1),
            }
        )
    table = pd.DataFrame(rows).set_index("Model")
    table.to_csv(path)
    print(f"\n  Results table saved -> {path}")
    print(table.to_string())
    return table


def save_enriched_predictions(
    model_name: str,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    path=None,
):
    """
    Save predictions with race and driver metadata attached.
    Output: outputs/predictions/<model_name>_predictions.csv
    """
    if path is None:
        path = OUT_PREDS / f"{model_name.lower()}_predictions.csv"

    available_meta = [c for c in META_COLS if c in test_df.columns]

    out = test_df[available_meta].copy().reset_index(drop=True)
    out[f"{TARGET_COL}_true"] = y_test.astype(int)
    out[f"{TARGET_COL}_pred"] = np.round(y_pred).astype(int)
    out["residual"] = y_test - y_pred

    out.to_csv(path, index=False)
    print(f"  Predictions saved -> {path}")
    return out
