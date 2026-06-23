"""
shap_analysis.py
================
SHAP feature importance analysis for the trained Gradient Boosting
model. Loads the already-fitted model from outputs/models/ (no
retraining) and computes exact SHAP values via shap.TreeExplainer
on the 2025 test set.

Usage:
    python shap_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils import load_and_split, load_model

MODEL_NAME = "GradientBoosting"

OUT_PAPER_FIGS = Path(__file__).parent / "outputs" / "figures" / "paper"
OUT_PAPER_FIGS.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading trained {MODEL_NAME} model (no retraining) ...")
    model = load_model(MODEL_NAME)

    print("Loading test set ...")
    _, X_test, _, y_test, _, _ = load_and_split()

    print("Computing SHAP values (TreeExplainer, exact) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # ── Figure: SHAP summary (beeswarm) ──────────────────────────
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    out_path = OUT_PAPER_FIGS / "fig_shap_summary.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")

    # ── Figure: mean |SHAP value| bar chart (global importance) ──
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    importance.plot(kind="barh", ax=ax, color="#1D9E75", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (impact on predicted grid position)")
    ax.set_title(f"{MODEL_NAME}: global feature importance (SHAP)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_path = OUT_PAPER_FIGS / "fig_shap_importance.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # ── Print ranked importance table for the paper text ─────────
    print("\nFeature importance ranking (mean |SHAP value|):")
    print(importance.round(4).to_string())

    importance.to_csv(
        OUT_PAPER_FIGS.parent / "shap_importance_table.csv", header=["mean_abs_shap"]
    )
    print(f"\n  Table saved -> {OUT_PAPER_FIGS.parent / 'shap_importance_table.csv'}")


if __name__ == "__main__":
    main()
