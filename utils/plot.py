"""
utils/plot.py
=============
All plotting functions for the F1 qualifying prediction study.
Figures are saved to outputs/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUT_FIGS


def plot_pred_vs_actual(
    model_name: str, y_test: np.ndarray, y_pred: np.ndarray, path=None
):
    """Scatter plot of predicted vs true grid position."""
    if path is None:
        path = OUT_FIGS / f"{model_name.lower()}_pred_vs_actual.pdf"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color="steelblue")
    ax.plot([1, 20], [1, 20], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True grid position")
    ax.set_ylabel("Predicted grid position")
    ax.set_title(f"{model_name}: Predicted vs True Grid Position")
    ax.set_xlim(1, 20)
    ax.set_ylim(1, 20)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close()


def plot_residuals(model_name: str, y_test: np.ndarray, y_pred: np.ndarray, path=None):
    """Residual plot (true − predicted) against predicted value."""
    if path is None:
        path = OUT_FIGS / f"{model_name.lower()}_residuals.pdf"

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.scatter(y_pred, residuals, alpha=0.35, s=18, color="coral")
    ax.set_xlabel("Predicted grid position")
    ax.set_ylabel("Residual (true − predicted)")
    ax.set_title(f"{model_name}: Residual Plot")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close()


def plot_model_comparison(results_list: list, metric: str = "MAE", path=None):
    """
    Horizontal bar chart comparing all models on a single metric.

    Parameters
    ----------
    results_list : list of dicts returned by evaluate()
    metric       : one of 'MAE', 'RMSE', 'R2', 'Spearman_rho',
                   'Top1_acc', 'Top3_acc'
    """
    if path is None:
        path = OUT_FIGS / f"comparison_{metric.lower()}.pdf"

    names = [r["model"] for r in results_list]
    values = [r[metric] for r in results_list]

    if metric in ("Top1_acc", "Top3_acc"):
        values = [v * 100 for v in values]
        xlabel = f"{metric} (%)"
    else:
        xlabel = metric

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(names, values, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Model comparison — {xlabel}")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close()


def plot_search_results(model_name: str, search, path=None):
    """CV MAE across all RandomizedSearchCV trials (search trajectory)."""
    if path is None:
        path = OUT_FIGS / f"{model_name.lower()}_search_results.pdf"

    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values("rank_test_score")
    maes = -results["mean_test_score"].values

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(maes, marker="o", markersize=4, linewidth=1, color="steelblue")
    ax.axhline(
        maes.min(),
        color="red",
        linestyle="--",
        linewidth=0.9,
        label=f"Best MAE = {maes.min():.3f}",
    )
    ax.set_xlabel("Trial (sorted by CV MAE)")
    ax.set_ylabel("CV MAE (grid positions)")
    ax.set_title(f"{model_name}: Hyperparameter Search Results")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close()
