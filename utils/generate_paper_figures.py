"""
utils/generate_paper_figures.py
==========================
Generates the three figures used in the Experiments section of the
paper, reading directly from the project's actual results:

  Figure 1: Overall Model Comparison
            results_table.csv, "full" ablation rows only.
  Figure 2: Feature Ablation Analysis (delta-MAE heatmap)
            results_table.csv, all rows, pivoted by model x ablation.
  Figure 3: Qualitative Error Analysis (best model: Gradient Boosting)
            <model>_predictions.csv, decoded with label_encoders.json.

Usage
-----
    python generate_paper_figures.py

Outputs (PDF, vector, ready for \\includegraphics):
    outputs/figures/paper/fig1_overall_comparison.pdf
    outputs/figures/paper/fig2_ablation_heatmap.pdf
    outputs/figures/paper/fig3_error_analysis.pdf
"""

import json
import sys
from pathlib import Path

# Injeta a raiz do projeto no path para importar o config
sys.path.insert(0, str(Path(__file__).parent.parent))


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import OUT_RESULTS, OUT_PREDS, DATA_DIR, OUT_FIGS

OUT_PAPER_FIGS = OUT_FIGS / "paper"
OUT_PAPER_FIGS.mkdir(parents=True, exist_ok=True)

ENCODERS_PATH = DATA_DIR / "label_encoders.json"

# Which model's predictions to use for the Qualitative Error Analysis.
# Change this if a different model ends up being the best performer.
BEST_MODEL_FILE = OUT_PREDS / "gradientboosting_predictions.csv"
BEST_MODEL_LABEL = "Gradient Boosting"

MODEL_COLORS = {
    "GradientBoosting": "#1D9E75",
    "SVR": "#378ADD",
    "MLP": "#7F77DD",
    "LinearRegression": "#888780",
    "GPR": "#D85A30",
}
MODEL_DISPLAY_NAMES = {
    "GradientBoosting": "Gradient Boosting",
    "SVR": "SVR",
    "MLP": "MLP",
    "LinearRegression": "Linear Regression",
    "GPR": "GPR",
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def load_label_encoders():
    """Returns a dict of {column_name: {int_code: original_label}}."""
    if not ENCODERS_PATH.exists():
        print(
            f"  [WARNING] {ENCODERS_PATH} not found. Driver/Team/Circuit "
            f"will remain as integer codes in Figure 3. Re-run data.py "
            f"to generate this file."
        )
        return {}
    with open(ENCODERS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # JSON keys are always strings; convert back to int for lookup
    return {
        col: {int(k): v for k, v in mapping.items()} for col, mapping in raw.items()
    }


def decode_column(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).fillna(series.astype(str))


# ──────────────────────────────────────────────────────────────────
# Figure 1: Overall Model Comparison
# ──────────────────────────────────────────────────────────────────


def make_figure1():
    df = pd.read_csv(OUT_RESULTS)
    # "full" rows are those with no underscore in the model name
    full = df[~df["Model"].str.contains("_")].copy()

    metrics = [
        ("MAE", "MAE (grid positions, lower is better)", False),
        ("Spearman_rho", "Spearman correlation (higher is better)", True),
        ("R2", "R-squared (higher is better)", True),
        ("Top3_acc_%", "Top-3 accuracy (%, higher is better)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()

    for ax, (col, title, higher_better) in zip(axes, metrics):
        ordered = full.sort_values(col, ascending=not higher_better)
        names = [MODEL_DISPLAY_NAMES.get(m, m) for m in ordered["Model"]]
        colors = [MODEL_COLORS.get(m, "#888780") for m in ordered["Model"]]
        values = ordered[col].values

        bars = ax.barh(names, values, color=colors, edgecolor="white")
        ax.bar_label(
            bars, fmt="%.3f" if col != "Top3_acc_%" else "%.1f", padding=4, fontsize=9
        )
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "Overall model comparison (full feature set, 2025 test season)", fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUT_PAPER_FIGS / "fig1_overall_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 2: Feature Ablation Analysis (delta-MAE heatmap)
# ──────────────────────────────────────────────────────────────────


def make_figure2():
    df = pd.read_csv(OUT_RESULTS)
    df["base_model"] = df["Model"].apply(lambda x: x.split("_")[0] if "_" in x else x)
    df["ablation"] = df["Model"].apply(
        lambda x: "_".join(x.split("_")[1:]) if "_" in x else "full"
    )

    pivot = df.pivot(index="base_model", columns="ablation", values="MAE")

    # Column order: full first, then the rest alphabetically
    other_cols = sorted(c for c in pivot.columns if c != "full")
    pivot = pivot[["full"] + other_cols]

    # Row order: best-to-worst by "full" MAE, for readability
    pivot = pivot.loc[pivot["full"].sort_values().index]

    baseline = pivot["full"]
    delta = pivot.sub(baseline, axis=0)

    display_rows = [MODEL_DISPLAY_NAMES.get(m, m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Diverging colormap centred at 0: green = improvement, coral/red = degradation
    vmax = max(abs(delta.values.min()), abs(delta.values.max()), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn_r")

    im = ax.imshow(delta.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(display_rows)))
    ax.set_yticklabels(display_rows, fontsize=9)

    # Annotate each cell with the raw MAE value (not the delta), so the
    # figure is self-contained without needing the results table.
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            d = delta.values[i, j]
            text_color = "black" if abs(d) < vmax * 0.6 else "white"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("$\\Delta$ MAE vs. full feature set", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Feature ablation analysis: MAE by model and ablation "
        "(cell values = MAE; color = change from full feature set)",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = OUT_PAPER_FIGS / "fig2_ablation_heatmap.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 3: Qualitative Error Analysis
# ──────────────────────────────────────────────────────────────────


def make_figure3():
    if not BEST_MODEL_FILE.exists():
        print(f"  [WARNING] {BEST_MODEL_FILE} not found, skipping Figure 3.")
        return

    preds = pd.read_csv(BEST_MODEL_FILE)
    encoders = load_label_encoders()

    if "Circuit" in encoders:
        preds["Circuit_name"] = decode_column(preds["Circuit"], encoders["Circuit"])
    if "Driver" in encoders:
        preds["Driver_name"] = decode_column(preds["Driver"], encoders["Driver"])

    y_true = preds["GridPosition_true"].values
    y_pred = preds["GridPosition_pred"].values
    residual = preds[
        "residual"
    ].values  # convention: true - pred (per save_enriched_predictions)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: predicted vs true scatter ---
    ax = axes[0]
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.4,
        s=22,
        color=MODEL_COLORS.get("GradientBoosting", "#1D9E75"),
    )
    ax.plot([1, 20], [1, 20], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True grid position")
    ax.set_ylabel("Predicted grid position")
    ax.set_title(f"{BEST_MODEL_LABEL}: predicted vs. true grid position")
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Right: mean signed residual by true position (bias plot) ---
    ax = axes[1]
    by_pos = pd.Series(residual, index=y_true).groupby(level=0).mean().sort_index()
    colors = ["#85B7EB" if v < 0 else "#F0997B" for v in by_pos.values]
    ax.bar(by_pos.index, by_pos.values, color=colors, edgecolor="white", width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("True grid position")
    ax.set_ylabel("Mean residual (true $-$ predicted)")
    ax.set_title(f"{BEST_MODEL_LABEL}: systematic bias by grid position")
    ax.set_xticks(range(1, 21))
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    out_path = OUT_PAPER_FIGS / "fig3_error_analysis.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # --- Print the worst-predicted race weekend for discussion in text ---
    preds["abs_residual"] = preds["residual"].abs()
    worst_round = (
        preds.groupby(
            ["Round", "Circuit_name"]
            if "Circuit_name" in preds.columns
            else ["Round", "Circuit"]
        )["abs_residual"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n  Worst-predicted race weekends (mean |residual| across the field):")
    print(worst_round.head(5).to_string())

    print("\n  Largest individual prediction errors:")
    cols = [
        "Round",
        "Driver_name" if "Driver_name" in preds.columns else "Driver",
        "Circuit_name" if "Circuit_name" in preds.columns else "Circuit",
        "GridPosition_true",
        "GridPosition_pred",
        "residual",
    ]
    print(
        preds.reindex(columns=cols)
        .loc[preds["abs_residual"].sort_values(ascending=False).index[:10]]
        .to_string(index=False)
    )


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


def main():
    print("Generating Figure 1: Overall Model Comparison ...")
    make_figure1()

    print("\nGenerating Figure 2: Feature Ablation Analysis ...")
    make_figure2()

    print("\nGenerating Figure 3: Qualitative Error Analysis ...")
    make_figure3()

    print(f"\nAll figures saved to: {OUT_PAPER_FIGS}")


if __name__ == "__main__":
    main()
