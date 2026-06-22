"""
run_all.py
==========
Runs all five model pipelines in sequence, then produces the
aggregated results table and comparison plots.

Usage:
    python run_all.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))

from utils import save_results_table, plot_model_comparison


# Uncomment as you implement each model:
from models import lr_pipeline
from models import mlp_pipeline
from models import gpr_pipeline
# from models import svr_pipeline
# from models import gb_pipeline

PIPELINES = [
    lr_pipeline,
    mlp_pipeline,
    gpr_pipeline,
    # lr_pipeline,
    # svr_pipeline,
    # gb_pipeline,
]


def main():
    results = []
    for pipeline in PIPELINES:
        r = pipeline.main()
        results.append(r)

    print("\n\n" + "=" * 55)
    print("  AGGREGATED RESULTS")
    print("=" * 55)

    save_results_table(results)

    for metric in ["MAE", "RMSE", "R2", "Spearman_rho", "Top3_acc"]:
        plot_model_comparison(results, metric=metric)


if __name__ == "__main__":
    main()
