"""
run_all.py
==========
Runs one or more model pipelines, optionally under a named feature
ablation, then produces the aggregated results table and comparison
plots.

Usage:
    python run_all.py                          # all models, full features
    python run_all.py --mlp                    # only MLP, full features
    python run_all.py --ablation no_fp          # all models, without FP data
    python run_all.py --mlp --ablation no_fp    # only MLP, without FP data
    python run_all.py --list                    # show available model flags
    python run_all.py --list-ablations           # show available ablations
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent / "models"))

from utils import save_results_table, plot_model_comparison
from config import ABLATIONS

from models import lr_pipeline
from models import mlp_pipeline
from models import gpr_pipeline
from models import gb_pipeline
from models import svr_pipeline

# Maps CLI flag name -> pipeline module
PIPELINE_REGISTRY = {
    "lr": lr_pipeline,
    "mlp": mlp_pipeline,
    "gpr": gpr_pipeline,
    "gb": gb_pipeline,
    "svr": svr_pipeline,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run F1 qualifying prediction model pipelines."
    )
    for flag in PIPELINE_REGISTRY:
        parser.add_argument(
            f"--{flag}",
            action="store_true",
            help=f"Run only the {flag.upper()} pipeline.",
        )
    parser.add_argument(
        "--ablation",
        default="full",
        choices=list(ABLATIONS.keys()),
        help="Feature ablation to run (default: full, i.e. no exclusions).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available model flags and exit.",
    )
    parser.add_argument(
        "--list-ablations",
        action="store_true",
        help="List available ablation names and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available models:")
        for flag in PIPELINE_REGISTRY:
            print(f"  --{flag}")
        return

    if args.list_ablations:
        print("Available ablations:")
        for tag, cols in ABLATIONS.items():
            print(f"  {tag:<15} drops: {cols if cols else '(none — full feature set)'}")
        return

    selected_flags = [flag for flag in PIPELINE_REGISTRY if getattr(args, flag)]

    # No model flags passed → run everything
    if not selected_flags:
        selected_flags = list(PIPELINE_REGISTRY.keys())

    print(f"Running: {', '.join(selected_flags)}   |   ablation: {args.ablation}\n")

    results = []
    for flag in selected_flags:
        pipeline = PIPELINE_REGISTRY[flag]
        r = pipeline.main(ablation=args.ablation)
        results.append(r)

    print("\n\n" + "=" * 55)
    print("  AGGREGATED RESULTS")
    print("=" * 55)

    save_results_table(results)

    if len(results) > 1:
        for metric in ["MAE", "RMSE", "R2", "Spearman_rho", "Top3_acc"]:
            plot_model_comparison(results, metric=metric)
    else:
        print("\n  Only one model run — skipping comparison plots.")


if __name__ == "__main__":
    main()
