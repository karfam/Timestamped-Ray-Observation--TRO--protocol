"""Command-line entry point for the TRO simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, ROOT_OUTPUT_DIR
from .experiments import EXPERIMENTS, run_validation_checks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Timestamped Ray Observation protocol simulator")
    parser.add_argument(
        "--experiment",
        default="all",
        choices=sorted(EXPERIMENTS.keys()) + ["validate"],
        help="Experiment to run",
    )
    parser.add_argument("--runs", type=int, default=20, help="Monte Carlo runs per condition")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed")
    parser.add_argument("--output-dir", type=Path, default=ROOT_OUTPUT_DIR, help="Directory for CSV and figure outputs")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration for supported experiments")
    return parser.parse_args()


def main() -> None:
    """Run the selected simulator experiment."""
    args = parse_args()
    if args.experiment == "validate":
        run_validation_checks()
        print("Validation checks passed.")
        return

    exp_config = ExperimentConfig(
        monte_carlo_runs=args.runs,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        seed=args.seed,
        duration_s=args.duration,
    )
    summary = EXPERIMENTS[args.experiment](exp_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(summary, pd.DataFrame) and args.experiment not in {"sync_vs_tro", "synchronized_vs_tro"}:
        summary.to_csv(args.output_dir / "results_summary.csv", index=False)
    print(f"Done. Outputs saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
