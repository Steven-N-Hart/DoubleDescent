"""CLI for checking experiment status."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def get_experiment_status(experiment_dir: Path) -> dict:
    """Get status of a single experiment.

    Args:
        experiment_dir: Path to experiment directory.

    Returns:
        Status dictionary.
    """
    status = {
        "experiment_dir": str(experiment_dir),
        "experiment_id": experiment_dir.name,
        "status": "UNKNOWN",
    }

    # Load config
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        status["name"] = config.get("name", "Unknown")
        status["status"] = config.get("status", "UNKNOWN")

    # Load progress
    progress_path = experiment_dir / "progress.json"
    if progress_path.exists():
        with open(progress_path, "r") as f:
            progress = json.load(f)
        status["progress"] = {
            "completed": progress.get("n_completed", 0),
            "total": progress.get("n_total", 0),
            "percentage": 100 * progress.get("n_completed", 0) / max(progress.get("n_total", 1), 1),
        }
        status["last_updated"] = progress.get("last_updated")
    else:
        status["progress"] = {"completed": 0, "total": 0, "percentage": 0}

    # Get completed widths
    runs_dir = experiment_dir / "runs"
    if runs_dir.exists():
        completed_runs = []
        failed_runs = []
        pending_runs = []

        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                run_info_path = run_dir / "run_info.json"
                if run_info_path.exists():
                    with open(run_info_path, "r") as f:
                        run_info = json.load(f)
                    run_status = run_info.get("status", "UNKNOWN")
                    width = run_info.get("width", 0)

                    if run_status == "COMPLETED":
                        completed_runs.append(width)
                    elif run_status in ("FAILED", "SKIPPED"):
                        failed_runs.append(width)
                    else:
                        pending_runs.append(width)

        status["completed_widths"] = completed_runs
        status["failed_widths"] = failed_runs

    return status


def print_status(status: dict, as_json: bool = False) -> None:
    """Print experiment status.

    Args:
        status: Status dictionary.
        as_json: If True, print as JSON.
    """
    if as_json:
        print(json.dumps(status, indent=2))
        return

    print(f"Experiment: {status.get('name', status['experiment_id'])}")
    print(f"Status: {status['status']}")

    progress = status.get("progress", {})
    completed = progress.get("completed", 0)
    total = progress.get("total", 0)
    percentage = progress.get("percentage", 0)
    print(f"Progress: {completed}/{total} widths completed ({percentage:.0f}%)")

    if status.get("last_updated"):
        print(f"Last updated: {status['last_updated']}")

    if status.get("completed_widths"):
        print(f"Completed widths: {', '.join(map(str, status['completed_widths']))}")

    if status.get("failed_widths"):
        print(f"Failed widths: {', '.join(map(str, status['failed_widths']))}")


def list_all_experiments(base_dir: Path) -> list:
    """List all experiments in the base directory.

    Args:
        base_dir: Base experiments directory.

    Returns:
        List of experiment status dictionaries.
    """
    experiments = []

    if not base_dir.exists():
        return experiments

    for exp_dir in sorted(base_dir.iterdir()):
        if exp_dir.is_dir() and (exp_dir / "config.json").exists():
            status = get_experiment_status(exp_dir)
            experiments.append(status)

    return experiments


def main() -> int:
    """Main entry point for status CLI.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.status",
        description="Check status of running or completed experiments.",
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        help="Path to specific experiment directory",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="List all experiments in outputs/experiments/",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Base experiments directory (default: outputs/experiments/)",
    )

    args = parser.parse_args()

    if args.experiment:
        if not args.experiment.exists():
            print(f"ERROR: Experiment not found: {args.experiment}", file=sys.stderr)
            return 1

        status = get_experiment_status(args.experiment)
        print_status(status, as_json=args.json)

    elif args.all:
        experiments = list_all_experiments(args.base_dir)

        if not experiments:
            print("No experiments found.")
            return 0

        if args.json:
            print(json.dumps(experiments, indent=2))
        else:
            print(f"Found {len(experiments)} experiment(s):\n")
            for exp in experiments:
                progress = exp.get("progress", {})
                print(f"  {exp['experiment_id']}: {exp['status']} "
                      f"({progress.get('completed', 0)}/{progress.get('total', 0)})")
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
