"""CLI for running experiments."""

import argparse
import sys
from pathlib import Path

from ..experiments.runner import run_experiment


def main() -> int:
    """Main entry point for run_experiment CLI.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.run_experiment",
        description="Execute a complete experiment with capacity sweep.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment JSON config",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Output directory (default: outputs/experiments/)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device: cuda, cpu, auto (default: auto)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    args = parser.parse_args()

    # Check config exists
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1

    verbose = args.verbose and not args.quiet

    return run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
        dry_run=args.dry_run,
        verbose=verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
