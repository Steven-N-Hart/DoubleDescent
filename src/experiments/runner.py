"""Experiment runner for orchestrating training sweeps."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .config import Experiment
from .run import ExperimentRun
from .sweep import CapacitySweep, SweepPoint
from .logging import ExperimentLogger
from ..data.generator import SurvivalDataGenerator, SurvivalData, DataSplitter
from ..data.types import ExperimentStatus, RunStatus
from ..models.deepsurv import DeepSurv
from ..models.trainer import train_with_retry
from ..models.checkpoint import CheckpointManager
from ..metrics.results import MetricResult, RunSummary
from ..visualization.curves import DoubleDescentCurve


class ExperimentRunner:
    """Orchestrates experiment execution.

    Handles:
    - Data generation and splitting
    - Sequential model training across sweep points
    - Progress tracking and resumption
    - Results aggregation

    Args:
        experiment: Experiment configuration.
        output_dir: Output directory for results.
        device: Device to train on.
        verbose: Whether to print progress.
    """

    def __init__(
        self,
        experiment: Experiment,
        output_dir: Union[str, Path],
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        self.experiment = experiment
        self.output_dir = Path(output_dir) / experiment.experiment_id
        self.verbose = verbose

        # Setup device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Create output directories
        self._setup_directories()

        # Initialize sweep
        self.sweep = CapacitySweep(
            widths=experiment.width_sweep,
            depths=experiment.depth_sweep,
            base_config=experiment.base_model_config,
            sweep_type=experiment.sweep_type,
        )

        # Results storage
        self.run_results: Dict[str, RunSummary] = {}
        self.all_metrics: List[MetricResult] = []

    def _setup_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "runs").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "tensorboard").mkdir(exist_ok=True)

    def run(self, resume: bool = False) -> ExperimentStatus:
        """Execute the full experiment.

        Args:
            resume: If True, resume from previous progress.

        Returns:
            Final experiment status.
        """
        try:
            # Save experiment config
            self.experiment.to_json(self.output_dir / "config.json")

            # Load progress if resuming
            if resume:
                self._load_progress()

            # Generate or load data
            train_data, val_data, test_data = self._prepare_data()

            # Start experiment
            self.experiment.start()
            self._log(f"Starting experiment: {self.experiment.name}")
            self._log(f"Device: {self.device}")
            self._log(f"Sweep: {len(self.sweep)} configurations")

            # Run sweep
            n_failed = 0
            while True:
                point = self.sweep.get_next()
                if point is None:
                    break

                self._log(f"\n[{self.sweep.progress_string()}] "
                          f"Training width={point.width}, depth={point.depth}")

                # Train this configuration
                success = self._train_point(
                    point, train_data, val_data, test_data
                )

                if not success:
                    n_failed += 1

                # Save progress after each run
                self._save_progress()

            # Aggregate results
            self._aggregate_results()

            # Determine final status
            if n_failed == 0:
                self.experiment.complete()
                self._log("\nExperiment completed successfully!")
            elif n_failed < len(self.sweep):
                self.experiment.status = ExperimentStatus.COMPLETED
                self._log(f"\nExperiment completed with {n_failed} failed runs")
            else:
                self.experiment.fail()
                self._log("\nExperiment failed: all runs failed")

            # Save final config
            self.experiment.to_json(self.output_dir / "config.json")

            return self.experiment.status

        except Exception as e:
            self.experiment.fail()
            self._log(f"\nExperiment failed with error: {e}")
            return ExperimentStatus.FAILED

    def _prepare_data(self) -> tuple:
        """Generate or load experiment data.

        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        train_path = self.output_dir / "data" / "train.npz"
        val_path = self.output_dir / "data" / "val.npz"
        test_path = self.output_dir / "data" / "test.npz"

        # Check if data already exists
        if all(p.exists() for p in [train_path, val_path, test_path]):
            self._log("Loading existing data...")
            train_data = SurvivalData.load(train_path)
            val_data = SurvivalData.load(val_path)
            test_data = SurvivalData.load(test_path)
            return train_data, val_data, test_data

        # Generate new data
        self._log("Generating synthetic data...")
        generator = SurvivalDataGenerator(
            self.experiment.data_scenario,
            seed=self.experiment.seed,
        )
        full_data = generator.generate()

        # Split data
        splitter = DataSplitter(
            train_ratio=self.experiment.train_ratio,
            val_ratio=self.experiment.val_ratio,
            test_ratio=self.experiment.test_ratio,
            seed=self.experiment.seed,
        )
        train_data, val_data, test_data = splitter.split(full_data)

        # Save data
        train_data.save(train_path)
        val_data.save(val_path)
        test_data.save(test_path)

        # Save ground truth
        ground_truth = {
            "beta": full_data.beta.tolist(),
            "n_samples": self.experiment.data_scenario.n_samples,
            "n_features": self.experiment.data_scenario.n_features,
            "n_predictive": self.experiment.data_scenario.n_predictive,
            "censoring_rate": float(1 - np.mean(train_data.E)),
        }
        with open(self.output_dir / "data" / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)

        self._log(f"  Train: {len(train_data.T)} samples, "
                  f"{np.mean(train_data.E):.1%} events")
        self._log(f"  Val: {len(val_data.T)} samples")
        self._log(f"  Test: {len(test_data.T)} samples")

        return train_data, val_data, test_data

    def _train_point(
        self,
        point: SweepPoint,
        train_data: SurvivalData,
        val_data: SurvivalData,
        test_data: SurvivalData,
    ) -> bool:
        """Train a single sweep point.

        Args:
            point: Sweep point to train.
            train_data: Training data.
            val_data: Validation data.
            test_data: Test data.

        Returns:
            True if training succeeded.
        """
        run_id = point.run_id
        n_features = train_data.X.shape[1]

        # Create logger
        logger = ExperimentLogger(
            experiment_dir=self.output_dir,
            run_id=run_id,
            width=point.width,
            depth=point.depth,
        )

        # Create experiment run record
        run = ExperimentRun.create(
            experiment_id=self.experiment.experiment_id,
            width=point.width,
            depth=point.depth,
            model_config=point.config,
        )
        run.start()

        try:
            # Train with retry
            state, metrics, model = train_with_retry(
                n_features=n_features,
                config=point.config,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                device=self.device,
                run_id=run_id,
                log_callback=logger.log_metrics,
            )

            self.all_metrics.extend(metrics)

            if state.failed:
                run.fail(state.failure_reason or "Unknown failure")
                self._log(f"  FAILED: {state.failure_reason}")
                success = False
            else:
                run.complete(epochs=state.epoch, best_epoch=state.best_epoch)
                ibs_str = f"{state.best_val_metric:.4f}" if state.best_val_metric < float("inf") else "N/A"
                self._log(f"  Completed: {state.epoch} epochs, "
                          f"best val IBS={ibs_str}")
                success = True

                # Save best model
                if model is not None:
                    checkpoint_dir = self.output_dir / "runs" / run_id / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint_dir / "best.pt")

        except Exception as e:
            run.fail(str(e))
            self._log(f"  ERROR: {e}")
            success = False

        finally:
            logger.close()

        # Save run info
        run_info_path = self.output_dir / "runs" / run_id / "run_info.json"
        with open(run_info_path, "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

        # Mark as completed (even if failed)
        self.sweep.mark_completed(point)

        return success

    def _save_progress(self) -> None:
        """Save current progress for resumption."""
        progress = {
            "completed_runs": list(self.sweep.completed),
            "n_completed": self.sweep.n_completed,
            "n_total": len(self.sweep),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.output_dir / "progress.json", "w") as f:
            json.dump(progress, f, indent=2)

    def _load_progress(self) -> None:
        """Load progress from previous run."""
        progress_path = self.output_dir / "progress.json"
        if not progress_path.exists():
            return

        with open(progress_path, "r") as f:
            progress = json.load(f)

        for run_id in progress.get("completed_runs", []):
            self.sweep.mark_completed_by_id(run_id)

        self._log(f"Resuming: {self.sweep.n_completed}/{len(self.sweep)} already completed")

    def _aggregate_results(self) -> None:
        """Aggregate results and create summary files."""
        results_dir = self.output_dir / "results"

        # Create summary CSV
        summary_rows = []
        for point in self.sweep:
            run_dir = self.output_dir / "runs" / point.run_id
            metrics_path = run_dir / "metrics.csv"

            if metrics_path.exists():
                import csv
                with open(metrics_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                if rows:
                    # Get final test metrics
                    test_rows = [r for r in rows if r["split"] == "test"]
                    if test_rows:
                        final = test_rows[-1]
                        summary_rows.append({
                            "width": point.width,
                            "depth": point.depth,
                            "n_parameters": point.config.n_parameters(
                                self.experiment.data_scenario.n_features
                            ),
                            "c_index": final.get("c_index", ""),
                            "ibs": final.get("ibs", ""),
                            "nll": final.get("nll", ""),
                            "cal_slope": final.get("cal_slope", ""),
                            "cal_large": final.get("cal_large", ""),
                            "ici": final.get("ici", ""),
                        })

        # Write summary
        if summary_rows:
            import csv
            with open(results_dir / "summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["width", "depth", "n_parameters", "c_index", "ibs", "nll",
                                "cal_slope", "cal_large", "ici"]
                )
                writer.writeheader()
                writer.writerows(summary_rows)

        # Create double descent curves
        curves = {}
        for metric in ["c_index", "ibs", "nll"]:
            curve = DoubleDescentCurve(
                experiment_id=self.experiment.experiment_id,
                metric_name=metric,
                split="test",
            )
            for row in summary_rows:
                curve.capacities.append(int(row["width"]))
                value = row.get(metric, "")
                curve.values.append(float(value) if value else np.nan)

            curve.analyze(
                int(self.experiment.data_scenario.n_samples * self.experiment.train_ratio)
            )
            curves[metric] = curve.to_dict()

        with open(results_dir / "curves.json", "w") as f:
            json.dump(curves, f, indent=2)

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message, file=sys.stderr)


def run_experiment(
    config_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs/experiments",
    device: str = "auto",
    resume: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """Run an experiment from a config file.

    Args:
        config_path: Path to experiment config JSON.
        output_dir: Base output directory.
        device: Device string ("cuda", "cpu", or "auto").
        resume: Whether to resume from checkpoint.
        dry_run: If True, validate config without running.
        verbose: Whether to print progress.

    Returns:
        Exit code (0=success, 1=config error, 2=runtime error, 3=partial).
    """
    # Load config
    try:
        experiment = Experiment.from_json(config_path)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        return 1

    if dry_run:
        print(f"Config validation successful: {experiment.name}")
        print(f"  Experiment ID: {experiment.experiment_id}")
        print(f"  Sweep points: {len(experiment.width_sweep) * len(experiment.depth_sweep)}")
        return 0

    # Setup device
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    # Create runner
    runner = ExperimentRunner(
        experiment=experiment,
        output_dir=output_dir,
        device=torch_device,
        verbose=verbose,
    )

    # Run experiment
    status = runner.run(resume=resume)

    # Map status to exit code
    if status == ExperimentStatus.COMPLETED:
        return 0
    elif status == ExperimentStatus.FAILED:
        return 2
    else:
        return 3
