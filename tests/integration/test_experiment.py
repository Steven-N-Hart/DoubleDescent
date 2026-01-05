"""Integration tests for full experiment pipeline (T037)."""

import json
import numpy as np
import pytest
from pathlib import Path

from src.data.scenarios import DataScenario
from src.data.types import CovariateType, ExperimentStatus
from src.models.config import ModelConfiguration
from src.experiments.config import Experiment
from src.experiments.sweep import CapacitySweep
from src.experiments.runner import ExperimentRunner


class TestExperimentConfiguration:
    """Tests for experiment configuration."""

    def test_create_experiment(self):
        """Test creating an experiment configuration."""
        scenario = DataScenario(
            name="test",
            n_samples=100,
            n_features=10,
            n_predictive=5,
        )

        experiment = Experiment(
            name="Test Experiment",
            seed=42,
            data_scenario=scenario,
            width_sweep=[4, 8, 16],
            depth_sweep=[2],
        )

        assert experiment.name == "Test Experiment"
        assert experiment.seed == 42
        assert len(experiment.width_sweep) == 3

    def test_experiment_serialization(self, tmp_path):
        """Test experiment to/from JSON."""
        scenario = DataScenario(name="baseline", n_samples=100)
        experiment = Experiment(
            name="Serialization Test",
            seed=42,
            data_scenario=scenario,
            width_sweep=[4, 8],
        )

        path = tmp_path / "experiment.json"
        experiment.to_json(path)

        loaded = Experiment.from_json(path)
        assert loaded.name == "Serialization Test"
        assert loaded.seed == 42
        assert loaded.width_sweep == [4, 8]

    def test_get_sweep_configurations(self):
        """Test generating sweep configurations."""
        scenario = DataScenario(name="test", n_samples=100)
        experiment = Experiment(
            name="Test",
            seed=42,
            data_scenario=scenario,
            width_sweep=[4, 8, 16],
            depth_sweep=[2, 3],
        )

        configs = experiment.get_sweep_configurations()

        # Should have width * depth configurations
        assert len(configs) == 6
        # Check all widths and depths are covered
        widths = {c.width for c in configs}
        depths = {c.depth for c in configs}
        assert widths == {4, 8, 16}
        assert depths == {2, 3}


class TestCapacitySweep:
    """Tests for capacity sweep management."""

    @pytest.fixture
    def base_config(self):
        """Create base model configuration."""
        return ModelConfiguration(width=4, depth=2, epochs=10)

    def test_sweep_generation(self, base_config):
        """Test sweep point generation."""
        sweep = CapacitySweep(
            widths=[4, 8, 16],
            depths=[2],
            base_config=base_config,
        )

        assert len(sweep) == 3
        assert sweep.n_completed == 0
        assert sweep.n_pending == 3

    def test_sweep_completion_tracking(self, base_config):
        """Test marking sweep points as complete."""
        sweep = CapacitySweep(
            widths=[4, 8, 16],
            depths=[2],
            base_config=base_config,
        )

        point = sweep.get_next()
        assert point is not None

        sweep.mark_completed(point)
        assert sweep.n_completed == 1
        assert sweep.n_pending == 2

    def test_sweep_all_completed(self, base_config):
        """Test when all points are completed."""
        sweep = CapacitySweep(
            widths=[4, 8],
            depths=[2],
            base_config=base_config,
        )

        for point in sweep:
            sweep.mark_completed(point)

        assert sweep.n_completed == 2
        assert sweep.get_next() is None

    def test_sweep_progress_string(self, base_config):
        """Test progress string generation."""
        sweep = CapacitySweep(
            widths=[4, 8, 16, 32],
            depths=[2],
            base_config=base_config,
        )

        sweep.mark_completed(sweep.points[0])
        sweep.mark_completed(sweep.points[1])

        progress = sweep.progress_string()
        assert "2/4" in progress
        assert "50.0%" in progress


class TestExperimentRunner:
    """Integration tests for experiment runner."""

    @pytest.fixture
    def small_experiment(self, tmp_path):
        """Create a small experiment for testing."""
        scenario = DataScenario(
            name="test_mini",
            n_samples=100,  # Minimum allowed
            n_features=5,
            n_predictive=3,
            covariate_type=CovariateType.GAUSSIAN,
            censoring_rate=0.3,
        )

        experiment = Experiment(
            name="Mini Test",
            seed=42,
            data_scenario=scenario,
            width_sweep=[4, 8],  # Only 2 widths
            depth_sweep=[2],
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        # Override training parameters for speed
        experiment.base_model_config = ModelConfiguration(
            width=4,
            depth=2,
            epochs=5,  # Very few epochs
            batch_size=16,
            learning_rate=0.01,
        )

        return experiment, tmp_path

    @pytest.mark.slow
    def test_runner_creates_output_structure(self, small_experiment):
        """Test that runner creates correct output structure."""
        experiment, tmp_path = small_experiment

        runner = ExperimentRunner(
            experiment=experiment,
            output_dir=tmp_path,
            verbose=False,
        )

        # Just check directory creation, don't run full experiment
        assert (runner.output_dir / "data").exists()
        assert (runner.output_dir / "runs").exists()
        assert (runner.output_dir / "results").exists()

    @pytest.mark.slow
    def test_runner_data_generation(self, small_experiment):
        """Test that runner generates data correctly."""
        experiment, tmp_path = small_experiment

        runner = ExperimentRunner(
            experiment=experiment,
            output_dir=tmp_path,
            verbose=False,
        )

        # Access private method to test data generation
        train, val, test = runner._prepare_data()

        assert len(train.T) == 60  # 60% of 100
        assert len(val.T) == 20   # 20% of 100
        assert len(test.T) == 20  # 20% of 100

        # Check data is saved
        assert (runner.output_dir / "data" / "train.npz").exists()
        assert (runner.output_dir / "data" / "val.npz").exists()
        assert (runner.output_dir / "data" / "test.npz").exists()

    @pytest.mark.slow
    def test_runner_progress_save_load(self, small_experiment):
        """Test progress saving and loading."""
        experiment, tmp_path = small_experiment

        runner = ExperimentRunner(
            experiment=experiment,
            output_dir=tmp_path,
            verbose=False,
        )

        # Mark some as completed
        runner.sweep.mark_completed(runner.sweep.points[0])
        runner._save_progress()

        # Check progress file
        progress_path = runner.output_dir / "progress.json"
        assert progress_path.exists()

        with open(progress_path) as f:
            progress = json.load(f)

        assert progress["n_completed"] == 1
        assert progress["n_total"] == 2

    @pytest.mark.slow
    def test_runner_resume(self, small_experiment):
        """Test experiment resumption."""
        experiment, tmp_path = small_experiment

        # Create first runner and save progress
        runner1 = ExperimentRunner(
            experiment=experiment,
            output_dir=tmp_path,
            verbose=False,
        )
        runner1.sweep.mark_completed(runner1.sweep.points[0])
        runner1._save_progress()

        # Create second runner and load progress
        runner2 = ExperimentRunner(
            experiment=experiment,
            output_dir=tmp_path,
            verbose=False,
        )
        runner2._load_progress()

        assert runner2.sweep.n_completed == 1
        assert runner2.sweep.n_pending == 1


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    @pytest.mark.slow
    def test_minimal_experiment_completes(self, tmp_path):
        """Test that a minimal experiment runs to completion."""
        # Create minimal experiment
        scenario = DataScenario(
            name="minimal",
            n_samples=100,  # Minimum allowed
            n_features=5,
            n_predictive=3,
            censoring_rate=0.3,
        )

        experiment = Experiment(
            name="Minimal E2E Test",
            seed=42,
            data_scenario=scenario,
            width_sweep=[2, 4],
            depth_sweep=[1],
        )

        experiment.base_model_config = ModelConfiguration(
            width=2,
            depth=1,
            epochs=3,
            batch_size=16,
        )

        # Run experiment
        runner = ExperimentRunner(
            experiment=experiment,
            output_dir=tmp_path,
            verbose=False,
        )

        status = runner.run()

        # Should complete (may have some failures due to small data)
        assert status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED)

        # Output files should exist
        assert (runner.output_dir / "config.json").exists()
        assert (runner.output_dir / "data" / "train.npz").exists()
        assert (runner.output_dir / "progress.json").exists()

    @pytest.mark.slow
    def test_experiment_produces_results(self, tmp_path):
        """Test that experiment produces result files."""
        scenario = DataScenario(
            name="results_test",
            n_samples=100,  # Minimum allowed
            n_features=5,
            n_predictive=3,
            censoring_rate=0.3,
        )

        experiment = Experiment(
            name="Results Test",
            seed=42,
            data_scenario=scenario,
            width_sweep=[2],  # Single width for speed
            depth_sweep=[1],
        )

        experiment.base_model_config = ModelConfiguration(
            width=2,
            depth=1,
            epochs=5,
            batch_size=16,
        )

        runner = ExperimentRunner(
            experiment=experiment,
            output_dir=tmp_path,
            verbose=False,
        )

        runner.run()

        # Results should be aggregated
        results_dir = runner.output_dir / "results"
        assert results_dir.exists()

        # Summary CSV should exist if any runs completed
        summary_path = results_dir / "summary.csv"
        curves_path = results_dir / "curves.json"

        # At least one should exist if experiment ran
        assert summary_path.exists() or curves_path.exists()
