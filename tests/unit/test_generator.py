"""Unit tests for data generation (T034)."""

import numpy as np
import pytest

from src.data.types import CovariateType
from src.data.scenarios import DataScenario, get_scenario, PREDEFINED_SCENARIOS
from src.data.generator import SurvivalDataGenerator, SurvivalData, DataSplitter
from src.data.copula import generate_correlated_uniform, create_ar1_correlation
from src.data.censoring import calibrate_censoring_rate


class TestDataScenario:
    """Tests for DataScenario configuration."""

    def test_default_scenario_creation(self):
        """Test creating a scenario with default values."""
        scenario = DataScenario(name="test")
        assert scenario.n_samples == 1000
        assert scenario.n_features == 20
        assert scenario.n_predictive == 10
        assert scenario.censoring_rate == 0.3

    def test_validation_n_predictive_exceeds_n_features(self):
        """Test that n_predictive > n_features raises error."""
        with pytest.raises(ValueError, match="n_predictive.*cannot exceed"):
            DataScenario(
                name="test",
                n_features=10,
                n_predictive=20,
            )

    def test_validation_invalid_censoring_rate(self):
        """Test that invalid censoring rate raises error."""
        with pytest.raises(ValueError, match="censoring_rate"):
            DataScenario(name="test", censoring_rate=1.5)

    def test_predefined_scenarios_exist(self):
        """Test that all predefined scenarios are available."""
        expected = ["baseline", "skewed", "high_cardinality", "imbalanced"]
        for name in expected:
            scenario = get_scenario(name)
            assert scenario.name == name

    def test_scenario_serialization(self, tmp_path):
        """Test scenario to/from JSON."""
        scenario = DataScenario(
            name="test_scenario",
            n_samples=500,
            n_features=15,
            censoring_rate=0.4,
        )
        path = tmp_path / "scenario.json"
        scenario.to_json(path)

        loaded = DataScenario.from_json(path)
        assert loaded.name == "test_scenario"
        assert loaded.n_samples == 500
        assert loaded.n_features == 15


class TestSurvivalDataGenerator:
    """Tests for synthetic data generation."""

    @pytest.fixture
    def baseline_scenario(self):
        """Create a baseline scenario for testing."""
        return DataScenario(
            name="test_baseline",
            n_samples=200,
            n_features=10,
            n_predictive=5,
            covariate_type=CovariateType.GAUSSIAN,
            censoring_rate=0.3,
        )

    def test_generate_correct_shapes(self, baseline_scenario):
        """Test that generated data has correct shapes."""
        generator = SurvivalDataGenerator(baseline_scenario, seed=42)
        data = generator.generate()

        assert data.X.shape == (200, 10)
        assert data.T.shape == (200,)
        assert data.E.shape == (200,)
        assert data.T_true.shape == (200,)
        assert data.beta.shape == (10,)

    def test_generate_sparse_coefficients(self, baseline_scenario):
        """Test that only first n_predictive coefficients are non-zero."""
        generator = SurvivalDataGenerator(baseline_scenario, seed=42)
        data = generator.generate()

        # First n_predictive should be non-zero
        assert np.all(data.beta[:5] != 0)
        # Rest should be zero
        assert np.all(data.beta[5:] == 0)

    def test_generate_positive_times(self, baseline_scenario):
        """Test that all survival times are positive."""
        generator = SurvivalDataGenerator(baseline_scenario, seed=42)
        data = generator.generate()

        assert np.all(data.T > 0)
        assert np.all(data.T_true > 0)

    def test_generate_binary_events(self, baseline_scenario):
        """Test that event indicators are binary."""
        generator = SurvivalDataGenerator(baseline_scenario, seed=42)
        data = generator.generate()

        assert set(np.unique(data.E)).issubset({0.0, 1.0})

    def test_censoring_rate_approximate(self, baseline_scenario):
        """Test that censoring rate is approximately correct."""
        generator = SurvivalDataGenerator(baseline_scenario, seed=42)
        data = generator.generate()

        actual_rate = 1 - np.mean(data.E)
        # Allow some tolerance due to randomness
        assert abs(actual_rate - 0.3) < 0.1

    def test_reproducibility_with_seed(self, baseline_scenario):
        """Test that same seed produces same data."""
        gen1 = SurvivalDataGenerator(baseline_scenario, seed=42)
        gen2 = SurvivalDataGenerator(baseline_scenario, seed=42)

        data1 = gen1.generate()
        data2 = gen2.generate()

        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.T, data2.T)
        np.testing.assert_array_equal(data1.E, data2.E)

    def test_different_seeds_produce_different_data(self, baseline_scenario):
        """Test that different seeds produce different data."""
        gen1 = SurvivalDataGenerator(baseline_scenario, seed=42)
        gen2 = SurvivalDataGenerator(baseline_scenario, seed=123)

        data1 = gen1.generate()
        data2 = gen2.generate()

        assert not np.allclose(data1.X, data2.X)

    def test_lognormal_covariates(self):
        """Test log-normal covariate generation."""
        scenario = DataScenario(
            name="lognormal_test",
            n_samples=500,
            n_features=10,
            n_predictive=5,
            covariate_type=CovariateType.LOGNORMAL,
        )
        generator = SurvivalDataGenerator(scenario, seed=42)
        data = generator.generate()

        # Log-normal should be positive
        assert np.all(data.X >= 0)
        # Should have right-skewed distribution
        assert np.mean(data.X) > np.median(data.X)


class TestDataSplitter:
    """Tests for data splitting."""

    @pytest.fixture
    def sample_data(self):
        """Create sample survival data."""
        n = 100
        return SurvivalData(
            X=np.random.randn(n, 5).astype(np.float32),
            T=np.random.exponential(1, n).astype(np.float32),
            E=np.random.binomial(1, 0.7, n).astype(np.float32),
            T_true=np.random.exponential(1, n).astype(np.float32),
            beta=np.array([1.0, -0.5, 0.3, 0, 0]),
        )

    def test_split_sizes(self, sample_data):
        """Test that splits have correct sizes."""
        splitter = DataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
        train, val, test = splitter.split(sample_data)

        assert len(train.T) == 60
        assert len(val.T) == 20
        assert len(test.T) == 20

    def test_split_no_overlap(self, sample_data):
        """Test that splits don't overlap."""
        splitter = DataSplitter(seed=42)
        train, val, test = splitter.split(sample_data)

        # Check that no sample appears in multiple splits
        all_samples = np.vstack([train.X, val.X, test.X])
        assert len(all_samples) == len(sample_data.X)

    def test_split_reproducibility(self, sample_data):
        """Test that same seed produces same splits."""
        splitter1 = DataSplitter(seed=42)
        splitter2 = DataSplitter(seed=42)

        train1, _, _ = splitter1.split(sample_data)
        train2, _, _ = splitter2.split(sample_data)

        np.testing.assert_array_equal(train1.X, train2.X)

    def test_invalid_ratios(self):
        """Test that invalid ratios raise error."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            DataSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


class TestCopula:
    """Tests for Gaussian copula."""

    def test_uncorrelated_uniform(self):
        """Test generating uncorrelated uniform samples."""
        samples = generate_correlated_uniform(1000, 5, rng=np.random.default_rng(42))

        # Should be uniform [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)
        # Mean should be approximately 0.5
        assert abs(np.mean(samples) - 0.5) < 0.05

    def test_ar1_correlation_structure(self):
        """Test AR(1) correlation matrix creation."""
        corr = create_ar1_correlation(5, 0.5)

        assert corr.shape == (5, 5)
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(5))
        # Should be symmetric
        np.testing.assert_array_almost_equal(corr, corr.T)


class TestCensoring:
    """Tests for censoring calibration."""

    def test_calibration_achieves_target(self):
        """Test that calibration achieves approximate target rate."""
        event_times = np.random.exponential(1.0, 500)
        target_rate = 0.3
        rng = np.random.default_rng(42)

        scale = calibrate_censoring_rate(event_times, target_rate, rng)

        # Simulate censoring with calibrated scale
        censoring_times = rng.exponential(scale, len(event_times))
        actual_rate = np.mean(event_times > censoring_times)

        assert abs(actual_rate - target_rate) < 0.1

    def test_zero_censoring_rate(self):
        """Test that zero censoring rate returns infinity."""
        event_times = np.random.exponential(1.0, 100)
        scale = calibrate_censoring_rate(event_times, 0.0, np.random.default_rng(42))
        assert scale == np.inf
