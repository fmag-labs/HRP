"""Tests for HMM regime detection."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hrp.ml.regime import (
    MarketRegime,
    HMMConfig,
    RegimeResult,
    RegimeDetector,
)


class TestMarketRegime:
    """Tests for MarketRegime enum."""

    def test_all_regimes_defined(self):
        """Test all expected regimes are defined."""
        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.CRISIS.value == "crisis"


class TestHMMConfig:
    """Tests for HMMConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = HMMConfig()
        assert config.n_regimes == 3
        assert config.features == ["returns_20d", "volatility_20d"]
        assert config.covariance_type == "full"
        assert config.n_iter == 100
        assert config.random_state == 42
        assert config.tol == 1e-4

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = HMMConfig(
            n_regimes=4,
            features=["returns_20d"],
            covariance_type="diag",
            n_iter=200,
            random_state=123,
        )
        assert config.n_regimes == 4
        assert config.features == ["returns_20d"]
        assert config.covariance_type == "diag"
        assert config.n_iter == 200
        assert config.random_state == 123

    def test_config_invalid_n_regimes(self):
        """Test config rejects n_regimes < 2."""
        with pytest.raises(ValueError, match="n_regimes must be >= 2"):
            HMMConfig(n_regimes=1)

    def test_config_invalid_covariance_type(self):
        """Test config rejects invalid covariance type."""
        with pytest.raises(ValueError, match="Invalid covariance_type"):
            HMMConfig(covariance_type="invalid")

    def test_config_invalid_n_iter(self):
        """Test config rejects n_iter < 1."""
        with pytest.raises(ValueError, match="n_iter must be >= 1"):
            HMMConfig(n_iter=0)


class TestRegimeDetector:
    """Tests for RegimeDetector class."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data with clear regime patterns."""
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", "2023-12-31", freq="B")
        n = len(dates)

        # Create regime-like patterns
        prices = [100.0]
        for i in range(1, n):
            # Simulate different regimes
            if i < n // 3:
                # Bull market
                change = np.random.normal(0.001, 0.01)
            elif i < 2 * n // 3:
                # Bear market
                change = np.random.normal(-0.0005, 0.02)
            else:
                # Sideways
                change = np.random.normal(0.0, 0.015)
            prices.append(prices[-1] * (1 + change))

        return pd.DataFrame(
            {"close": prices},
            index=dates,
        )

    @pytest.fixture
    def detector(self):
        """Create a RegimeDetector instance."""
        config = HMMConfig(n_regimes=3)
        return RegimeDetector(config)

    def test_fit_returns_self(self, detector, sample_prices):
        """Test fit() returns self for method chaining."""
        pytest.importorskip("hmmlearn")

        result = detector.fit(sample_prices)
        assert result is detector
        assert detector.is_fitted is True

    def test_predict_returns_series(self, detector, sample_prices):
        """Test predict() returns a pandas Series."""
        pytest.importorskip("hmmlearn")

        detector.fit(sample_prices)
        regimes = detector.predict(sample_prices)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_prices)
        assert regimes.name == "regime"

    def test_predict_before_fit_raises(self, detector, sample_prices):
        """Test predict() raises error if not fitted."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(sample_prices)

    def test_predict_values_in_range(self, detector, sample_prices):
        """Test predicted regimes are valid indices."""
        pytest.importorskip("hmmlearn")

        detector.fit(sample_prices)
        regimes = detector.predict(sample_prices)

        # Remove NaN values
        valid_regimes = regimes.dropna()
        unique_regimes = valid_regimes.unique()

        for regime in unique_regimes:
            assert 0 <= regime < detector.config.n_regimes

    def test_regime_mapping_correct(self, detector, sample_prices):
        """Test regime labels are assigned based on characteristics."""
        pytest.importorskip("hmmlearn")

        detector.fit(sample_prices)
        result = detector.get_regime_statistics(sample_prices)

        # Check all regimes have labels
        assert len(result.regime_labels) == detector.config.n_regimes

        # Check labels are valid MarketRegime values
        for label in result.regime_labels.values():
            assert isinstance(label, MarketRegime)

    def test_transition_matrix_probabilities(self, detector, sample_prices):
        """Test transition matrix has valid probabilities."""
        pytest.importorskip("hmmlearn")

        detector.fit(sample_prices)
        trans_mat = detector.get_transition_matrix()

        assert isinstance(trans_mat, pd.DataFrame)
        assert trans_mat.shape == (3, 3)

        # Each row should sum to 1 (approximately)
        row_sums = trans_mat.sum(axis=1)
        for row_sum in row_sums:
            assert abs(row_sum - 1.0) < 1e-6

        # All values should be between 0 and 1
        assert (trans_mat.values >= 0).all()
        assert (trans_mat.values <= 1).all()

    def test_get_transition_matrix_before_fit_raises(self, detector):
        """Test get_transition_matrix() raises error if not fitted."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.get_transition_matrix()

    def test_get_regime_statistics_returns_result(self, detector, sample_prices):
        """Test get_regime_statistics returns RegimeResult."""
        pytest.importorskip("hmmlearn")

        detector.fit(sample_prices)
        result = detector.get_regime_statistics(sample_prices)

        assert isinstance(result, RegimeResult)
        assert isinstance(result.regimes, pd.Series)
        assert isinstance(result.transition_matrix, np.ndarray)
        assert isinstance(result.regime_means, dict)
        assert isinstance(result.regime_covariances, dict)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.regime_durations, dict)

    def test_regime_durations_positive(self, detector, sample_prices):
        """Test regime durations are positive."""
        pytest.importorskip("hmmlearn")

        detector.fit(sample_prices)
        result = detector.get_regime_statistics(sample_prices)

        for duration in result.regime_durations.values():
            assert duration > 0

    def test_regime_means_contain_features(self, detector, sample_prices):
        """Test regime means contain expected features."""
        pytest.importorskip("hmmlearn")

        detector.fit(sample_prices)
        result = detector.get_regime_statistics(sample_prices)

        for regime_idx in range(detector.config.n_regimes):
            assert regime_idx in result.regime_means
            for feature in detector.config.features:
                assert feature in result.regime_means[regime_idx]

    def test_is_fitted_property(self, detector, sample_prices):
        """Test is_fitted property."""
        assert detector.is_fitted is False

        pytest.importorskip("hmmlearn")
        detector.fit(sample_prices)

        assert detector.is_fitted is True

    def test_different_covariance_types(self, sample_prices):
        """Test different covariance types work."""
        pytest.importorskip("hmmlearn")

        for cov_type in ["full", "diag", "spherical", "tied"]:
            config = HMMConfig(n_regimes=2, covariance_type=cov_type)
            detector = RegimeDetector(config)
            detector.fit(sample_prices)
            result = detector.get_regime_statistics(sample_prices)

            assert len(result.regime_covariances) == 2

    def test_insufficient_data_raises(self):
        """Test fitting with insufficient data raises error."""
        pytest.importorskip("hmmlearn")

        # Create very short price series
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")
        prices = pd.DataFrame({"close": np.random.randn(len(dates))}, index=dates)

        config = HMMConfig(n_regimes=3)
        detector = RegimeDetector(config)

        with pytest.raises(ValueError, match="Insufficient data"):
            detector.fit(prices)


class TestRegimeResult:
    """Tests for RegimeResult dataclass."""

    def test_result_creation(self):
        """Test creating RegimeResult."""
        regimes = pd.Series([0, 1, 2, 1, 0])
        trans_mat = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]])

        result = RegimeResult(
            regimes=regimes,
            transition_matrix=trans_mat,
            regime_means={0: {"returns_20d": 0.01}, 1: {"returns_20d": -0.01}},
            regime_covariances={0: np.eye(1), 1: np.eye(1)},
            log_likelihood=-100.0,
            regime_durations={0: 10.0, 1: 5.0},
            regime_labels={0: MarketRegime.BULL, 1: MarketRegime.BEAR},
        )

        assert len(result.regimes) == 5
        assert result.log_likelihood == -100.0
        assert result.regime_labels[0] == MarketRegime.BULL


class TestModuleExports:
    """Test that regime module is properly exported."""

    def test_import_classes(self):
        """Test importing classes from hrp.ml.regime."""
        from hrp.ml.regime import (
            MarketRegime,
            HMMConfig,
            RegimeResult,
            RegimeDetector,
        )

        assert MarketRegime is not None
        assert HMMConfig is not None
        assert RegimeResult is not None
        assert RegimeDetector is not None
