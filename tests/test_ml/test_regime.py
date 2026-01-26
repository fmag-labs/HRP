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


class TestPrepareFeatures:
    """Tests for RegimeDetector._prepare_features method."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
        n = len(dates)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
        return pd.DataFrame({"close": prices}, index=dates)

    def test_prepare_features_returns_correct_shape(self, sample_prices):
        """Test _prepare_features returns correct array shape."""
        config = HMMConfig(n_regimes=2, features=["returns_20d", "volatility_20d"])
        detector = RegimeDetector(config)

        X = detector._prepare_features(sample_prices, fit=True)

        assert X.ndim == 2
        assert X.shape[0] == len(sample_prices)
        assert X.shape[1] == 2

    def test_prepare_features_single_feature(self, sample_prices):
        """Test _prepare_features with single feature."""
        config = HMMConfig(n_regimes=2, features=["returns_20d"])
        detector = RegimeDetector(config)

        X = detector._prepare_features(sample_prices, fit=True)

        assert X.shape[1] == 1

    def test_prepare_features_custom_column_in_df(self, sample_prices):
        """Test _prepare_features with custom feature in DataFrame."""
        sample_prices["custom_feature"] = np.random.randn(len(sample_prices))
        config = HMMConfig(n_regimes=2, features=["custom_feature"])
        detector = RegimeDetector(config)

        X = detector._prepare_features(sample_prices, fit=True)

        assert X.shape[1] == 1
        assert not np.isnan(X).all()

    def test_prepare_features_unknown_feature_raises(self, sample_prices):
        """Test _prepare_features raises for unknown feature."""
        config = HMMConfig(n_regimes=2, features=["nonexistent_feature"])
        detector = RegimeDetector(config)

        with pytest.raises(ValueError, match="Unknown feature"):
            detector._prepare_features(sample_prices, fit=True)

    def test_prepare_features_no_close_column_fallback(self):
        """Test _prepare_features uses first column if no 'close'."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
        n = len(dates)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
        df = pd.DataFrame({"price": prices}, index=dates)

        config = HMMConfig(n_regimes=2, features=["returns_20d"])
        detector = RegimeDetector(config)

        X = detector._prepare_features(df, fit=True)

        assert X.shape[1] == 1
        assert not np.isnan(X[25:]).all()  # After warmup period

    def test_prepare_features_normalizes_data(self, sample_prices):
        """Test _prepare_features normalizes features when fit=True."""
        config = HMMConfig(n_regimes=2, features=["returns_20d"])
        detector = RegimeDetector(config)

        X = detector._prepare_features(sample_prices, fit=True)

        # After normalization, mean should be ~0 and std ~1
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        assert abs(np.mean(X_valid)) < 0.1
        assert abs(np.std(X_valid) - 1.0) < 0.1

    def test_prepare_features_uses_stored_params(self, sample_prices):
        """Test _prepare_features uses stored normalization params when fit=False."""
        config = HMMConfig(n_regimes=2, features=["returns_20d"])
        detector = RegimeDetector(config)

        # First call with fit=True stores params
        detector._prepare_features(sample_prices, fit=True)

        # Second call with fit=False should use stored params
        X = detector._prepare_features(sample_prices, fit=False)

        assert detector._feature_means is not None
        assert detector._feature_stds is not None


class TestLabelRegimes:
    """Tests for RegimeDetector._label_regimes method."""

    def test_label_regimes_2_regimes(self):
        """Test _label_regimes with 2 regimes."""
        config = HMMConfig(n_regimes=2, features=["returns_20d", "volatility_20d"])
        detector = RegimeDetector(config)

        regime_means = {
            0: {"returns_20d": -0.05, "volatility_20d": 0.25},
            1: {"returns_20d": 0.10, "volatility_20d": 0.15},
        }

        labels = detector._label_regimes(regime_means)

        assert labels[0] == MarketRegime.BEAR
        assert labels[1] == MarketRegime.BULL

    def test_label_regimes_3_regimes(self):
        """Test _label_regimes with 3 regimes."""
        config = HMMConfig(n_regimes=3, features=["returns_20d", "volatility_20d"])
        detector = RegimeDetector(config)

        regime_means = {
            0: {"returns_20d": -0.10, "volatility_20d": 0.30},
            1: {"returns_20d": 0.00, "volatility_20d": 0.15},
            2: {"returns_20d": 0.15, "volatility_20d": 0.10},
        }

        labels = detector._label_regimes(regime_means)

        assert labels[0] == MarketRegime.BEAR
        assert labels[1] == MarketRegime.SIDEWAYS
        assert labels[2] == MarketRegime.BULL

    def test_label_regimes_4_regimes_with_crisis(self):
        """Test _label_regimes with 4 regimes identifies CRISIS."""
        config = HMMConfig(n_regimes=4, features=["returns_20d", "volatility_20d"])
        detector = RegimeDetector(config)

        regime_means = {
            0: {"returns_20d": -0.25, "volatility_20d": 0.50},  # Lowest return + high vol = CRISIS
            1: {"returns_20d": -0.05, "volatility_20d": 0.20},
            2: {"returns_20d": 0.02, "volatility_20d": 0.12},
            3: {"returns_20d": 0.15, "volatility_20d": 0.10},
        }

        labels = detector._label_regimes(regime_means)

        assert labels[0] == MarketRegime.CRISIS
        assert labels[3] == MarketRegime.BULL

    def test_label_regimes_4_regimes_without_crisis(self):
        """Test _label_regimes with 4 regimes without CRISIS condition."""
        config = HMMConfig(n_regimes=4, features=["returns_20d", "volatility_20d"])
        detector = RegimeDetector(config)

        # Lowest return has LOW vol (below median), so it's BEAR not CRISIS
        regime_means = {
            0: {"returns_20d": -0.10, "volatility_20d": 0.10},  # Low vol
            1: {"returns_20d": -0.02, "volatility_20d": 0.20},
            2: {"returns_20d": 0.02, "volatility_20d": 0.25},
            3: {"returns_20d": 0.15, "volatility_20d": 0.30},  # Higher vol
        }

        labels = detector._label_regimes(regime_means)

        assert labels[0] == MarketRegime.BEAR
        assert labels[3] == MarketRegime.BULL

    def test_label_regimes_no_return_feature(self):
        """Test _label_regimes defaults to SIDEWAYS when no return feature."""
        config = HMMConfig(n_regimes=2, features=["volatility_20d"])
        detector = RegimeDetector(config)

        regime_means = {
            0: {"volatility_20d": 0.25},
            1: {"volatility_20d": 0.15},
        }

        labels = detector._label_regimes(regime_means)

        assert labels[0] == MarketRegime.SIDEWAYS
        assert labels[1] == MarketRegime.SIDEWAYS


class TestFitEdgeCases:
    """Additional tests for RegimeDetector.fit edge cases."""

    def test_fit_hmmlearn_import_error(self):
        """Test fit raises ImportError with helpful message when hmmlearn missing."""
        import sys
        from unittest.mock import patch

        config = HMMConfig(n_regimes=2)
        detector = RegimeDetector(config)

        dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
        prices = pd.DataFrame({"close": np.random.randn(len(dates)).cumsum() + 100}, index=dates)

        # Mock the import to fail
        with patch.dict(sys.modules, {"hmmlearn": None, "hmmlearn.hmm": None}):
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "hmmlearn.hmm" or name == "hmmlearn":
                    raise ImportError("No module named 'hmmlearn'")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = mock_import
            try:
                with pytest.raises(ImportError, match="hmmlearn is required"):
                    detector.fit(prices)
            finally:
                builtins.__import__ = original_import


class TestPredictEdgeCases:
    """Additional tests for RegimeDetector.predict edge cases."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", "2023-12-31", freq="B")
        n = len(dates)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
        return pd.DataFrame({"close": prices}, index=dates)

    def test_predict_with_nan_in_data(self, sample_prices):
        """Test predict handles NaN values in price data."""
        pytest.importorskip("hmmlearn")

        # Add some NaN values
        sample_prices_with_nan = sample_prices.copy()
        sample_prices_with_nan.loc[sample_prices_with_nan.index[100:110], "close"] = np.nan

        config = HMMConfig(n_regimes=2)
        detector = RegimeDetector(config)

        # Fit on clean data
        detector.fit(sample_prices)

        # Predict on data with NaN
        regimes = detector.predict(sample_prices_with_nan)

        assert len(regimes) == len(sample_prices_with_nan)

    def test_predict_preserves_index(self, sample_prices):
        """Test predict preserves the original DataFrame index."""
        pytest.importorskip("hmmlearn")

        config = HMMConfig(n_regimes=2)
        detector = RegimeDetector(config)
        detector.fit(sample_prices)

        regimes = detector.predict(sample_prices)

        assert regimes.index.equals(sample_prices.index)


class TestGetRegimeStatisticsEdgeCases:
    """Additional tests for get_regime_statistics edge cases."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", "2023-12-31", freq="B")
        n = len(dates)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
        return pd.DataFrame({"close": prices}, index=dates)

    def test_get_regime_statistics_before_fit_raises(self, sample_prices):
        """Test get_regime_statistics raises error if not fitted."""
        config = HMMConfig(n_regimes=2)
        detector = RegimeDetector(config)

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.get_regime_statistics(sample_prices)

    def test_regime_durations_infinite_for_absorbing_state(self, sample_prices):
        """Test regime durations handles absorbing state (p_stay = 1)."""
        pytest.importorskip("hmmlearn")

        config = HMMConfig(n_regimes=2)
        detector = RegimeDetector(config)
        detector.fit(sample_prices)

        result = detector.get_regime_statistics(sample_prices)

        # Verify durations are computed
        for i in range(config.n_regimes):
            assert i in result.regime_durations
            # Duration should be positive (or inf for absorbing state)
            assert result.regime_durations[i] > 0


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
