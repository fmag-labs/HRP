"""Tests for walk-forward validation."""

from datetime import date

import pytest

from hrp.ml.validation import WalkForwardConfig


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert config.model_type == "ridge"
        assert config.n_folds == 5
        assert config.window_type == "expanding"
        assert config.min_train_periods == 252
        assert config.feature_selection is True
        assert config.max_features == 20
        assert config.hyperparameters == {}

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = WalkForwardConfig(
            model_type="random_forest",
            target="returns_5d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=10,
            window_type="rolling",
            min_train_periods=504,
            hyperparameters={"n_estimators": 100},
        )
        assert config.n_folds == 10
        assert config.window_type == "rolling"
        assert config.min_train_periods == 504
        assert config.hyperparameters == {"n_estimators": 100}

    def test_config_invalid_model_type(self):
        """Test config rejects invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            WalkForwardConfig(
                model_type="invalid_model",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
            )

    def test_config_invalid_window_type(self):
        """Test config rejects invalid window type."""
        with pytest.raises(ValueError, match="window_type must be"):
            WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                window_type="invalid",
            )

    def test_config_invalid_n_folds(self):
        """Test config rejects n_folds < 2."""
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                n_folds=1,
            )
