"""Tests for utility configuration classes."""

import pytest

from hrp.utils.config import Config, DefaultBacktestConfig, Environment, reset_config


class TestEnvironmentEnum:
    """Tests for Environment enum."""

    def test_environment_enum_values(self):
        """Environment enum should have correct string values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_config_environment_from_env(self, monkeypatch):
        """Config should read environment from HRP_ENVIRONMENT."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "production")
        reset_config()
        from hrp.utils.config import get_config

        config = get_config()
        assert config.environment == Environment.PRODUCTION
        reset_config()

    def test_config_environment_defaults_to_development(self, monkeypatch):
        """Config should default to development environment."""
        monkeypatch.delenv("HRP_ENVIRONMENT", raising=False)
        reset_config()
        from hrp.utils.config import get_config

        config = get_config()
        assert config.environment == Environment.DEVELOPMENT
        reset_config()

    def test_config_auth_required_by_environment(self):
        """auth_required should be False for development, True otherwise."""
        dev_config = Config(environment=Environment.DEVELOPMENT)
        staging_config = Config(environment=Environment.STAGING)
        prod_config = Config(environment=Environment.PRODUCTION)

        assert dev_config.auth_required is False
        assert staging_config.auth_required is True
        assert prod_config.auth_required is True

    def test_config_debug_mode_by_environment(self):
        """debug_mode should be True only for development."""
        dev_config = Config(environment=Environment.DEVELOPMENT)
        staging_config = Config(environment=Environment.STAGING)
        prod_config = Config(environment=Environment.PRODUCTION)

        assert dev_config.debug_mode is True
        assert staging_config.debug_mode is False
        assert prod_config.debug_mode is False

    def test_config_invalid_environment_defaults_to_development(self, monkeypatch):
        """Invalid HRP_ENVIRONMENT value should default to development."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "invalid_env")
        reset_config()
        from hrp.utils.config import get_config

        config = get_config()
        assert config.environment == Environment.DEVELOPMENT
        reset_config()


class TestDefaultBacktestConfig:
    """Tests for DefaultBacktestConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating DefaultBacktestConfig with default values."""
        config = DefaultBacktestConfig()
        assert config.max_position_pct == 0.10
        assert config.max_positions == 20
        assert config.min_position_pct == 0.02
        assert config.commission_pct == 0.0005
        assert config.slippage_pct == 0.001
        assert config.max_gross_exposure == 1.0
        assert config.strategy_stop_loss == 0.15
        assert config.portfolio_stop_loss == 0.20

    def test_config_with_custom_values(self):
        """Test creating DefaultBacktestConfig with custom values."""
        config = DefaultBacktestConfig(
            max_position_pct=0.05,
            max_positions=10,
            min_position_pct=0.01,
            commission_pct=0.001,
            slippage_pct=0.002,
            max_gross_exposure=0.8,
            strategy_stop_loss=0.10,
            portfolio_stop_loss=0.15,
        )

        assert config.max_position_pct == 0.05
        assert config.max_positions == 10
        assert config.min_position_pct == 0.01
        assert config.commission_pct == 0.001
        assert config.slippage_pct == 0.002
        assert config.max_gross_exposure == 0.8
        assert config.strategy_stop_loss == 0.10
        assert config.portfolio_stop_loss == 0.15
