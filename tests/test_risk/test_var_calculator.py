"""
Unit tests for VaR calculator.

Tests all three VaR methods (parametric, historical, Monte Carlo) and validates
mathematical properties like VaR <= CVaR, proper sign conventions, etc.
"""

import numpy as np
import pytest
from datetime import datetime

from hrp.data.risk.risk_config import Distribution, VaRConfig, VaRMethod
from hrp.data.risk.var_calculator import VaRCalculator, VaRResult


class TestVaRResult:
    """Test VaRResult dataclass."""

    def test_var_result_creation(self):
        """Test creating a VaRResult."""
        result = VaRResult(
            var=0.05,
            cvar=0.07,
            confidence_level=0.95,
            time_horizon=1,
            method="parametric",
            timestamp=datetime.now(),
        )
        assert result.var == 0.05
        assert result.cvar == 0.07
        assert result.confidence_level == 0.95
        assert result.var_dollar is None

    def test_var_result_with_portfolio_value(self):
        """Test VaRResult calculates dollar values when portfolio_value is provided."""
        portfolio_value = 1_000_000
        result = VaRResult(
            var=0.05,
            cvar=0.07,
            confidence_level=0.95,
            time_horizon=1,
            method="parametric",
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
        )
        assert result.var_dollar == 50_000  # 5% of 1M
        assert result.cvar_dollar == 70_000  # 7% of 1M


class TestVaRConfig:
    """Test VaRConfig validation."""

    def test_valid_config(self):
        """Test creating a valid VaRConfig."""
        config = VaRConfig(
            confidence_level=0.95,
            time_horizon=1,
            method=VaRMethod.PARAMETRIC,
        )
        assert config.confidence_level == 0.95
        assert config.time_horizon == 1

    def test_invalid_confidence_level(self):
        """Test that invalid confidence levels raise ValueError."""
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            VaRConfig(confidence_level=1.5)

        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            VaRConfig(confidence_level=-0.1)

    def test_invalid_time_horizon(self):
        """Test that invalid time horizons raise ValueError."""
        with pytest.raises(ValueError, match="time_horizon must be at least 1 day"):
            VaRConfig(time_horizon=0)

    def test_invalid_n_simulations(self):
        """Test that too few simulations raise ValueError."""
        with pytest.raises(ValueError, match="n_simulations must be at least 100"):
            VaRConfig(n_simulations=50)


class TestVaRCalculator:
    """Test VaR calculator with all three methods."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns data for testing."""
        np.random.seed(42)
        # Generate 252 days of returns (1 year) with mean 0.05% and std 1%
        return np.random.normal(0.0005, 0.01, 252)

    @pytest.fixture
    def calculator(self):
        """Create a VaR calculator instance."""
        return VaRCalculator()

    def test_parametric_var_normal(self, calculator, sample_returns):
        """Test parametric VaR with normal distribution."""
        config = VaRConfig(
            confidence_level=0.95,
            time_horizon=1,
            method=VaRMethod.PARAMETRIC,
            distribution=Distribution.NORMAL,
        )
        result = calculator.calculate(sample_returns, config=config)

        # Check that result has correct structure
        assert isinstance(result, VaRResult)
        assert result.confidence_level == 0.95
        assert result.method == "parametric"

        # VaR and CVaR should be positive (representing losses)
        assert result.var > 0
        assert result.cvar > 0

        # CVaR should be greater than or equal to VaR
        assert result.cvar >= result.var

        # For 95% confidence, VaR should be reasonable (roughly 1-3% for daily)
        assert 0.005 < result.var < 0.05

    def test_parametric_var_t_distribution(self, calculator, sample_returns):
        """Test parametric VaR with t-distribution (fat tails)."""
        config = VaRConfig(
            confidence_level=0.95,
            time_horizon=1,
            method=VaRMethod.PARAMETRIC,
            distribution=Distribution.T,
            df=5.0,
        )
        result = calculator.calculate(sample_returns, config=config)

        # t-distribution VaR should be higher than normal (fatter tails)
        config_normal = VaRConfig(
            confidence_level=0.95,
            time_horizon=1,
            method=VaRMethod.PARAMETRIC,
            distribution=Distribution.NORMAL,
        )
        result_normal = calculator.calculate(sample_returns, config=config_normal)

        # t-distribution should give higher VaR (more conservative)
        assert result.var > result_normal.var

    def test_historical_var_1day(self, calculator, sample_returns):
        """Test historical VaR with 1-day horizon."""
        config = VaRConfig(
            confidence_level=0.95,
            time_horizon=1,
            method=VaRMethod.HISTORICAL,
        )
        result = calculator.calculate(sample_returns, config=config)

        assert result.var > 0
        assert result.cvar > 0
        assert result.cvar >= result.var
        assert result.method == "historical"

        # Historical VaR should be the 5th percentile of losses
        # Verify it's approximately correct
        losses = -sample_returns
        expected_var = np.quantile(losses, 0.95)
        assert abs(result.var - expected_var) < 0.001

    def test_historical_var_multiday(self, calculator, sample_returns):
        """Test historical VaR with multi-day horizon."""
        config = VaRConfig(
            confidence_level=0.95,
            time_horizon=10,
            method=VaRMethod.HISTORICAL,
        )
        result = calculator.calculate(sample_returns, config=config)

        # 10-day VaR should be higher than 1-day VaR
        config_1d = VaRConfig(
            confidence_level=0.95,
            time_horizon=1,
            method=VaRMethod.HISTORICAL,
        )
        result_1d = calculator.calculate(sample_returns, config=config_1d)

        assert result.var > result_1d.var

    def test_monte_carlo_var(self, calculator, sample_returns):
        """Test Monte Carlo VaR."""
        np.random.seed(42)  # For reproducibility
        config = VaRConfig(
            confidence_level=0.95,
            time_horizon=1,
            method=VaRMethod.MONTE_CARLO,
            distribution=Distribution.T,
            n_simulations=5000,
        )
        result = calculator.calculate(sample_returns, config=config)

        assert result.var > 0
        assert result.cvar > 0
        assert result.cvar >= result.var
        assert result.method == "monte_carlo"

        # MC VaR should be somewhat close to parametric VaR
        # (within 50% as it's stochastic)
        config_parametric = VaRConfig(
            confidence_level=0.95,
            time_horizon=1,
            method=VaRMethod.PARAMETRIC,
            distribution=Distribution.T,
        )
        result_parametric = calculator.calculate(sample_returns, config=config_parametric)

        ratio = result.var / result_parametric.var
        assert 0.5 < ratio < 1.5

    def test_var_increases_with_confidence(self, calculator, sample_returns):
        """Test that VaR increases with confidence level."""
        result_95 = calculator.calculate(
            sample_returns,
            config=VaRConfig(confidence_level=0.95, method=VaRMethod.PARAMETRIC),
        )
        result_99 = calculator.calculate(
            sample_returns,
            config=VaRConfig(confidence_level=0.99, method=VaRMethod.PARAMETRIC),
        )

        # 99% VaR should be higher than 95% VaR
        assert result_99.var > result_95.var
        assert result_99.cvar > result_95.cvar

    def test_var_scales_with_time_horizon(self, calculator, sample_returns):
        """Test that VaR increases with time horizon (sqrt rule for parametric)."""
        result_1d = calculator.calculate(
            sample_returns,
            config=VaRConfig(time_horizon=1, method=VaRMethod.PARAMETRIC),
        )
        result_10d = calculator.calculate(
            sample_returns,
            config=VaRConfig(time_horizon=10, method=VaRMethod.PARAMETRIC),
        )

        # 10-day VaR should be approximately sqrt(10) times 1-day VaR
        expected_ratio = np.sqrt(10)
        actual_ratio = result_10d.var / result_1d.var

        # Allow 20% tolerance due to drift term
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.2

    def test_portfolio_value_calculation(self, calculator, sample_returns):
        """Test that dollar VaR is correctly calculated."""
        portfolio_value = 1_000_000
        result = calculator.calculate(
            sample_returns,
            portfolio_value=portfolio_value,
            config=VaRConfig(confidence_level=0.95),
        )

        assert result.portfolio_value == portfolio_value
        assert result.var_dollar == result.var * portfolio_value
        assert result.cvar_dollar == result.cvar * portfolio_value

    def test_calculate_all_methods(self, calculator, sample_returns):
        """Test calculating VaR with all methods at once."""
        results = calculator.calculate_all_methods(
            sample_returns,
            confidence_level=0.95,
            time_horizon=1,
        )

        # Should have results for all three methods
        assert "parametric" in results
        assert "historical" in results
        assert "monte_carlo" in results

        # All results should be valid
        for method_name, result in results.items():
            assert isinstance(result, VaRResult)
            assert result.var > 0
            assert result.cvar > 0
            assert result.cvar >= result.var

    def test_empty_returns_raises_error(self, calculator):
        """Test that empty returns array raises ValueError."""
        with pytest.raises(ValueError, match="Returns array cannot be empty"):
            calculator.calculate(np.array([]))

    def test_insufficient_returns_raises_error(self, calculator):
        """Test that too few returns raises ValueError."""
        with pytest.raises(ValueError, match="Need at least 30 valid return observations"):
            calculator.calculate(np.random.normal(0, 0.01, 20))

    def test_nan_handling(self, calculator):
        """Test that NaN values are properly filtered."""
        returns_with_nan = np.array([0.01, np.nan, -0.02, 0.015, np.nan] + list(np.random.normal(0, 0.01, 100)))
        result = calculator.calculate(returns_with_nan)

        # Should succeed (NaN filtered out)
        assert result.var > 0

    def test_cvar_invariant(self, calculator, sample_returns):
        """Test that CVaR >= VaR always holds (mathematical invariant)."""
        methods = [VaRMethod.PARAMETRIC, VaRMethod.HISTORICAL, VaRMethod.MONTE_CARLO]
        confidence_levels = [0.90, 0.95, 0.99]

        for method in methods:
            for conf in confidence_levels:
                config = VaRConfig(
                    confidence_level=conf,
                    time_horizon=1,
                    method=method,
                )
                if method == VaRMethod.MONTE_CARLO:
                    np.random.seed(42)  # For reproducibility

                result = calculator.calculate(sample_returns, config=config)

                # CVaR must always be >= VaR
                assert result.cvar >= result.var, (
                    f"CVaR < VaR for method={method.value}, conf={conf}: "
                    f"CVaR={result.cvar}, VaR={result.var}"
                )
