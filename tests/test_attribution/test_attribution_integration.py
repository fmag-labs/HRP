"""End-to-end integration tests for performance attribution system."""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hrp.data.attribution.factor_attribution import BrinsonAttribution, FactorAttribution
from hrp.data.attribution.feature_importance import FeatureImportanceTracker, RollingImportance
from hrp.data.attribution.decision_attribution import (
    DecisionAttributor,
    RebalanceAnalyzer,
    TradeDecision,
)
from hrp.data.attribution.attribution_config import AttributionConfig


class TestAttributionE2E:
    """End-to-end tests for complete attribution pipeline."""

    def test_full_attribution_pipeline(self):
        """Brinson attribution decomposes active return into allocation/selection/interaction."""
        sectors = ["Technology", "Healthcare", "Finance"]
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=sectors)
        benchmark_weights = pd.Series([0.35, 0.35, 0.30], index=sectors)
        portfolio_returns = pd.Series([0.02, -0.01, 0.01], index=sectors)
        benchmark_returns = pd.Series([0.015, 0.005, 0.01], index=sectors)

        attributor = BrinsonAttribution()
        results = attributor.attribute(
            portfolio_weights=portfolio_weights,
            portfolio_returns=portfolio_returns,
            benchmark_weights=benchmark_weights,
            benchmark_returns=benchmark_returns,
        )

        assert len(results) > 0
        effect_types = {r.effect_type for r in results}
        assert effect_types & {"allocation", "selection", "interaction"}
        assert all(np.isfinite(r.contribution_pct) for r in results)

    def test_regression_attribution_pipeline(self):
        """Regression-based factor attribution attributes returns to each factor."""
        n_days = 60
        dates = pd.date_range(end=date.today(), periods=n_days, freq="D")

        factor_returns = pd.DataFrame({
            "Market": np.random.normal(0.0005, 0.01, n_days),
            "Value": np.random.normal(0.0002, 0.005, n_days),
            "Momentum": np.random.normal(0.0001, 0.006, n_days),
        }, index=dates)

        portfolio_returns = pd.Series(
            0.5 * factor_returns["Market"]
            + 0.3 * factor_returns["Value"]
            + 0.2 * factor_returns["Momentum"]
            + np.random.normal(0, 0.002, n_days),
            index=dates,
        )

        attributor = FactorAttribution()
        results = attributor.attribute(
            portfolio_returns=portfolio_returns,
            factor_returns=factor_returns,
        )

        assert len(results) >= 3
        factor_names = {r.factor for r in results}
        assert {"Market", "Value", "Momentum"} <= factor_names

    def test_feature_importance_integration(self):
        """Permutation importance ranks the truly predictive features highest."""
        from sklearn.ensemble import RandomForestRegressor

        n_samples = 100
        n_features = 5
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = 2 * X["feature_0"] + 1.5 * X["feature_1"] + np.random.normal(0, 0.1, n_samples)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        tracker = FeatureImportanceTracker(n_repeats=5)
        results = tracker.compute_permutation_importance(model=model, X=X, y=y)

        importance_dict = {r.feature_name: r.importance_score for r in results}
        assert len(results) == n_features
        assert importance_dict["feature_0"] > 0
        assert importance_dict["feature_1"] > 0

        top_2_names = {
            r.feature_name
            for r in sorted(results, key=lambda x: x.importance_score, reverse=True)[:2]
        }
        assert "feature_0" in top_2_names or "feature_1" in top_2_names

    def test_decision_attribution_integration(self):
        """Trade-level attribution decomposes P&L into timing/sizing/residual."""
        entry_date = datetime.today() - timedelta(days=10)
        exit_date = datetime.today() - timedelta(days=1)

        trade = TradeDecision(
            trade_id="T-INT-001",
            asset="AAPL",
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=100.0,
            exit_price=110.0,
            quantity=100.0,
            pnl=None,
        )

        attributor = DecisionAttributor()
        result = attributor.attribute_trade(
            trade,
            benchmark_entry_price=101.0,
            benchmark_exit_price=109.0,
            optimal_quantity=120.0,
        )

        assert result.timing_contribution is not None
        assert result.sizing_contribution is not None
        assert result.residual is not None

        total_decomposed = (
            result.timing_contribution + result.sizing_contribution + result.residual
        )
        assert abs(total_decomposed - result.pnl) < 0.01

    def test_rolling_feature_importance(self):
        """Rolling importance produces a per-window importance DataFrame."""
        from sklearn.ensemble import RandomForestRegressor

        n_days = 90
        n_features = 4
        dates = pd.date_range(end=date.today(), periods=n_days, freq="D")

        X = pd.DataFrame(
            np.random.randn(n_days, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = 2 * X["feature_0"] + np.random.normal(0, 0.1, n_days)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        rolling = RollingImportance(window_days=30, step_days=15)
        results = rolling.compute_rolling_importance(
            model=model, X=X, y=y, dates=dates
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_rebalancing_value_add(self):
        """Rebalance analysis reports the value added by a weight change."""
        analyzer = RebalanceAnalyzer()
        result = analyzer.analyze_rebalance_event(
            date=datetime(2024, 1, 15),
            pre_rebalance_weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
            post_rebalance_weights={"ASSET_A": 0.5, "ASSET_B": 0.5},
            asset_returns_since_rebalance={"ASSET_A": 0.05, "ASSET_B": 0.03},
        )

        assert "value_add" in result
        assert np.isfinite(result["value_add"])


class TestAttributionConfigIntegration:
    """Test configuration and pipeline integration."""

    def test_config_creation(self):
        """Test attribution config can be created with all options."""
        config = AttributionConfig(
            method="brinson",
            benchmark="SPY",
            lookback_days=252,
            factor_model="fama_french_3",
            permutation_n_repeats=10,
            shap_enabled=False,
            rolling_window_days=60,
            include_timing=True,
            include_sizing=True,
            validate_summation=True,
        )

        assert config.method == "brinson"
        assert config.benchmark == "SPY"
        assert config.lookback_days == 252
        assert config.validate_summation is True

    def test_default_config(self):
        """Test default configuration works."""
        from hrp.data.attribution.attribution_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG.method == "brinson"
        assert DEFAULT_CONFIG.validate_summation is True
        assert DEFAULT_CONFIG.cache_enabled is True
