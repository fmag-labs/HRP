"""Tests for MCP server tools."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.mcp import research_server


# Helper to call the underlying function from a FunctionTool
def call_tool(tool, *args, **kwargs):
    """Call the underlying function of an MCP tool."""
    return tool.fn(*args, **kwargs)


@pytest.fixture
def mock_api():
    """Create a mock PlatformAPI."""
    with patch.object(research_server, "_api", None):
        with patch.object(research_server, "get_api") as mock_get_api:
            api = MagicMock()
            mock_get_api.return_value = api
            yield api


# =============================================================================
# Hypothesis Management Tools Tests
# =============================================================================


class TestListHypotheses:
    """Tests for list_hypotheses tool."""

    def test_list_hypotheses_success(self, mock_api):
        """List hypotheses successfully."""
        mock_api.list_hypotheses.return_value = [
            {
                "hypothesis_id": "HYP-2026-001",
                "title": "Momentum predicts returns",
                "status": "testing",
                "created_at": "2026-01-15T10:00:00",
                "updated_at": None,
            }
        ]

        result = call_tool(research_server.list_hypotheses)

        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["hypothesis_id"] == "HYP-2026-001"
        mock_api.list_hypotheses.assert_called_once_with(status=None, limit=100)

    def test_list_hypotheses_with_filter(self, mock_api):
        """List hypotheses with status filter."""
        mock_api.list_hypotheses.return_value = []

        result = call_tool(research_server.list_hypotheses, status="draft", limit=10)

        mock_api.list_hypotheses.assert_called_once_with(status="draft", limit=10)


class TestGetHypothesis:
    """Tests for get_hypothesis tool."""

    def test_get_hypothesis_found(self, mock_api):
        """Get existing hypothesis."""
        mock_api.get_hypothesis.return_value = {
            "hypothesis_id": "HYP-2026-001",
            "title": "Test",
            "thesis": "Test thesis",
            "prediction": "Test prediction",
            "falsification": "Test criteria",
            "status": "draft",
            "created_at": "2026-01-15T10:00:00",
            "updated_at": None,
        }

        result = call_tool(research_server.get_hypothesis, "HYP-2026-001")

        assert result["success"] is True
        assert result["data"]["hypothesis_id"] == "HYP-2026-001"

    def test_get_hypothesis_not_found(self, mock_api):
        """Handle non-existent hypothesis."""
        mock_api.get_hypothesis.return_value = None

        result = call_tool(research_server.get_hypothesis, "HYP-2026-999")

        assert result["success"] is False
        assert "not found" in result["message"].lower()


class TestCreateHypothesis:
    """Tests for create_hypothesis tool."""

    def test_create_hypothesis_success(self, mock_api):
        """Create hypothesis successfully."""
        mock_api.create_hypothesis.return_value = "HYP-2026-002"

        result = call_tool(
            research_server.create_hypothesis,
            title="New hypothesis",
            thesis="Something is true",
            prediction="We expect X",
            falsification="Reject if Y",
        )

        assert result["success"] is True
        assert result["data"]["hypothesis_id"] == "HYP-2026-002"
        mock_api.create_hypothesis.assert_called_once_with(
            title="New hypothesis",
            thesis="Something is true",
            prediction="We expect X",
            falsification="Reject if Y",
            actor="agent:claude-interactive",
        )


class TestUpdateHypothesis:
    """Tests for update_hypothesis tool."""

    def test_update_hypothesis_success(self, mock_api):
        """Update hypothesis status."""
        result = call_tool(
            research_server.update_hypothesis,
            hypothesis_id="HYP-2026-001",
            status="validated",
            outcome="Results confirmed prediction",
        )

        assert result["success"] is True
        mock_api.update_hypothesis.assert_called_once_with(
            hypothesis_id="HYP-2026-001",
            status="validated",
            outcome="Results confirmed prediction",
            actor="agent:claude-interactive",
        )


# =============================================================================
# Data Access Tools Tests
# =============================================================================


class TestGetUniverse:
    """Tests for get_universe tool."""

    def test_get_universe_success(self, mock_api):
        """Get trading universe."""
        mock_api.get_universe.return_value = ["AAPL", "MSFT", "GOOGL"]

        result = call_tool(research_server.get_universe)

        assert result["success"] is True
        assert len(result["data"]["symbols"]) == 3
        assert result["data"]["count"] == 3

    def test_get_universe_with_date(self, mock_api):
        """Get universe for specific date."""
        mock_api.get_universe.return_value = ["AAPL"]

        result = call_tool(research_server.get_universe, as_of_date="2023-01-15")

        mock_api.get_universe.assert_called_once_with(date(2023, 1, 15))


class TestGetFeatures:
    """Tests for get_features tool."""

    def test_get_features_success(self, mock_api):
        """Get feature values."""
        mock_api.get_features.return_value = pd.DataFrame({
            "symbol": ["AAPL"],
            "momentum_20d": [0.05],
            "volatility_60d": [0.20],
        })

        result = call_tool(
            research_server.get_features,
            symbols=["AAPL"],
            features=["momentum_20d", "volatility_60d"],
            as_of_date="2023-01-15",
        )

        assert result["success"] is True

    def test_get_features_missing_date(self, mock_api):
        """Handle missing date parameter."""
        result = call_tool(
            research_server.get_features,
            symbols=["AAPL"],
            features=["momentum_20d"],
            as_of_date=None,
        )

        assert result["success"] is False
        assert "required" in result["error"].lower()


class TestGetAvailableFeatures:
    """Tests for get_available_features tool."""

    def test_get_available_features(self, mock_api):
        """List available features."""
        with patch("hrp.data.features.registry.FeatureRegistry") as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.list_all_features.return_value = [
                {
                    "feature_name": "momentum_20d",
                    "version": "v1",
                    "description": "20-day momentum",
                }
            ]
            MockRegistry.return_value = mock_registry

            result = call_tool(research_server.get_available_features)

            assert result["success"] is True
            assert len(result["data"]) == 1


class TestIsTradingDay:
    """Tests for is_trading_day tool."""

    def test_is_trading_day_true(self, mock_api):
        """Check trading day - yes."""
        mock_api.is_trading_day.return_value = True

        result = call_tool(research_server.is_trading_day, "2023-01-16")

        assert result["success"] is True
        assert result["data"]["is_trading_day"] is True

    def test_is_trading_day_false(self, mock_api):
        """Check trading day - no (weekend)."""
        mock_api.is_trading_day.return_value = False

        result = call_tool(research_server.is_trading_day, "2023-01-15")

        assert result["success"] is True
        assert result["data"]["is_trading_day"] is False


# =============================================================================
# Backtesting Tools Tests
# =============================================================================


class TestRunBacktest:
    """Tests for run_backtest tool."""

    def test_run_backtest_success(self, mock_api):
        """Run backtest successfully."""
        mock_api.run_backtest.return_value = "exp-123"
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {
                "sharpe_ratio": 1.5,
                "total_return": 0.25,
                "max_drawdown": -0.10,
                "cagr": 0.15,
                "volatility": 0.10,
            },
        }

        result = call_tool(
            research_server.run_backtest,
            hypothesis_id="HYP-2026-001",
            symbols=["AAPL", "MSFT"],
            start_date="2020-01-01",
            end_date="2023-12-31",
        )

        assert result["success"] is True
        assert result["data"]["experiment_id"] == "exp-123"
        assert result["data"]["metrics"]["sharpe_ratio"] == 1.5


class TestGetExperiment:
    """Tests for get_experiment tool."""

    def test_get_experiment_found(self, mock_api):
        """Get existing experiment."""
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "status": "FINISHED",
            "metrics": {"sharpe_ratio": 1.2},
            "params": {},
            "tags": {},
        }

        result = call_tool(research_server.get_experiment, "exp-123")

        assert result["success"] is True
        assert result["data"]["experiment_id"] == "exp-123"

    def test_get_experiment_not_found(self, mock_api):
        """Handle non-existent experiment."""
        mock_api.get_experiment.return_value = None

        result = call_tool(research_server.get_experiment, "nonexistent")

        assert result["success"] is False
        assert "not found" in result["message"].lower()


class TestCompareExperiments:
    """Tests for compare_experiments tool."""

    def test_compare_experiments_success(self, mock_api):
        """Compare experiments."""
        mock_api.compare_experiments.return_value = pd.DataFrame({
            "sharpe_ratio": [1.2, 1.5],
            "total_return": [0.20, 0.25],
        }, index=["exp-1", "exp-2"])

        result = call_tool(
            research_server.compare_experiments,
            experiment_ids=["exp-1", "exp-2"],
        )

        assert result["success"] is True


# =============================================================================
# Quality & Health Tools Tests
# =============================================================================


class TestRunQualityChecks:
    """Tests for run_quality_checks tool."""

    def test_run_quality_checks_success(self, mock_api):
        """Run quality checks."""
        with patch("hrp.data.quality.report.QualityReportGenerator") as MockGenerator:
            mock_report = MagicMock()
            mock_report.report_date = date(2026, 1, 15)
            mock_report.health_score = 95.0
            mock_report.passed = True
            mock_report.checks_run = 5
            mock_report.checks_passed = 5
            mock_report.critical_issues = 0
            mock_report.warning_issues = 1
            mock_report.get_summary_text.return_value = "All checks passed"
            MockGenerator.return_value.generate_report.return_value = mock_report

            result = call_tool(research_server.run_quality_checks)

            assert result["success"] is True
            assert result["data"]["health_score"] == 95.0
            assert result["data"]["passed"] is True


class TestGetHealthStatus:
    """Tests for get_health_status tool."""

    def test_get_health_status_success(self, mock_api):
        """Get health status."""
        mock_api.health_check.return_value = {
            "api": "ok",
            "database": "ok",
            "tables": {"prices": {"status": "ok", "count": 1000}},
        }

        result = call_tool(research_server.get_health_status)

        assert result["success"] is True
        assert result["data"]["api"] == "ok"
        assert result["data"]["database"] == "ok"


# =============================================================================
# Lineage Tools Tests
# =============================================================================


class TestGetLineage:
    """Tests for get_lineage tool."""

    def test_get_lineage_success(self, mock_api):
        """Get lineage events."""
        mock_api.get_lineage.return_value = [
            {
                "lineage_id": 1,
                "event_type": "hypothesis_created",
                "timestamp": "2026-01-15T10:00:00",
                "actor": "user",
                "hypothesis_id": "HYP-2026-001",
                "experiment_id": None,
                "details": {},
            }
        ]

        result = call_tool(research_server.get_lineage, hypothesis_id="HYP-2026-001")

        assert result["success"] is True
        assert len(result["data"]) == 1


class TestGetDeployedStrategies:
    """Tests for get_deployed_strategies tool."""

    def test_get_deployed_strategies_success(self, mock_api):
        """Get deployed strategies."""
        mock_api.get_deployed_strategies.return_value = [
            {
                "hypothesis_id": "HYP-2025-010",
                "title": "Deployed strategy",
                "status": "deployed",
                "created_at": "2025-06-01T10:00:00",
            }
        ]

        result = call_tool(research_server.get_deployed_strategies)

        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["status"] == "deployed"


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityConstraints:
    """Tests for security constraints."""

    def test_approve_deployment_not_exposed(self):
        """Verify approve_deployment is NOT exposed as a tool."""
        # Get all tool names from the MCP server
        tool_names = list(research_server.mcp._tool_manager._tools.keys())

        assert "approve_deployment" not in tool_names

    def test_actor_is_agent_identifier(self):
        """Verify ACTOR constant identifies as agent."""
        assert research_server.ACTOR.startswith("agent:")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for tool workflows."""

    def test_hypothesis_to_experiment_workflow(self, mock_api):
        """Test workflow: create hypothesis -> run backtest -> get results."""
        # 1. Create hypothesis
        mock_api.create_hypothesis.return_value = "HYP-2026-001"
        create_result = call_tool(
            research_server.create_hypothesis,
            title="Test hypothesis",
            thesis="Test thesis",
            prediction="Test prediction",
            falsification="Test falsification",
        )
        assert create_result["success"] is True
        hypothesis_id = create_result["data"]["hypothesis_id"]

        # 2. Run backtest
        mock_api.run_backtest.return_value = "exp-123"
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {"sharpe_ratio": 1.2},
        }
        backtest_result = call_tool(
            research_server.run_backtest,
            hypothesis_id=hypothesis_id,
            symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2023-12-31",
        )
        assert backtest_result["success"] is True
        experiment_id = backtest_result["data"]["experiment_id"]

        # 3. Get experiment results
        mock_api.get_experiment.return_value = {
            "experiment_id": experiment_id,
            "metrics": {"sharpe_ratio": 1.2, "total_return": 0.3, "max_drawdown": -0.1},
            "params": {},
            "tags": {},
        }
        exp_result = call_tool(research_server.get_experiment, experiment_id)
        assert exp_result["success"] is True
        assert exp_result["data"]["metrics"]["sharpe_ratio"] == 1.2

        # 4. Get linked experiments
        mock_api.get_experiments_for_hypothesis.return_value = [experiment_id]
        linked = call_tool(
            research_server.get_experiments_for_hypothesis,
            hypothesis_id,
        )
        assert linked["success"] is True
        assert experiment_id in linked["data"]["experiment_ids"]
