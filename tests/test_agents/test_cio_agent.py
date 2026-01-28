"""Tests for CIOAgent class."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from hrp.agents.cio import CIOAgent


class TestCIOAgentInit:
    """Test CIOAgent initialization."""

    def test_init_with_defaults(self):
        """Test CIOAgent can be initialized with defaults."""
        with patch("hrp.agents.cio.PlatformAPI"):
            agent = CIOAgent(
                job_id="test-job-001",
                actor="agent:cio-test",
            )

            assert agent.agent_name == "cio"
            assert agent.agent_version == "1.0.0"
            assert agent.api is not None
            assert agent.thresholds["min_sharpe"] == 1.0
            assert agent.thresholds["max_drawdown"] == 0.20

    def test_init_with_custom_thresholds(self):
        """Test CIOAgent accepts custom thresholds."""
        with patch("hrp.agents.cio.PlatformAPI"):
            custom_thresholds = {
                "min_sharpe": 1.5,
                "max_drawdown": 0.15,
            }
            agent = CIOAgent(
                job_id="test-job-002",
                actor="agent:cio-test",
                thresholds=custom_thresholds,
            )

            assert agent.thresholds["min_sharpe"] == 1.5
            assert agent.thresholds["max_drawdown"] == 0.15
            # Defaults still present
            assert agent.thresholds["sharpe_decay_limit"] == 0.50

    def test_init_with_passed_api(self):
        """Test CIOAgent accepts a PlatformAPI instance."""
        with patch("hrp.agents.cio.PlatformAPI") as mock_api_class:
            mock_api = Mock()
            agent = CIOAgent(
                job_id="test-job-003",
                actor="agent:cio-test",
                api=mock_api,
            )

            assert agent.api == mock_api
            # PlatformAPI class not called again since we passed instance
            mock_api_class.return_value.assert_not_called()
