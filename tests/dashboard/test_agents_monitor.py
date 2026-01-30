"""Tests for agents monitor backend functions."""

from __future__ import annotations

from unittest.mock import patch, MagicMock


def test_agents_monitor_module_exists():
    """agents_monitor module should exist with core functions."""
    from hrp.dashboard.agents_monitor import (
        get_all_agent_status,
        get_timeline,
        AgentStatus
    )
    assert callable(get_all_agent_status)
    assert callable(get_timeline)


def test_get_all_agent_status_returns_list():
    """get_all_agent_status should return list of AgentStatus."""
    from hrp.dashboard.agents_monitor import get_all_agent_status
    from hrp.api.platform import PlatformAPI

    # Mock get_lineage to return empty events (all agents will be idle)
    with patch("hrp.dashboard.agents_monitor.get_lineage", return_value=[]):
        api = PlatformAPI()
        result = get_all_agent_status(api)
        assert isinstance(result, list)
        # All agents should be present
        agent_ids = {a.agent_id for a in result}
        expected_agents = {
            "signal-scientist", "alpha-researcher", "code-materializer",
            "ml-scientist", "ml-quality-sentinel", "quant-developer",
            "pipeline-orchestrator", "validation-analyst", "risk-manager",
            "cio", "report-generator"
        }
        assert expected_agents.issubset(agent_ids)


def test_agent_status_has_valid_status_field():
    """Each AgentStatus should have valid status field."""
    from hrp.dashboard.agents_monitor import get_all_agent_status
    from hrp.api.platform import PlatformAPI

    # Mock get_lineage to return empty events (all agents will be idle)
    with patch("hrp.dashboard.agents_monitor.get_lineage", return_value=[]):
        api = PlatformAPI()
        result = get_all_agent_status(api)
        valid_statuses = {"running", "completed", "failed", "idle"}
        for agent in result:
            assert agent.status in valid_statuses
