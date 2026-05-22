"""Tests for the recommendation job runner."""

from unittest.mock import MagicMock, patch

from hrp.agents.run_job import JOBS, run_recommendations


def test_recommendations_registered_as_job():
    """The advisory recommendation agent should be runnable through run_job."""
    assert "recommendations" in JOBS
    assert JOBS["recommendations"] is run_recommendations


def test_run_recommendations_dry_run():
    """Dry-run mode should not instantiate the recommendation agent."""
    result = run_recommendations(dry_run=True)

    assert result == {"status": "dry_run", "job": "recommendations"}


@patch("hrp.agents.recommendation_agent.RecommendationAgent")
def test_run_recommendations_executes_agent(mock_agent_class):
    """run_recommendations should instantiate and run RecommendationAgent."""
    mock_agent = MagicMock()
    mock_agent.run.return_value = {
        "recommendations_generated": 2,
        "recommendations_closed": 1,
    }
    mock_agent_class.return_value = mock_agent

    result = run_recommendations()

    mock_agent_class.assert_called_once_with()
    mock_agent.run.assert_called_once_with()
    assert result["recommendations_generated"] == 2
    assert result["recommendations_closed"] == 1
