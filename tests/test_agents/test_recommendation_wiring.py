"""Tests that the recommendation agent is wired into job execution and scheduling."""

from unittest.mock import patch

from hrp.agents.run_job import JOBS, run_recommendations
from hrp.agents.scheduler import IngestionScheduler


def test_recommendation_job_registered():
    """The 'recommendations' job must be discoverable via the run_job registry."""
    assert "recommendations" in JOBS
    assert JOBS["recommendations"] is run_recommendations


def test_run_recommendations_dry_run():
    """Dry-run must not instantiate the agent and returns a dry_run marker."""
    result = run_recommendations(dry_run=True)
    assert result == {"status": "dry_run", "job": "recommendations"}


def test_run_recommendations_invokes_agent():
    """Non-dry-run delegates to RecommendationAgent.run()."""
    with patch("hrp.agents.recommendation_agent.RecommendationAgent") as MAgent:
        MAgent.return_value.run.return_value = {
            "recommendations_generated": 3,
            "recommendations_closed": 1,
        }
        result = run_recommendations(dry_run=False)

    MAgent.return_value.run.assert_called_once()
    assert result["recommendations_generated"] == 3


def test_scheduler_registers_weekly_recommendations():
    """setup_weekly_recommendations should register a scheduled job."""
    scheduler = IngestionScheduler()
    scheduler.setup_weekly_recommendations()

    job_ids = {j["id"] for j in scheduler.list_jobs()}
    assert "weekly_recommendations" in job_ids
