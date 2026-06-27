"""Tests for the RecommendationAgent (weekly advisory pipeline driver).

These tests exercise the agent's orchestration logic (circuit breaker,
pre-trade gating, recommendation generation, track-record update and digest)
with all advisory components mocked, so no database or network is required.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hrp.research.lineage import EventType


def _check(passed: bool, severity: str = "info", name: str = "check", message: str = ""):
    """Build a stand-in for advisory.safeguards.CheckResult."""
    return SimpleNamespace(passed=passed, severity=severity, check_name=name, message=message)


@pytest.fixture
def agent():
    """RecommendationAgent with PlatformAPI and lineage logging stubbed out."""
    # Patch PlatformAPI in both the job base and research-agent base so the
    # constructor never opens a real database connection.
    with patch("hrp.agents.jobs.PlatformAPI"), patch("hrp.agents.base.PlatformAPI"):
        from hrp.agents.recommendation_agent import RecommendationAgent

        a = RecommendationAgent()

    a.api = Mock()
    a._log_agent_event = Mock(return_value=1)
    return a


@pytest.fixture
def comps():
    """Patch every advisory component used inside execute() and yield the instances."""
    with patch("hrp.advisory.safeguards.CircuitBreaker") as MB, patch(
        "hrp.advisory.safeguards.PreTradeChecks"
    ) as MP, patch(
        "hrp.advisory.recommendation_engine.RecommendationEngine"
    ) as ME, patch(
        "hrp.advisory.explainer.RecommendationExplainer"
    ), patch(
        "hrp.advisory.track_record.TrackRecordTracker"
    ) as MT:
        breaker = MB.return_value
        pretrade = MP.return_value
        engine = ME.return_value
        tracker = MT.return_value

        # Sensible defaults: nothing wrong, no recommendations.
        breaker.should_halt.return_value = (False, "")
        pretrade.run_all_checks.return_value = []
        engine.review_open_recommendations.return_value = []
        engine.generate_weekly_recommendations.return_value = []
        tracker.compute_track_record.return_value = SimpleNamespace(
            win_rate=0.6, excess_return=0.05
        )

        yield SimpleNamespace(
            breaker=breaker, pretrade=pretrade, engine=engine, tracker=tracker
        )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure email/risk env vars don't leak between tests."""
    monkeypatch.delenv("NOTIFICATION_EMAIL", raising=False)
    monkeypatch.delenv("HRP_ADVISORY_RISK_TOLERANCE", raising=False)


def _recs(*symbols):
    return [
        SimpleNamespace(symbol=s, confidence="HIGH", batch_id="BATCH-1") for s in symbols
    ]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_data_requirements_configured(agent):
    """Agent must require recent prices and at least one deployed model."""
    tables = {r.table for r in agent.data_requirements}
    assert tables == {"prices", "model_deployments"}
    assert agent.actor == "agent:recommendation"
    assert agent.job_id == "recommendation-agent"


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


def test_circuit_breaker_halts_pipeline(agent, comps):
    """When the circuit breaker trips, no recommendations are generated."""
    comps.breaker.should_halt.return_value = (True, "drawdown breach")

    result = agent.execute()

    assert result["circuit_breaker"] is True
    assert result["halt_reason"] == "drawdown breach"
    assert result["recommendations_generated"] == 0
    comps.engine.generate_weekly_recommendations.assert_not_called()
    agent._log_agent_event.assert_called_once()
    assert agent._log_agent_event.call_args[0][0] == EventType.CIRCUIT_BREAKER_ACTIVATED


# --------------------------------------------------------------------------- #
# Pre-trade gating
# --------------------------------------------------------------------------- #


def test_pretrade_error_halts_pipeline(agent, comps):
    """A failing pre-trade check with error severity stops generation."""
    comps.pretrade.run_all_checks.return_value = [
        _check(passed=False, severity="error", name="staleness", message="data stale")
    ]

    result = agent.execute()

    assert "failed_checks" in result
    assert result["recommendations_generated"] == 0
    comps.engine.generate_weekly_recommendations.assert_not_called()


def test_pretrade_warning_does_not_halt(agent, comps):
    """A warning-severity check is logged but the pipeline proceeds."""
    comps.pretrade.run_all_checks.return_value = [
        _check(passed=False, severity="warning", name="liquidity", message="thin volume")
    ]
    comps.engine.generate_weekly_recommendations.return_value = _recs("AAPL")

    result = agent.execute()

    assert "failed_checks" not in result
    assert result["recommendations_generated"] == 1
    comps.engine.generate_weekly_recommendations.assert_called_once()


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_happy_path_generates_and_logs(agent, comps):
    """Full run: closes stale recs, generates new ones, logs lineage, updates track record."""
    comps.engine.review_open_recommendations.return_value = [
        SimpleNamespace(status="closed"),
        SimpleNamespace(status="active"),
    ]
    comps.engine.generate_weekly_recommendations.return_value = _recs("AAPL", "MSFT")

    result = agent.execute()

    assert result["recommendations_generated"] == 2
    assert result["recommendations_closed"] == 1
    assert result["symbols"] == ["AAPL", "MSFT"]
    assert result["track_record_win_rate"] == 0.6
    assert result["track_record_excess_return"] == 0.05

    # Lineage event logged for generated recommendations.
    agent._log_agent_event.assert_called_once()
    assert agent._log_agent_event.call_args[0][0] == EventType.RECOMMENDATION_GENERATED

    # Track record persisted.
    comps.tracker.persist_track_record.assert_called_once()


def test_risk_tolerance_from_env(agent, comps, monkeypatch):
    """HRP_ADVISORY_RISK_TOLERANCE is passed through to the engine."""
    monkeypatch.setenv("HRP_ADVISORY_RISK_TOLERANCE", "5")
    comps.engine.generate_weekly_recommendations.return_value = _recs("AAPL")

    agent.execute()

    _, kwargs = comps.engine.generate_weekly_recommendations.call_args
    assert kwargs["risk_tolerance"] == 5


def test_no_recommendations_skips_lineage_event(agent, comps):
    """With zero recommendations, no RECOMMENDATION_GENERATED event is logged."""
    comps.engine.generate_weekly_recommendations.return_value = []

    result = agent.execute()

    assert result["recommendations_generated"] == 0
    agent._log_agent_event.assert_not_called()


def test_track_record_failure_is_swallowed(agent, comps):
    """A track-record error must not crash the run."""
    comps.engine.generate_weekly_recommendations.return_value = _recs("AAPL")
    comps.tracker.compute_track_record.side_effect = RuntimeError("db down")

    result = agent.execute()

    assert result["recommendations_generated"] == 1
    assert "track_record_win_rate" not in result


# --------------------------------------------------------------------------- #
# Digest
# --------------------------------------------------------------------------- #


def test_digest_sent_when_email_configured(agent, comps, monkeypatch):
    """When NOTIFICATION_EMAIL is set and recs exist, a digest is sent."""
    monkeypatch.setenv("NOTIFICATION_EMAIL", "me@example.com")
    comps.engine.generate_weekly_recommendations.return_value = _recs("AAPL")

    with patch("hrp.advisory.digest.WeeklyDigest") as MD:
        digest = MD.return_value
        digest.generate.return_value = "<html>report</html>"
        digest.send.return_value = True

        result = agent.execute()

    assert result["digest_sent"] is True
    digest.send.assert_called_once()


def test_digest_skipped_without_email(agent, comps):
    """No email configured -> digest is never attempted."""
    comps.engine.generate_weekly_recommendations.return_value = _recs("AAPL")

    result = agent.execute()

    assert result["digest_sent"] is False
