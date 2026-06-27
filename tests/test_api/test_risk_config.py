"""Tests for the Risk Configuration API (hrp.api.risk_config).

Covers limit retrieval/caching, persistence, validation rules and the
impact-preview path. The PlatformAPI and RiskManager dependencies are mocked
so no database or risk engine is required.
"""

import json
from unittest.mock import Mock, patch

import pytest

from hrp.api.risk_config import ImpactPreview, RiskConfigAPI, RiskLimits


@pytest.fixture
def mock_api():
    return Mock()


@pytest.fixture
def risk_api(mock_api):
    return RiskConfigAPI(mock_api)


def _valid_limits(**overrides) -> RiskLimits:
    base = dict(
        max_drawdown=0.20,
        max_drawdown_duration_days=100,
        max_position_correlation=0.6,
        max_sector_exposure=0.30,
        max_single_position=0.10,
        min_diversification=10,
        target_positions=20,
    )
    base.update(overrides)
    return RiskLimits(**base)


# --------------------------------------------------------------------------- #
# get_limits
# --------------------------------------------------------------------------- #


def test_get_limits_returns_defaults_when_db_empty(risk_api, mock_api):
    mock_api._db.fetchone.return_value = None

    limits = risk_api.get_limits(use_cache=False)

    assert limits == RiskConfigAPI.DEFAULT_LIMITS


def test_get_limits_reads_from_db(risk_api, mock_api):
    stored = _valid_limits(max_drawdown=0.15)
    mock_api._db.fetchone.return_value = (json.dumps(stored.__dict__),)

    limits = risk_api.get_limits(use_cache=False)

    assert limits == stored
    assert limits.max_drawdown == 0.15


def test_get_limits_uses_cache(risk_api, mock_api):
    mock_api._db.fetchone.return_value = None
    first = risk_api.get_limits(use_cache=False)  # populates cache

    mock_api._db.fetchone.reset_mock()
    second = risk_api.get_limits(use_cache=True)

    assert second == first
    mock_api._db.fetchone.assert_not_called()


def test_get_limits_db_error_returns_defaults(risk_api, mock_api):
    mock_api._db.fetchone.side_effect = RuntimeError("db unavailable")

    limits = risk_api.get_limits(use_cache=False)

    assert limits == RiskConfigAPI.DEFAULT_LIMITS


# --------------------------------------------------------------------------- #
# set_limits
# --------------------------------------------------------------------------- #


def test_set_limits_persists_and_caches(risk_api, mock_api):
    limits = _valid_limits()

    ok = risk_api.set_limits(limits, actor="user")

    assert ok is True
    mock_api._db.execute.assert_called_once()
    # Cache now holds the new limits (no DB read needed).
    assert risk_api.get_limits(use_cache=True) == limits


def test_set_limits_invalid_returns_false_without_write(risk_api, mock_api):
    bad = _valid_limits(max_drawdown=1.5)  # > 1.0

    ok = risk_api.set_limits(bad)

    assert ok is False
    mock_api._db.execute.assert_not_called()


def test_reset_to_defaults_writes_defaults(risk_api, mock_api):
    result = risk_api.reset_to_defaults(actor="user")

    assert result == RiskConfigAPI.DEFAULT_LIMITS
    mock_api._db.execute.assert_called_once()


# --------------------------------------------------------------------------- #
# Validation rules
# --------------------------------------------------------------------------- #


def test_default_limits_pass_validation(risk_api):
    # Should not raise.
    risk_api._validate_limits(RiskConfigAPI.DEFAULT_LIMITS)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_drawdown": 0.0},
        {"max_drawdown": 1.5},
        {"max_drawdown_duration_days": 0},
        {"max_position_correlation": -0.1},
        {"max_position_correlation": 1.1},
        {"max_sector_exposure": 0.0},
        {"max_sector_exposure": 1.5},
        {"max_single_position": 0.0},
        {"max_single_position": 1.5},
        {"min_diversification": 0},
        # target_positions < min_diversification
        {"target_positions": 5, "min_diversification": 10},
        # single position exceeds sector exposure
        {"max_single_position": 0.40, "max_sector_exposure": 0.30},
    ],
)
def test_validate_limits_rejects_bad_values(risk_api, overrides):
    with pytest.raises(ValueError):
        risk_api._validate_limits(_valid_limits(**overrides))


# --------------------------------------------------------------------------- #
# preview_impact
# --------------------------------------------------------------------------- #


def test_preview_impact_no_hypotheses(risk_api):
    with patch("hrp.agents.risk_manager.RiskManager") as MRM:
        MRM.return_value._get_hypotheses_to_assess.return_value = []

        preview = risk_api.preview_impact(_valid_limits())

    assert isinstance(preview, ImpactPreview)
    assert preview.total_hypotheses == 0
    assert preview.hypotheses_passed == 0
    assert preview.hypotheses_vetoed == 0
    assert preview.veto_details == []


def test_preview_impact_counts_passed_and_vetoed(risk_api):
    hypotheses = [
        {"hypothesis_id": "HYP-2026-001", "title": "Clean", "metadata": "{}"},
        {"hypothesis_id": "HYP-2026-002", "title": "Risky", "metadata": "{}"},
    ]
    critical_veto = Mock(severity="critical", veto_reason="dd too high", veto_type="drawdown")

    with patch("hrp.agents.risk_manager.RiskManager") as MRM:
        rm = MRM.return_value
        rm._get_hypotheses_to_assess.return_value = hypotheses
        rm._get_experiment_metrics.return_value = {}
        # First hypothesis clean, second vetoed on drawdown.
        rm._check_drawdown_risk.side_effect = [None, critical_veto]
        rm._check_concentration_risk.return_value = ([], None)
        rm._check_correlation_risk.return_value = None
        rm._check_risk_limits.return_value = []

        preview = risk_api.preview_impact(_valid_limits())

    assert preview.total_hypotheses == 2
    assert preview.hypotheses_passed == 1
    assert preview.hypotheses_vetoed == 1
    assert preview.veto_details[0]["hypothesis_id"] == "HYP-2026-002"
