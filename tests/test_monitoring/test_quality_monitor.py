"""Tests for the automated data-quality monitor (hrp.monitoring.quality_monitor).

The QualityReportGenerator and QualityAlertManager dependencies are patched so
these tests run without a database or email backend.
"""

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hrp.monitoring.quality_monitor import (
    DataQualityMonitor,
    MonitoringResult,
    MonitoringThresholds,
)


def _report(
    health_score=95.0,
    passed=True,
    critical_issues=0,
    warning_issues=0,
    total_issues=0,
    results=None,
    critical_list=None,
):
    """Build a stand-in QualityReport."""
    rep = Mock()
    rep.health_score = health_score
    rep.passed = passed
    rep.critical_issues = critical_issues
    rep.warning_issues = warning_issues
    rep.total_issues = total_issues
    rep.results = results or []
    rep.get_critical_issues.return_value = critical_list or []
    return rep


@pytest.fixture
def monitor():
    """A monitor with report generator / alert manager patched out."""
    with patch(
        "hrp.monitoring.quality_monitor.QualityReportGenerator"
    ) as MGen, patch("hrp.monitoring.quality_monitor.QualityAlertManager") as MAlert:
        m = DataQualityMonitor(send_alerts=True)
        m._gen = MGen.return_value
        m._alert = MAlert.return_value
        yield m


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


def test_thresholds_defaults():
    t = MonitoringThresholds()
    assert t.health_score_warning == 90.0
    assert t.health_score_critical == 70.0
    assert t.freshness_warning_days == 3
    assert t.freshness_critical_days == 5
    assert t.anomaly_count_critical == 100


def test_monitoring_result_to_dict_roundtrip():
    ts = datetime(2026, 6, 27, 8, 0, 0)
    result = MonitoringResult(
        timestamp=ts,
        health_score=88.0,
        passed=True,
        critical_issues=0,
        warning_issues=2,
        total_issues=2,
        trend="declining",
    )
    d = result.to_dict()
    assert d["timestamp"] == ts.isoformat()
    assert d["health_score"] == 88.0
    assert d["trend"] == "declining"
    assert d["alerts_sent"] == {}


# --------------------------------------------------------------------------- #
# Trend calculation
# --------------------------------------------------------------------------- #


def test_trend_insufficient_data_is_stable(monitor):
    monitor._gen.get_health_trend.return_value = [{"health_score": 90}]
    assert monitor._calculate_trend(date(2026, 6, 27)) == "stable"


def test_trend_improving(monitor):
    monitor._gen.get_health_trend.return_value = [
        {"health_score": s} for s in [70, 72, 71, 85, 90, 92]
    ]
    assert monitor._calculate_trend(date(2026, 6, 27)) == "improving"


def test_trend_declining(monitor):
    monitor._gen.get_health_trend.return_value = [
        {"health_score": s} for s in [95, 94, 93, 80, 75, 70]
    ]
    assert monitor._calculate_trend(date(2026, 6, 27)) == "declining"


def test_trend_stable_within_band(monitor):
    monitor._gen.get_health_trend.return_value = [
        {"health_score": s} for s in [90, 91, 89, 90, 91, 90]
    ]
    assert monitor._calculate_trend(date(2026, 6, 27)) == "stable"


def test_trend_handles_generator_error(monitor):
    monitor._gen.get_health_trend.side_effect = RuntimeError("boom")
    assert monitor._calculate_trend(date(2026, 6, 27)) == "stable"


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #


def test_recommendations_critical_health(monitor):
    report = _report(health_score=60.0, critical_issues=0)
    recs = monitor._generate_recommendations(report)
    assert any("URGENT" in r for r in recs)


def test_recommendations_warning_health(monitor):
    report = _report(health_score=85.0)
    recs = monitor._generate_recommendations(report)
    assert any("below target" in r for r in recs)


def test_recommendations_group_critical_issues(monitor):
    critical = [
        SimpleNamespace(check_name="gaps"),
        SimpleNamespace(check_name="gaps"),
        SimpleNamespace(check_name="stale_data"),
    ]
    report = _report(health_score=95.0, critical_issues=3, critical_list=critical)
    recs = monitor._generate_recommendations(report)
    assert any("2 critical gaps" in r for r in recs)
    assert any("1 critical stale_data" in r for r in recs)


def test_recommendations_high_warning_count(monitor):
    report = _report(health_score=95.0, warning_issues=80)
    recs = monitor._generate_recommendations(report)
    assert any("High warning count" in r for r in recs)


# --------------------------------------------------------------------------- #
# run_daily_check
# --------------------------------------------------------------------------- #


def test_run_daily_check_builds_result(monitor):
    report = _report(health_score=95.0, passed=True, warning_issues=1, total_issues=1)
    monitor._gen.generate_report.return_value = report
    monitor._gen.store_report.return_value = 42
    monitor._gen.get_health_trend.return_value = []
    monitor.send_alerts = False  # skip alerting branch

    with patch(
        "hrp.utils.calendar.get_previous_trading_day", return_value=date(2026, 6, 26)
    ), patch("hrp.api.platform.PlatformAPI") as MAPI:
        MAPI.return_value.fetchone_readonly.return_value = None
        result = monitor.run_daily_check(date(2026, 6, 27))

    assert isinstance(result, MonitoringResult)
    assert result.health_score == 95.0
    assert result.passed is True
    monitor._gen.generate_report.assert_called_once_with(date(2026, 6, 26))
    monitor._gen.store_report.assert_called_once()


# --------------------------------------------------------------------------- #
# Alerting
# --------------------------------------------------------------------------- #


def test_check_and_alert_critical_health_sends_alert(monitor):
    report = _report(health_score=60.0, critical_issues=2, critical_list=[Mock()])
    monitor._alert.send_critical_alert.return_value = True

    with patch("hrp.api.platform.PlatformAPI") as MAPI:
        MAPI.return_value.fetchone_readonly.return_value = None
        sent = monitor._check_and_alert(report, date(2026, 6, 27))

    assert sent["health_critical"] is True
    assert sent["critical_issues"] is True


def test_check_and_alert_warning_health_sends_summary(monitor):
    report = _report(health_score=85.0, critical_issues=0)
    monitor._alert.send_daily_summary.return_value = True

    with patch("hrp.api.platform.PlatformAPI") as MAPI:
        MAPI.return_value.fetchone_readonly.return_value = None
        sent = monitor._check_and_alert(report, date(2026, 6, 27))

    assert sent["health_warning"] is True
    monitor._alert.send_daily_summary.assert_called_once()


def test_check_and_alert_healthy_no_alerts(monitor):
    report = _report(health_score=98.0, critical_issues=0, total_issues=0)

    with patch("hrp.api.platform.PlatformAPI") as MAPI:
        MAPI.return_value.fetchone_readonly.return_value = None
        sent = monitor._check_and_alert(report, date(2026, 6, 27))

    assert not any(sent.values())
