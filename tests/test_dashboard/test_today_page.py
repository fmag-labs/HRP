"""Tests for the consumer-mode Today dashboard helpers."""

from datetime import date, datetime, timedelta

from hrp.dashboard.pages import today


def test_freshness_no_data():
    label, help_text = today._freshness(None)

    assert label == "No data"
    assert "No price data" in help_text


def test_freshness_recent_date():
    label, help_text = today._freshness(date.today())

    assert label == "Fresh"
    assert str(date.today()) in help_text


def test_freshness_stale_date():
    label, _ = today._freshness(date.today() - timedelta(days=10))

    assert label == "Stale"


def test_format_datetime_today():
    value = datetime.combine(date.today(), datetime.min.time()).replace(hour=9, minute=30)

    assert today._format_datetime(value) == "09:30"


def test_money_and_percent_formatting():
    assert today._money(123.456) == "$123.46"
    assert today._pct(0.075) == "7.5%"
    assert today._money(None) == "-"
    assert today._pct(None) == "-"
