"""Tests for portfolio risk limits."""

import pytest
import pandas as pd
import numpy as np

from hrp.risk.limits import RiskLimits, ValidationReport, LimitViolation


class TestRiskLimits:
    """Tests for RiskLimits configuration."""

    def test_default_limits(self):
        """Default limits are conservative institutional."""
        limits = RiskLimits()
        assert limits.max_position_pct == 0.05
        assert limits.min_position_pct == 0.01
        assert limits.max_position_adv_pct == 0.10
        assert limits.max_sector_pct == 0.25
        assert limits.max_unknown_sector_pct == 0.10
        assert limits.max_gross_exposure == 1.00
        assert limits.min_gross_exposure == 0.80
        assert limits.max_turnover_pct == 0.20
        assert limits.max_top_n_concentration == 0.40
        assert limits.top_n_for_concentration == 5
        assert limits.min_adv_dollars == 1_000_000

    def test_custom_limits(self):
        """Custom limits override defaults."""
        limits = RiskLimits(
            max_position_pct=0.10,
            max_sector_pct=0.30,
        )
        assert limits.max_position_pct == 0.10
        assert limits.max_sector_pct == 0.30
        # Other defaults unchanged
        assert limits.min_position_pct == 0.01


class TestValidationReport:
    """Tests for ValidationReport."""

    def test_empty_report_is_valid(self):
        """Report with no violations is valid."""
        report = ValidationReport(violations=[], clips=[], warnings=[])
        assert report.is_valid
        assert len(report.violations) == 0

    def test_report_with_violations_invalid(self):
        """Report with violations is invalid."""
        violation = LimitViolation(
            limit_name="max_position_pct",
            symbol="AAPL",
            limit_value=0.05,
            actual_value=0.08,
            action="rejected",
        )
        report = ValidationReport(violations=[violation], clips=[], warnings=[])
        assert not report.is_valid
        assert len(report.violations) == 1

    def test_report_with_clips_is_valid(self):
        """Report with only clips (no violations) is valid."""
        clip = LimitViolation(
            limit_name="max_position_pct",
            symbol="AAPL",
            limit_value=0.05,
            actual_value=0.08,
            action="clipped",
        )
        report = ValidationReport(violations=[], clips=[clip], warnings=[])
        assert report.is_valid
        assert len(report.clips) == 1


class TestLimitViolation:
    """Tests for LimitViolation."""

    def test_violation_fields(self):
        """LimitViolation stores all fields."""
        violation = LimitViolation(
            limit_name="max_sector_pct",
            symbol="AAPL",
            limit_value=0.25,
            actual_value=0.30,
            action="clipped",
            details="Technology sector",
        )
        assert violation.limit_name == "max_sector_pct"
        assert violation.symbol == "AAPL"
        assert violation.limit_value == 0.25
        assert violation.actual_value == 0.30
        assert violation.action == "clipped"
        assert violation.details == "Technology sector"
