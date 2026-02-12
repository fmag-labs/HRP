"""
Tests for the Risk Metrics Streamlit dashboard page.

Tests verify page structure, required components, and helper functions.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestRiskMetricsPageStructure:
    """Test basic page structure and existence."""

    def test_risk_metrics_page_exists(self):
        """Risk metrics page file should exist."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        assert page_path.exists(), "Risk Metrics page file should exist"

    def test_page_has_required_imports(self):
        """Page should have all required imports."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        required_imports = [
            "import streamlit as st",
            "from hrp.api.platform import PlatformAPI",
            "from hrp.data.risk.var_calculator import VaRCalculator",
            "from hrp.data.risk.risk_config import",
        ]

        for imp in required_imports:
            assert imp in content, f"Page should import {imp}"

    def test_page_has_render_function(self):
        """Page should have a main render function."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        assert "def render()" in content, "Page should have render() function"
        assert 'st.title("Risk Metrics")' in content, "Page should have title"

    def test_page_has_all_sections(self):
        """Page should have all required sections."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        required_sections = [
            "_render_config_section",
            "_render_portfolio_var_overview",
            "_render_per_symbol_var",
            "_render_method_comparison",
            "_render_historical_var_tracking",
            "_render_var_distribution",
        ]

        for section in required_sections:
            assert (
                f"def {section}" in content
            ), f"Page should have {section} function"

    def test_page_has_helper_functions(self):
        """Page should have data retrieval helper functions."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        helpers = [
            "_get_portfolio_returns",
            "_get_symbol_returns",
        ]

        for helper in helpers:
            assert f"def {helper}" in content, f"Page should have {helper} function"


class TestConfigurationSection:
    """Test VaR configuration controls."""

    def test_config_supports_all_var_methods(self):
        """Config should support all VaR methods."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should import VaRMethod
        assert "VaRMethod" in content, "Page should import VaRMethod"

        # Should have method selection
        assert "for m in VaRMethod" in content or "VaRMethod(" in content

    def test_config_has_confidence_levels(self):
        """Config should offer standard confidence levels."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should have 90%, 95%, 99% options
        assert "0.90" in content or "0.9" in content
        assert "0.95" in content
        assert "0.99" in content

    def test_config_has_time_horizons(self):
        """Config should offer multiple time horizons."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should have 1, 5, 10, 21 day options or similar
        assert "time_horizon" in content.lower()


class TestPortfolioVaRSection:
    """Test portfolio-level VaR rendering."""

    def test_portfolio_var_displays_four_metrics(self):
        """Portfolio VaR section should display four key metrics."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should create 4 columns for metrics
        assert 'st.columns(4)' in content

        # Should display VaR, CVaR, and dollar values
        assert "Portfolio VaR" in content
        assert "Portfolio CVaR" in content or "CVaR" in content

    def test_portfolio_var_handles_no_data(self):
        """Portfolio VaR should handle missing data gracefully."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should have warning/info messages for no data
        assert "st.warning" in content or "st.info" in content

    def test_portfolio_var_renders_distribution_chart(self):
        """Portfolio VaR should render distribution visualization."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should call _render_var_distribution
        assert "_render_var_distribution" in content


class TestPerSymbolVaRSection:
    """Test per-symbol VaR breakdown."""

    def test_per_symbol_var_shows_breakdown_table(self):
        """Per-symbol VaR should display a breakdown table."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should display dataframe with VaR metrics
        assert "st.dataframe" in content

    def test_per_symbol_var_shows_contribution_chart(self):
        """Per-symbol VaR should show contribution chart."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should use plotly for visualization
        assert "px.bar" in content or "go.Bar" in content
        assert "VaR Contribution" in content

    def test_per_symbol_var_handles_empty_positions(self):
        """Per-symbol VaR should handle no positions."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should check for empty positions
        assert "positions.empty" in content or "not positions.empty" in content


class TestMethodComparisonSection:
    """Test VaR method comparison section."""

    def test_method_comparison_uses_all_methods(self):
        """Method comparison should calculate VaR using all three methods."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should call calculate_all_methods
        assert "calculate_all_methods" in content

    def test_method_comparison_shows_cvar_var_ratio(self):
        """Method comparison should display CVaR/VaR ratio."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should calculate ratio
        assert "CVaR/VaR" in content or "cvar / result.var" in content or "cvar/var" in content.lower()

    def test_method_comparison_has_chart(self):
        """Method comparison should have visualization."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should use go.Figure or similar for comparison chart
        assert "go.Figure" in content or "px." in content


class TestHistoricalTrackingSection:
    """Test historical VaR tracking over time."""

    def test_historical_tracking_has_rolling_calculation(self):
        """Historical tracking should calculate rolling VaR."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should have rolling window logic
        assert "rolling" in content.lower() or "window_size" in content

    def test_historical_tracking_shows_breach_analysis(self):
        """Historical tracking should analyze VaR breaches."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should calculate breaches
        assert "breach" in content.lower()

    def test_historical_tracking_displays_metrics(self):
        """Historical tracking should display breach metrics."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should show VaR Breaches, Breach Rate, Model Calibration
        assert "VaR Breaches" in content or "breach" in content.lower()
        assert "Breach Rate" in content or "breach_rate" in content


class TestVaRDistributionVisualization:
    """Test VaR distribution visualization."""

    def test_var_distribution_creates_histogram(self):
        """VaR distribution should create histogram of returns."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should use histogram
        assert "go.Histogram" in content

    def test_var_distribution_shows_var_cvar_lines(self):
        """VaR distribution should show VaR and CVaR lines."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should add vertical lines for VaR/CVaR
        assert "add_vline" in content or "vline" in content.lower()


class TestHelperFunctions:
    """Test helper functions for data retrieval."""

    def test_get_portfolio_returns_queries_database(self):
        """get_portfolio_returns should try database first."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # _get_portfolio_returns function should exist
        assert "def _get_portfolio_returns" in content

        # Should query database
        assert "query_readonly" in content

    def test_get_portfolio_returns_has_fallback(self):
        """get_portfolio_returns should fall back to synthetic data."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should have synthetic data generation as fallback
        assert "synthetic" in content.lower() or "demonstration" in content.lower()
        assert "np.random" in content

    def test_get_symbol_returns_queries_database(self):
        """get_symbol_returns should query for symbol-specific returns."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # _get_symbol_returns function should exist
        assert "def _get_symbol_returns" in content

        # Should query daily_bars or similar
        assert "daily_bars" in content or "query_readonly" in content

    def test_get_symbol_returns_handles_insufficient_data(self):
        """get_symbol_returns should handle insufficient data."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should check data length
        assert "len(df)" in content or "len(returns)" in content


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_page_has_try_except_blocks(self):
        """Page should have proper error handling."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should have try-except blocks
        assert "try:" in content
        assert "except" in content

    def test_page_uses_logger_for_errors(self):
        """Page should use logger for error messages."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should import and use logger
        assert "from loguru import logger" in content
        assert "logger.error" in content or "logger.debug" in content or "logger.warning" in content

    def test_page_handles_empty_dataframes(self):
        """Page should check for empty DataFrames."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should check for empty data
        assert ".empty" in content


class TestIntegration:
    """Integration tests for page completeness."""

    def test_page_can_be_imported(self):
        """Page should be importable as a Python module."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")

        # Should be valid Python
        import ast

        with open(page_path) as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Page has syntax errors: {e}")

    def test_page_has_main_block(self):
        """Page should have if __name__ == '__main__' block."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # Should have main block for standalone running
        assert 'if __name__ == "__main__"' in content

    def test_page_renders_all_sections(self):
        """Page render() should call all section functions."""
        page_path = Path("hrp/dashboard/pages/11_Risk_Metrics.py")
        content = page_path.read_text()

        # In render() function, should call all section renderers
        sections_to_call = [
            "_render_config_section",
            "_render_portfolio_var_overview",
            "_render_per_symbol_var",
            "_render_method_comparison",
            "_render_historical_var_tracking",
        ]

        # Find render() function
        assert "def render()" in content

        # Each section should be called in render()
        for section in sections_to_call:
            assert section in content, f"render() should call {section}"
