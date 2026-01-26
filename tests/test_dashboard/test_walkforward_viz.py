"""Tests for walk-forward visualization components."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.ml.validation import FoldResult, WalkForwardConfig


class TestWalkforwardVizImports:
    """Test that visualization module is importable."""

    def test_import_render_functions(self):
        """Test importing render functions."""
        from hrp.dashboard.components.walkforward_viz import (
            render_walkforward_splits,
            render_fold_metrics_heatmap,
            render_fold_comparison_chart,
            render_stability_summary,
        )

        assert render_walkforward_splits is not None
        assert render_fold_metrics_heatmap is not None
        assert render_fold_comparison_chart is not None
        assert render_stability_summary is not None


class TestWalkforwardVizFunctions:
    """Tests for walk-forward visualization functions."""

    @pytest.fixture
    def sample_fold_results(self):
        """Create sample fold results for testing."""
        return [
            FoldResult(
                fold_index=0,
                train_start=date(2015, 1, 1),
                train_end=date(2017, 12, 31),
                test_start=date(2018, 1, 1),
                test_end=date(2018, 12, 31),
                metrics={"mse": 0.001, "mae": 0.02, "r2": 0.15, "ic": 0.05},
                model=None,
                n_train_samples=756,
                n_test_samples=252,
            ),
            FoldResult(
                fold_index=1,
                train_start=date(2015, 1, 1),
                train_end=date(2018, 12, 31),
                test_start=date(2019, 1, 1),
                test_end=date(2019, 12, 31),
                metrics={"mse": 0.0012, "mae": 0.022, "r2": 0.12, "ic": 0.04},
                model=None,
                n_train_samples=1008,
                n_test_samples=252,
            ),
            FoldResult(
                fold_index=2,
                train_start=date(2015, 1, 1),
                train_end=date(2019, 12, 31),
                test_start=date(2020, 1, 1),
                test_end=date(2020, 12, 31),
                metrics={"mse": 0.0015, "mae": 0.025, "r2": 0.10, "ic": 0.03},
                model=None,
                n_train_samples=1260,
                n_test_samples=252,
            ),
        ]

    @pytest.fixture
    def sample_config(self):
        """Create sample walk-forward config."""
        return WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            window_type="expanding",
        )

    def test_render_splits_with_valid_data(self, sample_fold_results, sample_config):
        """Test render_walkforward_splits with valid data."""
        from hrp.dashboard.components.walkforward_viz import render_walkforward_splits

        # Mock streamlit
        with patch("hrp.dashboard.components.walkforward_viz.st") as mock_st, \
             patch("hrp.dashboard.components.walkforward_viz.px") as mock_px:

            mock_fig = MagicMock()
            mock_px.timeline.return_value = mock_fig

            # Should not raise
            render_walkforward_splits(sample_fold_results, sample_config)

            # Verify streamlit was called
            assert mock_st.markdown.called
            assert mock_st.plotly_chart.called

    def test_render_handles_empty_folds(self, sample_config):
        """Test render_walkforward_splits handles empty fold list."""
        from hrp.dashboard.components.walkforward_viz import render_walkforward_splits

        with patch("hrp.dashboard.components.walkforward_viz.st") as mock_st:
            render_walkforward_splits([], sample_config)

            # Should show warning for empty folds
            mock_st.warning.assert_called()

    def test_render_fold_metrics_heatmap(self, sample_fold_results):
        """Test render_fold_metrics_heatmap with valid data."""
        from hrp.dashboard.components.walkforward_viz import render_fold_metrics_heatmap

        with patch("hrp.dashboard.components.walkforward_viz.st") as mock_st, \
             patch("hrp.dashboard.components.walkforward_viz.px") as mock_px:

            mock_fig = MagicMock()
            mock_px.imshow.return_value = mock_fig

            render_fold_metrics_heatmap(sample_fold_results)

            assert mock_st.markdown.called
            assert mock_st.plotly_chart.called

    def test_render_fold_comparison_chart(self, sample_fold_results):
        """Test render_fold_comparison_chart with valid data."""
        from hrp.dashboard.components.walkforward_viz import render_fold_comparison_chart

        with patch("hrp.dashboard.components.walkforward_viz.st") as mock_st, \
             patch("hrp.dashboard.components.walkforward_viz.go") as mock_go:

            mock_fig = MagicMock()
            mock_go.Figure.return_value = mock_fig

            render_fold_comparison_chart(sample_fold_results)

            assert mock_st.markdown.called
            assert mock_st.plotly_chart.called

    def test_render_stability_summary(self, sample_fold_results):
        """Test render_stability_summary with valid data."""
        from hrp.dashboard.components.walkforward_viz import render_stability_summary

        aggregate_metrics = {
            "mean_mse": 0.00123,
            "std_mse": 0.00025,
            "mean_ic": 0.04,
            "std_ic": 0.01,
        }

        with patch("hrp.dashboard.components.walkforward_viz.st") as mock_st:
            # Mock columns to return 4 mock column context managers
            mock_col = MagicMock()
            mock_st.columns.return_value = [mock_col, mock_col, mock_col, mock_col]

            render_stability_summary(sample_fold_results, 0.2, aggregate_metrics)

            assert mock_st.markdown.called
            assert mock_st.metric.called


class TestModuleExports:
    """Test component module exports."""

    def test_import_from_components(self):
        """Test importing from components module."""
        from hrp.dashboard.components import (
            render_walkforward_splits,
            render_fold_metrics_heatmap,
            render_fold_comparison_chart,
            render_stability_summary,
        )

        assert render_walkforward_splits is not None
        assert render_fold_metrics_heatmap is not None
        assert render_fold_comparison_chart is not None
        assert render_stability_summary is not None
