"""Tests for Sharpe decay visualization components."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestSharpeDecayVizImports:
    """Test that Sharpe decay visualization module is importable."""

    def test_import_render_functions(self):
        """Test importing render functions."""
        from hrp.dashboard.components.sharpe_decay_viz import (
            render_sharpe_decay_heatmap,
            render_generalization_summary,
            render_parameter_sensitivity_chart,
            render_top_bottom_params,
        )

        assert render_sharpe_decay_heatmap is not None
        assert render_generalization_summary is not None
        assert render_parameter_sensitivity_chart is not None
        assert render_top_bottom_params is not None


class TestSharpeDecayVizFunctions:
    """Tests for Sharpe decay visualization functions."""

    @pytest.fixture
    def sample_sweep_result(self):
        """Create sample sweep result for testing."""
        from hrp.research.parameter_sweep import SweepResult

        # Create sample results DataFrame
        results_df = pd.DataFrame({
            "combo_idx": [0, 1, 2, 3, 4, 5],
            "fast_period": [10, 10, 15, 15, 20, 20],
            "slow_period": [20, 30, 25, 35, 30, 40],
            "train_sharpe_fold_0": [1.0, 1.2, 0.9, 1.1, 0.8, 1.0],
            "train_sharpe_fold_1": [0.9, 1.1, 0.8, 1.0, 0.7, 0.9],
            "test_sharpe_fold_0": [0.8, 1.1, 0.85, 0.95, 0.75, 0.85],
            "test_sharpe_fold_1": [0.7, 1.0, 0.75, 0.9, 0.65, 0.8],
            "train_sharpe_agg": [0.95, 1.15, 0.85, 1.05, 0.75, 0.95],
            "test_sharpe_agg": [0.75, 1.05, 0.8, 0.925, 0.7, 0.825],
            "sharpe_diff_agg": [-0.2, -0.1, -0.05, -0.125, -0.05, -0.125],
        })

        train_sharpe_matrix = results_df[["train_sharpe_fold_0", "train_sharpe_fold_1"]]
        test_sharpe_matrix = results_df[["test_sharpe_fold_0", "test_sharpe_fold_1"]]
        sharpe_diff_matrix = test_sharpe_matrix.values - train_sharpe_matrix.values

        return SweepResult(
            results_df=results_df,
            best_params={"fast_period": 10, "slow_period": 30},
            best_metrics={"train_sharpe": 1.15, "test_sharpe": 1.05, "sharpe_diff": -0.1},
            train_sharpe_matrix=train_sharpe_matrix,
            test_sharpe_matrix=test_sharpe_matrix,
            sharpe_diff_matrix=pd.DataFrame(sharpe_diff_matrix),
            sharpe_diff_median=pd.Series(results_df["sharpe_diff_agg"]),
            constraint_violations=2,
            execution_time_seconds=5.5,
            generalization_score=0.0,  # All negative diffs
        )

    def test_render_sharpe_decay_heatmap(self, sample_sweep_result):
        """Test render_sharpe_decay_heatmap with valid data."""
        from hrp.dashboard.components.sharpe_decay_viz import render_sharpe_decay_heatmap

        with patch("hrp.dashboard.components.sharpe_decay_viz.st") as mock_st, \
             patch("hrp.dashboard.components.sharpe_decay_viz.px") as mock_px:

            mock_fig = MagicMock()
            mock_px.imshow.return_value = mock_fig

            render_sharpe_decay_heatmap(
                sample_sweep_result,
                param_x="fast_period",
                param_y="slow_period",
            )

            assert mock_st.markdown.called
            assert mock_st.plotly_chart.called

    def test_heatmap_centers_at_zero(self, sample_sweep_result):
        """Test heatmap colorscale centers at zero."""
        from hrp.dashboard.components.sharpe_decay_viz import render_sharpe_decay_heatmap

        with patch("hrp.dashboard.components.sharpe_decay_viz.st") as mock_st, \
             patch("hrp.dashboard.components.sharpe_decay_viz.px") as mock_px:

            mock_fig = MagicMock()
            mock_px.imshow.return_value = mock_fig

            render_sharpe_decay_heatmap(
                sample_sweep_result,
                param_x="fast_period",
                param_y="slow_period",
            )

            # Check that imshow was called with color_continuous_midpoint=0
            call_kwargs = mock_px.imshow.call_args[1]
            assert call_kwargs.get("color_continuous_midpoint") == 0

    def test_generalization_summary_metrics(self, sample_sweep_result):
        """Test render_generalization_summary displays correct metrics."""
        from hrp.dashboard.components.sharpe_decay_viz import render_generalization_summary

        with patch("hrp.dashboard.components.sharpe_decay_viz.st") as mock_st:
            # Mock columns context manager
            mock_col = MagicMock()
            mock_st.columns.return_value = [mock_col, mock_col, mock_col]

            render_generalization_summary(sample_sweep_result)

            # Should call metric multiple times
            assert mock_st.metric.called

    def test_handles_triangular_constraint_data(self):
        """Test heatmap handles triangular data from constraints."""
        from hrp.research.parameter_sweep import SweepResult
        from hrp.dashboard.components.sharpe_decay_viz import render_sharpe_decay_heatmap

        # Create triangular data (slow > fast constraint)
        results_df = pd.DataFrame({
            "fast_period": [10, 10, 10, 15, 15, 20],
            "slow_period": [20, 30, 40, 30, 40, 40],
            "sharpe_diff_agg": [0.1, 0.05, -0.1, 0.02, -0.05, -0.08],
            "train_sharpe_agg": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "test_sharpe_agg": [1.1, 1.05, 0.9, 1.02, 0.95, 0.92],
        })

        sweep_result = SweepResult(
            results_df=results_df,
            best_params={"fast_period": 10, "slow_period": 20},
            best_metrics={"train_sharpe": 1.0, "test_sharpe": 1.1},
            train_sharpe_matrix=pd.DataFrame(),
            test_sharpe_matrix=pd.DataFrame(),
            sharpe_diff_matrix=pd.DataFrame(),
            sharpe_diff_median=pd.Series(results_df["sharpe_diff_agg"]),
            constraint_violations=0,
            execution_time_seconds=1.0,
            generalization_score=0.5,
        )

        with patch("hrp.dashboard.components.sharpe_decay_viz.st") as mock_st, \
             patch("hrp.dashboard.components.sharpe_decay_viz.px") as mock_px:

            mock_fig = MagicMock()
            mock_px.imshow.return_value = mock_fig

            # Should not raise with triangular data
            render_sharpe_decay_heatmap(
                sweep_result,
                param_x="fast_period",
                param_y="slow_period",
            )

            assert mock_st.plotly_chart.called

    def test_parameter_sensitivity_chart(self, sample_sweep_result):
        """Test render_parameter_sensitivity_chart."""
        from hrp.dashboard.components.sharpe_decay_viz import render_parameter_sensitivity_chart

        with patch("hrp.dashboard.components.sharpe_decay_viz.st") as mock_st, \
             patch("hrp.dashboard.components.sharpe_decay_viz.go") as mock_go:

            mock_fig = MagicMock()
            mock_go.Figure.return_value = mock_fig

            render_parameter_sensitivity_chart(sample_sweep_result, "fast_period")

            assert mock_st.markdown.called
            assert mock_st.plotly_chart.called

    def test_top_bottom_params(self, sample_sweep_result):
        """Test render_top_bottom_params."""
        from hrp.dashboard.components.sharpe_decay_viz import render_top_bottom_params

        with patch("hrp.dashboard.components.sharpe_decay_viz.st") as mock_st:
            mock_col = MagicMock()
            mock_st.columns.return_value = [mock_col, mock_col]

            render_top_bottom_params(sample_sweep_result, n_show=3)

            assert mock_st.markdown.called
            assert mock_st.dataframe.called


class TestModuleExports:
    """Test component module exports."""

    def test_import_from_components(self):
        """Test importing from components module."""
        from hrp.dashboard.components import (
            render_sharpe_decay_heatmap,
            render_generalization_summary,
            render_parameter_sensitivity_chart,
            render_top_bottom_params,
        )

        assert render_sharpe_decay_heatmap is not None
        assert render_generalization_summary is not None
        assert render_parameter_sensitivity_chart is not None
        assert render_top_bottom_params is not None
