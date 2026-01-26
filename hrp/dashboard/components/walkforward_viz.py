"""
Walk-Forward Validation Visualization Components.

Provides Streamlit components for visualizing walk-forward validation
splits, fold metrics, and stability analysis.

Usage:
    from hrp.dashboard.components.walkforward_viz import (
        render_walkforward_splits,
        render_fold_metrics_heatmap,
        render_fold_comparison_chart,
    )

    # In Streamlit app
    render_walkforward_splits(fold_results, config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

if TYPE_CHECKING:
    from hrp.ml.validation import FoldResult, WalkForwardConfig


def render_walkforward_splits(
    fold_results: list["FoldResult"],
    config: "WalkForwardConfig",
) -> None:
    """
    Render interactive walk-forward split visualization.

    Shows:
    - Timeline of train/test periods for each fold
    - Per-fold metrics (IC, MSE, R2) as annotations
    - Stability score indicator

    Args:
        fold_results: List of FoldResult from walk_forward_validate
        config: WalkForwardConfig used for validation
    """
    if not fold_results:
        st.warning("No fold results to display.")
        return

    st.markdown("#### Walk-Forward Split Timeline")

    # Build data for Gantt-style chart
    timeline_data = []

    for fold in fold_results:
        # Training period
        timeline_data.append({
            "Fold": f"Fold {fold.fold_index + 1}",
            "Type": "Training",
            "Start": pd.Timestamp(fold.train_start),
            "End": pd.Timestamp(fold.train_end),
            "Samples": fold.n_train_samples,
        })
        # Test period
        timeline_data.append({
            "Fold": f"Fold {fold.fold_index + 1}",
            "Type": "Test",
            "Start": pd.Timestamp(fold.test_start),
            "End": pd.Timestamp(fold.test_end),
            "Samples": fold.n_test_samples,
        })

    df = pd.DataFrame(timeline_data)

    # Create timeline chart using Plotly
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Fold",
        color="Type",
        color_discrete_map={"Training": "#636EFA", "Test": "#EF553B"},
        hover_data=["Samples"],
        title=f"Walk-Forward Splits ({config.window_type.title()} Window)",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="",
        yaxis={"categoryorder": "category descending"},
        height=300 + 40 * len(fold_results),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display per-fold metrics summary
    st.markdown("#### Per-Fold Metrics")

    metrics_data = []
    for fold in fold_results:
        metrics_data.append({
            "Fold": fold.fold_index + 1,
            "Train Period": f"{fold.train_start} - {fold.train_end}",
            "Test Period": f"{fold.test_start} - {fold.test_end}",
            "MSE": fold.metrics.get("mse", float("nan")),
            "MAE": fold.metrics.get("mae", float("nan")),
            "R²": fold.metrics.get("r2", float("nan")),
            "IC": fold.metrics.get("ic", float("nan")),
            "Train Samples": fold.n_train_samples,
            "Test Samples": fold.n_test_samples,
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(
        metrics_df.style.format({
            "MSE": "{:.6f}",
            "MAE": "{:.4f}",
            "R²": "{:.4f}",
            "IC": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


def render_fold_metrics_heatmap(
    fold_results: list["FoldResult"],
) -> None:
    """
    Render heatmap of metrics across folds.

    Shows metrics (MSE, MAE, R², IC) for each fold in a heatmap format
    to identify patterns and outlier folds.

    Args:
        fold_results: List of FoldResult from walk_forward_validate
    """
    if not fold_results:
        st.warning("No fold results for heatmap.")
        return

    st.markdown("#### Fold Metrics Heatmap")

    # Build heatmap data
    metrics_list = ["mse", "mae", "r2", "ic"]
    heatmap_data = []

    for fold in fold_results:
        row = {"Fold": f"Fold {fold.fold_index + 1}"}
        for metric in metrics_list:
            value = fold.metrics.get(metric, float("nan"))
            # Normalize MSE and MAE by inverting so higher = better for all
            if metric in ("mse", "mae") and value > 0:
                row[metric.upper()] = -value  # Negate so color scale is consistent
            else:
                row[metric.upper()] = value
        heatmap_data.append(row)

    df = pd.DataFrame(heatmap_data)
    df = df.set_index("Fold")

    # Create heatmap
    fig = px.imshow(
        df.T,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Metrics Across Folds (Green = Better)",
        labels={"x": "Fold", "y": "Metric", "color": "Value"},
    )

    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: MSE and MAE are negated so green indicates better performance "
        "(lower error) across all metrics."
    )


def render_fold_comparison_chart(
    fold_results: list["FoldResult"],
) -> None:
    """
    Render bar chart comparing fold performance.

    Shows side-by-side comparison of key metrics across folds.

    Args:
        fold_results: List of FoldResult from walk_forward_validate
    """
    if not fold_results:
        st.warning("No fold results for comparison.")
        return

    st.markdown("#### Fold Performance Comparison")

    # Build comparison data
    comparison_data = []
    for fold in fold_results:
        comparison_data.append({
            "Fold": f"Fold {fold.fold_index + 1}",
            "IC": fold.metrics.get("ic", 0),
            "R²": fold.metrics.get("r2", 0),
        })

    df = pd.DataFrame(comparison_data)

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Information Coefficient (IC)",
        x=df["Fold"],
        y=df["IC"],
        marker_color="#636EFA",
    ))

    fig.add_trace(go.Bar(
        name="R² Score",
        x=df["Fold"],
        y=df["R²"],
        marker_color="#EF553B",
    ))

    fig.update_layout(
        barmode="group",
        xaxis_title="",
        yaxis_title="Value",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)


def render_stability_summary(
    fold_results: list["FoldResult"],
    stability_score: float,
    aggregate_metrics: dict[str, float],
) -> None:
    """
    Render summary metrics for walk-forward stability.

    Shows key stability indicators and pass/fail status.

    Args:
        fold_results: List of FoldResult
        stability_score: Coefficient of variation of MSE
        aggregate_metrics: Aggregated metrics across folds
    """
    st.markdown("#### Stability Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Stability score with color indicator
        is_stable = stability_score <= 1.0
        status_color = "green" if is_stable else "red"
        status_text = "✅ STABLE" if is_stable else "⚠️ UNSTABLE"
        st.metric(
            "Stability Score",
            f"{stability_score:.3f}",
            delta=status_text,
            delta_color="off",
        )

    with col2:
        mean_ic = aggregate_metrics.get("mean_ic", float("nan"))
        st.metric(
            "Mean IC",
            f"{mean_ic:.4f}",
            help="Mean Information Coefficient across folds",
        )

    with col3:
        mean_mse = aggregate_metrics.get("mean_mse", float("nan"))
        st.metric(
            "Mean MSE",
            f"{mean_mse:.6f}",
            help="Mean Squared Error across folds",
        )

    with col4:
        std_mse = aggregate_metrics.get("std_mse", float("nan"))
        st.metric(
            "MSE Std Dev",
            f"{std_mse:.6f}",
            help="Standard deviation of MSE (lower = more stable)",
        )

    # Interpretation
    if stability_score <= 0.5:
        st.success(
            "🎯 **Excellent stability**: Model performance is highly consistent across folds. "
            "Low risk of overfitting."
        )
    elif stability_score <= 1.0:
        st.info(
            "✓ **Good stability**: Model performance is reasonably consistent. "
            "Consider monitoring for regime changes."
        )
    else:
        st.warning(
            "⚠️ **Poor stability**: High variability across folds. "
            "Consider simplifying the model or adding regularization."
        )
