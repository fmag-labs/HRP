"""
Sharpe Decay Visualization Components.

Provides Streamlit components for visualizing Sharpe ratio decay
heatmaps inspired by VectorBT PRO patterns.

Key visualization: Blue regions = good generalization (test > train),
Red regions = overfitting (test < train).

Usage:
    from hrp.dashboard.components.sharpe_decay_viz import (
        render_sharpe_decay_heatmap,
        render_generalization_summary,
        render_parameter_sensitivity_chart,
    )

    # In Streamlit app
    render_sharpe_decay_heatmap(sweep_result, "fast_period", "slow_period")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

if TYPE_CHECKING:
    from hrp.research.parameter_sweep import SweepResult


def render_sharpe_decay_heatmap(
    sweep_result: "SweepResult",
    param_x: str,
    param_y: str,
    colorscale: str = "RdBu",
) -> None:
    """
    Render Sharpe ratio decay heatmap (VectorBT PRO style).

    Shows test_sharpe - train_sharpe across parameter combinations.
    Blue = positive (good generalization), Red = negative (overfitting).

    Args:
        sweep_result: Result from parallel_parameter_sweep
        param_x: Parameter for X-axis (e.g., "fast_period")
        param_y: Parameter for Y-axis (e.g., "slow_period")
        colorscale: Plotly colorscale (RdBu recommended for diverging)
    """
    st.markdown("#### Sharpe Ratio Decay Heatmap")
    st.caption(
        "🔵 Blue = Good Generalization (Test > Train) | "
        "🔴 Red = Overfitting (Test < Train)"
    )

    results_df = sweep_result.results_df

    # Check if parameters exist in results
    if param_x not in results_df.columns or param_y not in results_df.columns:
        st.error(f"Parameters '{param_x}' or '{param_y}' not found in results.")
        return

    if "sharpe_diff_agg" not in results_df.columns:
        st.error("Sharpe diff not computed. Run sweep with n_folds >= 2.")
        return

    # Pivot to create heatmap matrix
    try:
        heatmap_data = results_df.pivot_table(
            values="sharpe_diff_agg",
            index=param_y,
            columns=param_x,
            aggfunc="mean",
        )
    except Exception as e:
        st.error(f"Could not create heatmap: {e}")
        return

    if heatmap_data.empty:
        st.warning("No data available for heatmap.")
        return

    # Create heatmap with centered colorscale at 0
    fig = px.imshow(
        heatmap_data,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale=colorscale,
        color_continuous_midpoint=0,  # Center at zero
        labels={"x": param_x, "y": param_y, "color": "Sharpe Diff"},
        title=f"Sharpe Ratio Decay: Test - Train",
        aspect="auto",
    )

    fig.update_layout(
        height=400 + 15 * len(heatmap_data),
        xaxis_title=param_x,
        yaxis_title=param_y,
    )

    # Add annotations for values
    for i, y_val in enumerate(heatmap_data.index):
        for j, x_val in enumerate(heatmap_data.columns):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                fig.add_annotation(
                    x=x_val,
                    y=y_val,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(
                        size=10,
                        color="white" if abs(value) > 0.3 else "black",
                    ),
                )

    st.plotly_chart(fig, use_container_width=True)

    # Show color scale interpretation
    with st.expander("How to Read This Heatmap"):
        st.markdown("""
        **Interpretation Guide:**

        - **Blue cells (positive values)**: Test Sharpe > Train Sharpe
          - These parameters generalize well to unseen data
          - Less likely to be overfit

        - **Red cells (negative values)**: Test Sharpe < Train Sharpe
          - Performance degrades on test data
          - Potential overfitting - avoid these parameters

        - **White/light cells (near zero)**: Similar train/test performance
          - Moderate generalization

        **Best Practice:** Choose parameters from blue regions for more
        robust out-of-sample performance.
        """)


def render_generalization_summary(
    sweep_result: "SweepResult",
) -> None:
    """
    Render summary metrics for parameter generalization.

    Shows:
    - % of parameter combos that generalize (test >= train)
    - Best generalizing parameters
    - Worst overfitting parameters

    Args:
        sweep_result: Result from parallel_parameter_sweep
    """
    st.markdown("#### Generalization Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Generalization score
        gen_score = sweep_result.generalization_score
        if gen_score >= 0.5:
            delta_color = "normal"
            status = "Good"
        else:
            delta_color = "inverse"
            status = "Poor"

        st.metric(
            "Generalization Score",
            f"{gen_score:.1%}",
            delta=status,
            delta_color=delta_color,
            help="Percentage of parameter combos where test Sharpe >= train Sharpe",
        )

    with col2:
        # Best params
        best_params_str = ", ".join(
            f"{k}={v}" for k, v in sweep_result.best_params.items()
        )
        st.metric(
            "Best Parameters",
            best_params_str[:30] + "..." if len(best_params_str) > 30 else best_params_str,
            help=f"Full: {best_params_str}",
        )

    with col3:
        # Number of combos tested
        st.metric(
            "Combinations Tested",
            len(sweep_result.results_df),
            delta=f"{sweep_result.constraint_violations} violated",
            delta_color="off",
            help="Total valid parameter combinations evaluated",
        )

    # Additional metrics row
    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Best Test Sharpe",
            f"{sweep_result.best_metrics.get('test_sharpe', 0):.3f}",
        )

    with col5:
        st.metric(
            "Best Train Sharpe",
            f"{sweep_result.best_metrics.get('train_sharpe', 0):.3f}",
        )

    with col6:
        st.metric(
            "Execution Time",
            f"{sweep_result.execution_time_seconds:.1f}s",
        )

    # Interpretation
    if gen_score >= 0.7:
        st.success(
            "🎯 **Strong generalization**: Most parameter combinations show good "
            "out-of-sample performance. Strategy is robust."
        )
    elif gen_score >= 0.4:
        st.info(
            "✓ **Moderate generalization**: Some overfitting detected. "
            "Consider using parameters from blue regions of the heatmap."
        )
    else:
        st.warning(
            "⚠️ **Weak generalization**: Significant overfitting across parameters. "
            "Strategy may be data-mined. Consider simplifying or using different features."
        )


def render_parameter_sensitivity_chart(
    sweep_result: "SweepResult",
    param_name: str,
) -> None:
    """
    Render sensitivity analysis for a single parameter.

    Shows how Sharpe diff varies as one parameter changes,
    holding others at their median values.

    Args:
        sweep_result: Result from parallel_parameter_sweep
        param_name: Parameter to analyze
    """
    st.markdown(f"#### Sensitivity: {param_name}")

    results_df = sweep_result.results_df

    if param_name not in results_df.columns:
        st.error(f"Parameter '{param_name}' not found in results.")
        return

    if "sharpe_diff_agg" not in results_df.columns:
        st.error("Sharpe diff not computed.")
        return

    # Group by parameter and compute mean sharpe diff
    sensitivity_data = results_df.groupby(param_name).agg({
        "sharpe_diff_agg": ["mean", "std"],
        "test_sharpe_agg": "mean",
        "train_sharpe_agg": "mean",
    }).reset_index()

    sensitivity_data.columns = [
        param_name, "sharpe_diff_mean", "sharpe_diff_std",
        "test_sharpe", "train_sharpe"
    ]

    # Create dual-axis chart
    fig = go.Figure()

    # Sharpe diff bars
    colors = ["#636EFA" if v >= 0 else "#EF553B" for v in sensitivity_data["sharpe_diff_mean"]]

    fig.add_trace(go.Bar(
        name="Sharpe Diff (Test - Train)",
        x=sensitivity_data[param_name].astype(str),
        y=sensitivity_data["sharpe_diff_mean"],
        marker_color=colors,
        error_y=dict(
            type="data",
            array=sensitivity_data["sharpe_diff_std"],
            visible=True,
        ),
    ))

    # Test and train Sharpe lines
    fig.add_trace(go.Scatter(
        name="Test Sharpe",
        x=sensitivity_data[param_name].astype(str),
        y=sensitivity_data["test_sharpe"],
        mode="lines+markers",
        line=dict(color="#00CC96", width=2),
        yaxis="y2",
    ))

    fig.add_trace(go.Scatter(
        name="Train Sharpe",
        x=sensitivity_data[param_name].astype(str),
        y=sensitivity_data["train_sharpe"],
        mode="lines+markers",
        line=dict(color="#AB63FA", width=2, dash="dash"),
        yaxis="y2",
    ))

    fig.update_layout(
        xaxis_title=param_name,
        yaxis_title="Sharpe Diff",
        yaxis2=dict(
            title="Sharpe Ratio",
            overlaying="y",
            side="right",
        ),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="relative",
    )

    # Add reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)

    # Show best value for this parameter
    best_idx = sensitivity_data["sharpe_diff_mean"].idxmax()
    best_value = sensitivity_data.loc[best_idx, param_name]
    best_diff = sensitivity_data.loc[best_idx, "sharpe_diff_mean"]

    st.info(f"**Best {param_name}**: {best_value} (Sharpe diff: {best_diff:+.3f})")


def render_top_bottom_params(
    sweep_result: "SweepResult",
    n_show: int = 5,
) -> None:
    """
    Render tables of top and bottom parameter combinations.

    Args:
        sweep_result: Result from parallel_parameter_sweep
        n_show: Number of top/bottom combinations to show
    """
    st.markdown("#### Top & Bottom Parameter Combinations")

    results_df = sweep_result.results_df

    if "sharpe_diff_agg" not in results_df.columns:
        st.warning("Sharpe diff not available.")
        return

    # Get parameter columns
    param_cols = [c for c in results_df.columns if c not in [
        "combo_idx", "sharpe_diff_agg", "train_sharpe_agg", "test_sharpe_agg"
    ] and not c.startswith("train_sharpe_fold_") and not c.startswith("test_sharpe_fold_")]

    display_cols = param_cols + ["train_sharpe_agg", "test_sharpe_agg", "sharpe_diff_agg"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏆 Best Generalizing (Blue Zone)**")
        top_df = results_df.nlargest(n_show, "sharpe_diff_agg")[display_cols]
        st.dataframe(
            top_df.style.format({
                "train_sharpe_agg": "{:.3f}",
                "test_sharpe_agg": "{:.3f}",
                "sharpe_diff_agg": "{:+.3f}",
            }).background_gradient(
                subset=["sharpe_diff_agg"],
                cmap="Blues",
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("**⚠️ Most Overfit (Red Zone)**")
        bottom_df = results_df.nsmallest(n_show, "sharpe_diff_agg")[display_cols]
        st.dataframe(
            bottom_df.style.format({
                "train_sharpe_agg": "{:.3f}",
                "test_sharpe_agg": "{:.3f}",
                "sharpe_diff_agg": "{:+.3f}",
            }).background_gradient(
                subset=["sharpe_diff_agg"],
                cmap="Reds_r",
            ),
            use_container_width=True,
            hide_index=True,
        )
