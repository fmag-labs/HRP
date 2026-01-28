"""
Features Data Explorer - Distribution & Correlation Visualization

Histograms, correlation heatmaps, and outlier detection for 44 technical features.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from hrp.dashboard.components.data_explorer.query_engine import QueryEngine
from hrp.dashboard.components.data_explorer.styles import (
    CHART_DEFAULTS,
    COLORS,
    FEATURE_PALETTE,
    apply_chart_theme,
    FONT_FAMILY,
)


def render_features_explorer() -> None:
    """Render the features data explorer interface."""
    # Get available features
    all_features = QueryEngine.get_available_features()

    if not all_features:
        st.info("No feature data available. Run feature computation to populate.")
        return

    # -------------------------------------------------------------------------
    # Filters Section
    # -------------------------------------------------------------------------
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 2])

        with col1:
            viz_type = st.selectbox(
                "Visualization",
                options=["Distribution Histograms", "Correlation Heatmap", "Feature Scatter", "Summary Stats"],
                key="feature_viz_type",
            )

        with col2:
            if viz_type == "Correlation Heatmap":
                selected_features = st.multiselect(
                    "Features",
                    options=all_features,
                    default=all_features[:10],
                    key="feature_corr_features",
                    help="Select features to correlate",
                )
            elif viz_type == "Feature Scatter":
                x_feature = st.selectbox("X Axis", options=all_features, key="feature_scatter_x")
                y_feature = st.selectbox(
                    "Y Axis", options=all_features, index=1 if len(all_features) > 1 else 0, key="feature_scatter_y"
                )

        with col3:
            recent_days = st.selectbox(
                "Period",
                options=["30", "90", "252", "ALL"],
                index=2,
                key="feature_period",
            )
            days = int(recent_days) if recent_days != "ALL" else None

    # -------------------------------------------------------------------------
    # Distribution Histograms
    # -------------------------------------------------------------------------
    if viz_type == "Distribution Histograms":
        st.subheader("Feature Distribution Histograms")

        # Feature selector
        col1, col2 = st.columns([3, 1])
        with col1:
            hist_feature = st.selectbox(
                "Select Feature",
                options=all_features,
                key="feature_hist",
            )
        with col2:
            n_bins = st.slider("Bins", min_value=10, max_value=100, value=50, key="feature_bins")

        # Get feature data
        with st.spinner("Loading feature distribution..."):
            # Calculate as_of_date from recent_days
            from datetime import date, timedelta
            as_of = date.today() - timedelta(days=days or 252)

            df = QueryEngine.get_feature_values(
                _features=(hist_feature,),
                _as_of_date=as_of,
                _limit=100000,
            )

        if df.empty:
            st.warning(f"No data for {hist_feature}")
            return

        # Get distribution stats
        stats = QueryEngine.get_feature_distribution(_feature_name=hist_feature)

        # Create histogram
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=df[hist_feature].dropna(),
                nbinsx=n_bins,
                name=hist_feature,
                marker_color=COLORS["accent"],
                opacity=0.8,
            )
        )

        # Add stats annotations
        if not stats.empty:
            mean_val = stats["mean"].iloc[0]
            median_val = stats["median"].iloc[0]
            std_val = stats["std"].iloc[0]

            fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS["warning"], annotation_text=f"Mean: {mean_val:.3f}")
            fig.add_vline(
                x=median_val, line_dash="dot", line_color=COLORS["success"], annotation_text=f"Median: {median_val:.3f}"
            )

        fig.update_layout(
            title={
                "text": f"{hist_feature} Distribution",
                "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            },
            xaxis_title=hist_feature,
            yaxis_title="Count",
            height=450,
            **CHART_DEFAULTS,
        )

        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        if not stats.empty:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Count", f"{int(stats['count'].iloc[0]):,}")
            with col2:
                st.metric("Mean", f"{stats['mean'].iloc[0]:.4f}")
            with col3:
                st.metric("Std Dev", f"{stats['std'].iloc[0]:.4f}")
            with col4:
                st.metric("Range", f"{stats['min'].iloc[0]:.4f} - {stats['max'].iloc[0]:.4f}")

    # -------------------------------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------------------------------
    elif viz_type == "Correlation Heatmap":
        st.subheader("Feature Correlation Matrix")

        if not selected_features or len(selected_features) < 2:
            st.warning("Select at least 2 features for correlation")
            return

        with st.spinner("Computing correlation matrix..."):
            corr_matrix = QueryEngine.get_feature_correlation(
                _features=tuple(selected_features),
                _recent_days=days or 252,
            )

        if corr_matrix.empty:
            st.warning("Could not compute correlation matrix")
            return

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10, "color": COLORS["text"]},
                colorbar=dict(
                    title="Correlation",
                    titlefont=dict(family=FONT_FAMILY, size=11),
                    tickfont=dict(family=FONT_FAMILY, size=10),
                ),
            )
        )

        fig.update_layout(
            title={
                "text": f"Feature Correlation ({selected_features[0]} vs others)",
                "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            },
            xaxis=dict(side="bottom", tickangle=-45),
            yaxis=dict(side="left"),
            height=max(400, len(selected_features) * 30),
            **CHART_DEFAULTS,
        )

        fig = apply_chart_theme(fig, "heatmap")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Feature Scatter Plot
    # -------------------------------------------------------------------------
    elif viz_type == "Feature Scatter":
        st.subheader(f"{x_feature} vs {y_feature}")

        with st.spinner("Loading feature data..."):
            # Calculate as_of_date from recent_days
            from datetime import date, timedelta
            as_of = date.today() - timedelta(days=days or 252)

            df = QueryEngine.get_feature_values(
                _features=(x_feature, y_feature),
                _as_of_date=as_of,
                _limit=100000,
            )

        if df.empty or x_feature not in df.columns or y_feature not in df.columns:
            st.warning(f"No data for {x_feature} and {y_feature}")
            return

        # Remove nulls
        scatter_df = df[[x_feature, y_feature]].dropna()

        if scatter_df.empty:
            st.warning("No valid data points after removing nulls")
            return

        # Calculate correlation
        corr = scatter_df[x_feature].corr(scatter_df[y_feature])

        # Create scatter
        fig = px.scatter(
            scatter_df,
            x=x_feature,
            y=y_feature,
            trendline="ols",
            opacity=0.6,
        )

        fig.update_traces(
            marker=dict(size=4, color=COLORS["accent"]),
            selector=dict(mode="markers"),
        )

        # Update trendline style
        for trace in fig.data:
            if hasattr(trace, "line") and trace.line.color:
                trace.update(line=dict(color=COLORS["warning"], width=2))

        fig.update_layout(
            title={
                "text": f"{x_feature} vs {y_feature} | Correlation: {corr:.3f}",
                "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            },
            height=500,
            **CHART_DEFAULTS,
        )

        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    elif viz_type == "Summary Stats":
        st.subheader("All Features Summary")

        # Compute stats for all features
        all_stats = []

        with st.spinner("Computing statistics for all features..."):
            for feature in all_features:
                stats = QueryEngine.get_feature_distribution(_feature_name=feature)
                if not stats.empty:
                    all_stats.append(
                        {
                            "Feature": feature,
                            "Count": int(stats["count"].iloc[0]),
                            "Mean": f"{stats['mean'].iloc[0]:.4f}",
                            "Std Dev": f"{stats['std'].iloc[0]:.4f}",
                            "Min": f"{stats['min'].iloc[0]:.4f}",
                            "Max": f"{stats['max'].iloc[0]:.4f}",
                            "Median": f"{stats['median'].iloc[0]:.4f}",
                        }
                    )

        stats_df = pd.DataFrame(all_stats)

        if not stats_df.empty:
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Feature": st.column_config.TextColumn("Feature", width="medium"),
                    "Count": st.column_config.NumberColumn("Count", width="small"),
                    "Mean": st.column_config.TextColumn("Mean", width="small"),
                    "Std Dev": st.column_config.TextColumn("Std Dev", width="small"),
                    "Min": st.column_config.TextColumn("Min", width="small"),
                    "Max": st.column_config.TextColumn("Max", width="small"),
                    "Median": st.column_config.TextColumn("Median", width="small"),
                },
            )
        else:
            st.warning("No statistics available")
