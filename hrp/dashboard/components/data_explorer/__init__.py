"""
Data Explorer Component - Interactive Database Visualization

Main entry point for the data explorer module. Provides tabbed interface
for exploring prices, features, fundamentals, and quality data.
"""

import streamlit as st

from hrp.dashboard.components.data_explorer.fundamentals_explorer import render_fundamentals_explorer
from hrp.dashboard.components.data_explorer.features_explorer import render_features_explorer
from hrp.dashboard.components.data_explorer.prices_explorer import render_prices_explorer
from hrp.dashboard.components.data_explorer.quality_viz import render_quality_viz
from hrp.dashboard.components.data_explorer.styles import COLORS, FONT_FAMILY


def render_data_explorer() -> None:
    """
    Render the main data explorer interface.

    Provides tabbed navigation between different data views:
    - Prices: OHLCV candlestick charts with indicators
    - Features: Distribution analysis and correlation heatmaps
    - Fundamentals: Company metrics and peer comparison
    - Quality: Anomalies, gaps, and data freshness
    """
    # Header with custom styling
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.75rem; font-weight: 600; font-family: {FONT_FAMILY}; margin: 0; color: {COLORS['text']};">
                🔍 Data Explorer
            </h2>
            <p style="color: {COLORS['text_dim']}; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Interactive database visualization - Prices, Features, Fundamentals, Quality
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data freshness indicator
    from hrp.dashboard.components.data_explorer.query_engine import QueryEngine

    freshness = QueryEngine.get_data_freshness()

    col1, col2 = st.columns([4, 1])
    with col1:
        if freshness["latest_date"]:
            freshness_color = COLORS["success"] if freshness["is_fresh"] else COLORS["warning"]
            st.markdown(
                f"""
                <div style="padding: 0.5rem 1rem; background: {COLORS['card']}; border-left: 3px solid {freshness_color}; border-radius: 4px; margin-bottom: 1rem;">
                    <span style="color: {COLORS['text']};">
                        Latest data: <strong>{freshness['latest_date']}</strong>
                        ({freshness['days_stale']} days stale) •
                        {freshness['symbol_count']:,} symbols •
                        {freshness['total_records']:,} total records
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No data available. Run ingestion to populate the database.")

    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False, key="explorer_auto_refresh")

    # Main tab navigation
    tabs = st.tabs(
        [
            "📈 Prices",
            "📊 Features",
            "💼 Fundamentals",
            "✅ Quality",
        ]
    )

    with tabs[0]:
        render_prices_explorer()

    with tabs[1]:
        render_features_explorer()

    with tabs[2]:
        render_fundamentals_explorer()

    with tabs[3]:
        render_quality_viz()

    # Footer
    st.markdown(
        f"""
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid {COLORS['border']}; text-align: center;">
            <p style="color: {COLORS['text_dim']}; font-size: 0.8rem; margin: 0;">
                Data Explorer • Cached queries • Interactive visualizations
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Auto-refresh handling
    if auto_refresh:
        st.runtime.legacy_caching.clear_cache()
        st.rerun()
