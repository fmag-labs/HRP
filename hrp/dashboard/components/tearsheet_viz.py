"""
Tear sheet visualization components for portfolio analysis.

Inspired by PyFolio tear sheets, providing visual analysis of backtest results.
"""

import empyrical as ep
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def render_returns_distribution(returns: pd.Series) -> None:
    """
    Render returns distribution histogram with normal distribution overlay.

    Args:
        returns: Daily returns series
    """
    st.markdown("##### Returns Distribution")

    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    skew = returns.skew()
    kurtosis = returns.kurtosis()

    # Create histogram
    fig = go.Figure()

    # Histogram of actual returns
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Actual Returns",
        histnorm="probability density",
        marker_color="rgba(99, 110, 250, 0.7)",
    ))

    # Normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_pdf = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                 np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_pdf,
        mode="lines",
        name="Normal Distribution",
        line=dict(color="red", width=2),
    ))

    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        height=300,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{mean_return * 100:.3f}%")
    col2.metric("Std Dev", f"{std_return * 100:.3f}%")
    col3.metric("Skewness", f"{skew:.2f}")
    col4.metric("Kurtosis", f"{kurtosis:.2f}")


def render_rolling_metrics(returns: pd.Series, window: int = 63) -> None:
    """
    Render rolling Sharpe ratio and volatility chart.

    Args:
        returns: Daily returns series
        window: Rolling window in days (default 63 = ~3 months)
    """
    st.markdown("##### Rolling Metrics")

    if len(returns) < window:
        st.warning(f"Insufficient data for {window}-day rolling metrics")
        return

    # Calculate rolling metrics
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
    )
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)

    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Rolling Sharpe Ratio", "Rolling Volatility"),
    )

    # Rolling Sharpe
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode="lines",
            name="Rolling Sharpe",
            line=dict(color="blue"),
        ),
        row=1, col=1
    )

    # Add zero line for Sharpe
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # Rolling Volatility
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode="lines",
            name="Rolling Vol",
            line=dict(color="orange"),
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.3)",
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=400,
        showlegend=False,
    )
    fig.update_yaxes(title_text="Sharpe", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", tickformat=".1%", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_analysis(returns: pd.Series) -> None:
    """
    Render drawdown analysis with underwater plot.

    Args:
        returns: Daily returns series
    """
    st.markdown("##### Drawdown Analysis")

    # Calculate cumulative returns and drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max

    # Create underwater plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode="lines",
        name="Drawdown",
        line=dict(color="red"),
        fill="tozeroy",
        fillcolor="rgba(255, 0, 0, 0.3)",
    ))

    fig.update_layout(
        title="Underwater Plot (Drawdown from Peak)",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        height=250,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Drawdown statistics
    max_dd = drawdown.min()
    avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

    # Calculate drawdown duration
    in_drawdown = drawdown < 0
    drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
    drawdown_lengths = in_drawdown.groupby(drawdown_periods).sum()
    max_dd_length = drawdown_lengths.max() if len(drawdown_lengths) > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
    col2.metric("Avg Drawdown", f"{avg_dd * 100:.2f}%")
    col3.metric("Max DD Duration", f"{max_dd_length} days")


def render_monthly_returns_heatmap(returns: pd.Series) -> None:
    """
    Render monthly returns heatmap.

    Args:
        returns: Daily returns series
    """
    st.markdown("##### Monthly Returns Heatmap")

    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    # Resample to monthly returns
    monthly_returns = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Create pivot table (year x month)
    monthly_returns_df = monthly_returns.to_frame(name="return")
    monthly_returns_df["year"] = monthly_returns_df.index.year
    monthly_returns_df["month"] = monthly_returns_df.index.month

    pivot = monthly_returns_df.pivot(index="year", columns="month", values="return")

    # Rename columns to month names
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    # Add annual returns column
    annual_returns = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    annual_returns.index = annual_returns.index.year
    pivot["Year"] = annual_returns

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,  # Convert to percentage
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0, "rgb(165, 0, 38)"],      # Deep red for losses
            [0.25, "rgb(215, 48, 39)"],  # Red
            [0.45, "rgb(254, 224, 144)"], # Light yellow
            [0.5, "rgb(255, 255, 255)"],  # White at 0
            [0.55, "rgb(171, 221, 164)"], # Light green
            [0.75, "rgb(26, 152, 80)"],   # Green
            [1, "rgb(0, 104, 55)"],       # Deep green for gains
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.values * 100],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Monthly Returns (%)",
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(200, 40 * len(pivot)),
        yaxis=dict(autorange="reversed"),  # Latest year at top
    )

    st.plotly_chart(fig, use_container_width=True)


def render_tail_risk_metrics(returns: pd.Series) -> None:
    """
    Render tail risk metrics visualization (VaR, CVaR).

    Args:
        returns: Daily returns series
    """
    st.markdown("##### Tail Risk Analysis")

    # Calculate tail risk metrics using Empyrical
    try:
        var_95 = ep.value_at_risk(returns, cutoff=0.05)
        cvar_95 = ep.conditional_value_at_risk(returns, cutoff=0.05)
        tail_ratio = ep.tail_ratio(returns)
    except Exception:
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        tail_ratio = abs(returns.quantile(0.95)) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0

    # Create returns histogram with VaR/CVaR markers
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Returns",
        marker_color="rgba(99, 110, 250, 0.7)",
    ))

    # VaR line
    fig.add_vline(
        x=var_95,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"VaR 95%: {var_95 * 100:.2f}%",
        annotation_position="top left",
    )

    # CVaR line
    fig.add_vline(
        x=cvar_95,
        line_dash="dash",
        line_color="red",
        annotation_text=f"CVaR 95%: {cvar_95 * 100:.2f}%",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Tail Risk: VaR and CVaR (95% Confidence)",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        height=300,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric("Value at Risk (95%)", f"{var_95 * 100:.2f}%",
                help="Maximum expected daily loss 95% of the time")
    col2.metric("CVaR / Expected Shortfall", f"{cvar_95 * 100:.2f}%",
                help="Average loss when losses exceed VaR")
    col3.metric("Tail Ratio", f"{tail_ratio:.2f}",
                help="Ratio of right tail (gains) to left tail (losses)")


def render_tear_sheet(returns: pd.Series, benchmark_returns: pd.Series = None) -> None:
    """
    Render a comprehensive tear sheet for backtest analysis.

    Args:
        returns: Daily strategy returns
        benchmark_returns: Optional benchmark returns for comparison
    """
    if returns is None or returns.empty:
        st.warning("No returns data available for tear sheet")
        return

    # Clean data
    returns = returns.dropna()

    if len(returns) < 30:
        st.warning("Insufficient data for tear sheet analysis (need at least 30 days)")
        return

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Returns Analysis",
        "Rolling Metrics",
        "Drawdown",
        "Tail Risk"
    ])

    with tab1:
        render_returns_distribution(returns)
        render_monthly_returns_heatmap(returns)

    with tab2:
        col1, col2 = st.columns([3, 1])
        with col2:
            window = st.selectbox(
                "Rolling Window",
                options=[21, 63, 126, 252],
                format_func=lambda x: f"{x} days (~{x // 21} months)",
                index=1,
                key="rolling_window"
            )
        render_rolling_metrics(returns, window=window)

    with tab3:
        render_drawdown_analysis(returns)

    with tab4:
        render_tail_risk_metrics(returns)
