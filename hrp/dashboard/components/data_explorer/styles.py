"""
Data Explorer Visual Styles - Dark Terminal Aesthetic

High-contrast, data-dense design inspired by Bloomberg Terminal
and cyberpunk command centers. Monospace fonts, neon accents.
"""

from typing import Any

import plotly.graph_objects as go

# Color Palette - Dark Terminal
COLORS = {
    "background": "#0a0a0f",  # Deep void
    "card": "#111116",  # Slightly lighter
    "border": "#1a1a24",  # Subtle borders
    "text": "#c4c4c8",  # Muted gray text
    "text_dim": "#6b6b76",  # Dimmed text
    "accent": "#00d4ff",  # Cyan neon
    "accent_dim": "rgba(0, 212, 255, 0.15)",  # Cyan glow
    "warning": "#ff9500",  # Amber
    "error": "#ff4757",  # Red
    "success": "#00e676",  # Green
    "chart_up": "#00e676",  # Green for up/bullish
    "chart_down": "#ff4757",  # Red for down/bearish
    "grid": "#1a1a24",  # Chart grid lines
    "selection": "rgba(0, 212, 255, 0.25)",  # Selection highlight
}

# Monospace font stack - terminal aesthetic
FONT_FAMILY = "'JetBrains Mono', 'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace"
FONT_FAMILY_ALT = "'IBM Plex Mono', 'Source Code Pro', monospace"

# Chart defaults
CHART_DEFAULTS = {
    "template": "plotly_dark",
    "paper_bgcolor": COLORS["background"],
    "plot_bgcolor": COLORS["card"],
    "font": {
        "family": FONT_FAMILY,
        "size": 11,
        "color": COLORS["text"],
    },
    "margin": dict(l=10, r=10, t=30, b=30),
    "hovermode": "x unified",
    "dragmode": "zoom",
    "xaxis": {
        "gridcolor": COLORS["grid"],
        "showgrid": True,
        "gridwidth": 1,
        "zerolinecolor": COLORS["grid"],
    },
    "yaxis": {
        "gridcolor": COLORS["grid"],
        "showgrid": True,
        "gridwidth": 1,
        "zerolinecolor": COLORS["grid"],
    },
}

# Chart-specific customizations
CANDLESTICK_COLORS = {
    "up": COLORS["chart_up"],
    "down": COLORS["chart_down"],
}

# Indicator line colors
INDICATOR_COLORS = {
    "sma_20": "#00d4ff",
    "sma_50": "#ffd700",
    "sma_200": "#ff9500",
    "ema_12": "#00d4ff",
    "ema_26": "#ffd700",
    "bollinger": "rgba(255, 149, 0, 0.3)",
    "rsi": "#ff9500",
    "macd": "#00d4ff",
    "volume": "rgba(0, 212, 255, 0.5)",
}

# Feature distribution colors
FEATURE_PALETTE = [
    "#00d4ff",  # Cyan
    "#ffd700",  # Gold
    "#ff9500",  # Amber
    "#00e676",  # Green
    "#ff4757",  # Red
    "#b967ff",  # Purple
    "#ff6ec7",  # Pink
    "#00f2ff",  # Light cyan
]


def apply_chart_theme(fig: go.Figure, chart_type: str = "default") -> go.Figure:
    """
    Apply dark terminal theme to a Plotly figure.

    Args:
        fig: Plotly figure to style
        chart_type: Type of chart for specific styling

    Returns:
        Styled figure
    """
    # Update layout with defaults
    fig.update_layout(**CHART_DEFAULTS)

    # Chart-specific adjustments
    if chart_type == "candlestick":
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                gridcolor=COLORS["grid"],
                showgrid=True,
            ),
        )
    elif chart_type == "heatmap":
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )

    # Update traces for consistent styling
    for trace in fig.data:
        if hasattr(trace, "line"):
            trace.update(line=dict(width=1.5))

    return fig


def get_metric_card_style(value: float, threshold: tuple[float, float] = (50, 80)) -> str:
    """
    Get color for metric value based on thresholds.

    Args:
        value: Metric value (0-100)
        threshold: (warning, good) thresholds

    Returns:
        CSS color string
    """
    if value >= threshold[1]:
        return COLORS["success"]
    elif value >= threshold[0]:
        return COLORS["warning"]
    return COLORS["error"]


def get_gradient_css(direction: str = "135deg") -> str:
    """
    Get subtle gradient CSS for backgrounds.

    Args:
        direction: CSS gradient direction

    Returns:
        CSS gradient string
    """
    return f"linear-gradient({direction}, {COLORS['background']}, {COLORS['card']})"


def get_glow_css(color: str = None) -> str:
    """
    Get neon glow effect CSS.

    Args:
        color: Color to glow (default: accent)

    Returns:
        CSS box-shadow string
    """
    glow_color = color or COLORS["accent"]
    return f"0 0 20px {glow_color}40, 0 0 40px {glow_color}20"


def get_status_badge_style(status: str) -> dict[str, str]:
    """
    Get styling for status badges.

    Args:
        status: Status string (healthy, warning, critical)

    Returns:
        Dict with CSS properties
    """
    styles = {
        "healthy": {
            "bg": f"{COLORS['success']}20",
            "text": COLORS["success"],
            "border": COLORS["success"],
        },
        "warning": {
            "bg": f"{COLORS['warning']}20",
            "text": COLORS["warning"],
            "border": COLORS["warning"],
        },
        "critical": {
            "bg": f"{COLORS['error']}20",
            "text": COLORS["error"],
            "border": COLORS["error"],
        },
    }
    return styles.get(status.lower(), styles["healthy"])
