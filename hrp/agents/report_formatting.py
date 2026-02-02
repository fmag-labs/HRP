"""
Shared report formatting utilities for institutional-grade markdown reports.

Provides reusable visual components: headers, dashboards, progress bars,
alert banners, scorecards, pipeline flows, and more.

All output is terminal-friendly pure markdown with Unicode box-drawing
characters and emoji for visual hierarchy.
"""

from datetime import datetime
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BRAND = "HRP | Hedgefund Research Platform"
DIVIDER_HEAVY = "━" * 60
DIVIDER_DOUBLE = "═" * 60
DIVIDER_LIGHT = "─" * 60

# Report type icons
REPORT_ICONS = {
    "daily": "📋",
    "weekly": "📊",
    "cio-review": "🎯",
    "ml-quality-sentinel": "🔬",
    "validation-analyst": "✅",
    "risk-manager": "🛡️",
    "kill-gates": "⚔️",
    "agent-execution": "🤖",
    "validation": "📝",
}

# Status emoji mappings
STATUS_EMOJI = {
    "passed": "🟢",
    "pass": "🟢",
    "success": "🟢",
    "validated": "🟢",
    "continue": "🟢",
    "approved": "🟢",
    "failed": "🔴",
    "fail": "🔴",
    "error": "🔴",
    "rejected": "🔴",
    "killed": "🔴",
    "kill": "🔴",
    "vetoed": "🔴",
    "warning": "🟡",
    "conditional": "🟡",
    "flagged": "🟡",
    "pending": "⚪",
    "running": "🔵",
    "pivot": "🔄",
}

PRIORITY_EMOJI = {
    "high": "🔴",
    "critical": "🔴",
    "medium": "🟡",
    "low": "🟢",
    "info": "🔵",
}

DECISION_EMOJI = {
    "CONTINUE": "✅",
    "CONDITIONAL": "⚠️",
    "KILL": "❌",
    "PIVOT": "🔄",
}


# ══════════════════════════════════════════════════════════════════════════════
# HEADER / FOOTER
# ══════════════════════════════════════════════════════════════════════════════


def render_header(
    title: str,
    report_type: str,
    date_str: str | None = None,
    subtitle: str = "",
) -> str:
    """
    Render branded report header.

    Args:
        title: Report title (e.g. "Daily Research Report")
        report_type: Key for icon lookup (e.g. "daily", "cio-review")
        date_str: Date string (defaults to today)
        subtitle: Optional subtitle line
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    icon = REPORT_ICONS.get(report_type, "📄")

    lines = [
        DIVIDER_HEAVY,
        f"📊 {BRAND}",
        DIVIDER_HEAVY,
        "",
        f"# {icon} {title} — {date_str}",
        "",
    ]

    if subtitle:
        lines.append(f"> {subtitle}")
        lines.append("")

    return "\n".join(lines)


def render_footer(
    agent_name: str,
    timestamp: datetime | None = None,
    cost_usd: float | None = None,
    duration_seconds: float | None = None,
    extra_lines: list[str] | None = None,
) -> str:
    """Render standardized report footer."""
    if timestamp is None:
        timestamp = datetime.now()

    lines = [
        "",
        DIVIDER_HEAVY,
        "",
        f"📊 **{BRAND}**",
        "",
    ]

    meta_parts = [f"🕐 {timestamp.strftime('%Y-%m-%d %H:%M')} ET"]
    if cost_usd is not None:
        meta_parts.append(f"💰 ${cost_usd:.4f}")
    if duration_seconds is not None:
        meta_parts.append(f"⏱️ {_format_duration(duration_seconds)}")
    meta_parts.append(f"🤖 {agent_name}")

    lines.append(" | ".join(meta_parts))

    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)

    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# KPI DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════


def render_kpi_dashboard(metrics: list[dict[str, Any]]) -> str:
    """
    Render a visual KPI dashboard with box-drawing characters.

    Args:
        metrics: List of dicts with keys: icon, label, value, detail (optional)
                 Example: {"icon": "📝", "label": "Hypotheses", "value": "14", "detail": "+2 today"}
    """
    if not metrics:
        return ""

    # Calculate column widths
    col_width = 18
    n = len(metrics)

    top = "┌" + "┬".join(["─" * col_width] * n) + "┐"
    mid = "├" + "┼".join(["─" * col_width] * n) + "┤"
    bot = "└" + "┴".join(["─" * col_width] * n) + "┘"

    # Row 1: icon + label
    row1_cells = []
    for m in metrics:
        cell = f" {m.get('icon', '')} {m['label']}"
        row1_cells.append(cell.ljust(col_width))

    # Row 2: value (centered, bold look)
    row2_cells = []
    for m in metrics:
        val = str(m["value"])
        padding = (col_width - len(val)) // 2
        cell = " " * max(padding, 1) + val
        row2_cells.append(cell.ljust(col_width))

    # Row 3: detail (smaller context)
    row3_cells = []
    for m in metrics:
        detail = m.get("detail", "")
        cell = f" {detail}" if detail else ""
        row3_cells.append(cell.ljust(col_width))

    lines = [
        "## 📊 Key Metrics",
        "",
        top,
        "│" + "│".join(row1_cells) + "│",
        "│" + "│".join(row2_cells) + "│",
        "│" + "│".join(row3_cells) + "│",
        bot,
        "",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS BARS & GAUGES
# ══════════════════════════════════════════════════════════════════════════════


def render_progress_bar(
    value: float,
    max_val: float = 1.0,
    width: int = 20,
    show_pct: bool = True,
) -> str:
    """
    Render a text-based progress bar.

    Args:
        value: Current value
        max_val: Maximum value (for percentage calculation)
        width: Bar width in characters
        show_pct: Whether to show percentage after bar
    """
    if max_val <= 0:
        pct = 0.0
    else:
        pct = min(value / max_val, 1.0)

    filled = int(pct * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty

    if show_pct:
        return f"[{bar}] {pct:.0%}"
    return f"[{bar}]"


def render_health_gauges(items: list[dict[str, Any]]) -> str:
    """
    Render health/progress gauges.

    Args:
        items: List of dicts with keys: label, value, max_val (default 100),
               trend (optional: "up", "down", "stable")
    """
    if not items:
        return ""

    lines = ["## 🏥 System Health", ""]

    trend_icons = {"up": "▲", "down": "▼", "stable": "●", "": ""}

    for item in items:
        label = item["label"].ljust(22)
        value = item["value"]
        max_val = item.get("max_val", 100)
        trend = trend_icons.get(item.get("trend", ""), "")

        bar = render_progress_bar(value, max_val, width=20)

        # Color indicator based on percentage
        pct = value / max_val if max_val > 0 else 0
        if pct >= 0.8:
            indicator = "🟢"
        elif pct >= 0.5:
            indicator = "🟡"
        else:
            indicator = "🔴"

        lines.append(f"{label} {bar} {trend} {indicator}")

    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ALERT BANNERS
# ══════════════════════════════════════════════════════════════════════════════


def render_alert_banner(
    messages: list[str],
    severity: str = "warning",
) -> str:
    """
    Render a boxed alert banner.

    Args:
        messages: Lines of text to display
        severity: "critical", "warning", or "info"
    """
    if not messages:
        return ""

    icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(severity, "⚠️")

    # Calculate box width
    max_len = max(len(m) for m in messages)
    box_width = max(max_len + 8, 40)

    top = "╔" + "═" * box_width + "╗"
    bot = "╚" + "═" * box_width + "╝"

    lines = [top]
    for msg in messages:
        padded = f"  {icon}  {msg}".ljust(box_width)
        lines.append(f"║{padded}║")
    lines.append(bot)
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FLOW
# ══════════════════════════════════════════════════════════════════════════════


def render_pipeline_flow(stages: list[dict[str, Any]]) -> str:
    """
    Render a visual pipeline flow.

    Args:
        stages: List of dicts with keys: icon, label, count, max_count (for bar)
    """
    if not stages:
        return ""

    lines = ["## 🔄 Pipeline Flow", "", "```"]

    # Row 1: icons and labels
    stage_strs = []
    for s in stages:
        cell = f"{s.get('icon', '📦')} {s['label']}"
        stage_strs.append(cell.center(14))
    lines.append("  →  ".join(stage_strs))

    # Row 2: counts
    count_strs = []
    for s in stages:
        count_strs.append(f"[{s['count']}]".center(14))
    lines.append("  →  ".join(count_strs))

    # Row 3: mini progress bars
    bar_strs = []
    for s in stages:
        count = s["count"]
        max_count = s.get("max_count", max(st["count"] for st in stages) or 1)
        bar = render_progress_bar(count, max_count, width=10, show_pct=False)
        bar_strs.append(bar.center(14))
    lines.append("     ".join(bar_strs))

    lines.extend(["```", ""])
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SCORECARDS
# ══════════════════════════════════════════════════════════════════════════════


def render_scorecard(
    title: str,
    dimensions: list[dict[str, Any]],
    overall_score: float | None = None,
    decision: str | None = None,
) -> str:
    """
    Render a visual scorecard with progress bars.

    Args:
        title: Scorecard title (e.g. hypothesis ID and name)
        dimensions: List of dicts with keys: label, score (0-1)
        overall_score: Overall score (0-1)
        decision: Decision string (e.g. "CONTINUE", "KILL")
    """
    col1_w = 16
    col2_w = 8
    col3_w = 28

    total_w = col1_w + col2_w + col3_w + 4  # +4 for separators

    top = "┌" + "─" * col1_w + "┬" + "─" * col2_w + "┬" + "─" * col3_w + "┐"
    hdr = "├" + "─" * col1_w + "┼" + "─" * col2_w + "┼" + "─" * col3_w + "┤"
    mid = "├" + "─" * col1_w + "┼" + "─" * col2_w + "┼" + "─" * col3_w + "┤"
    bot = "└" + "─" * col1_w + "┴" + "─" * col2_w + "┴" + "─" * col3_w + "┘"

    lines = [f"### 📊 {title}", "", top]

    # Header row
    lines.append(
        f"│{'  Dimension'.ljust(col1_w)}│{'  Score'.ljust(col2_w)}│{'  Rating'.ljust(col3_w)}│"
    )
    lines.append(hdr)

    # Dimension rows
    for dim in dimensions:
        label = f"  {dim['label']}".ljust(col1_w)
        score = f"  {dim['score']:.2f}".ljust(col2_w)
        bar = render_progress_bar(dim["score"], 1.0, width=18, show_pct=True)
        rating = f"  {bar}".ljust(col3_w)
        lines.append(f"│{label}│{score}│{rating}│")

    # Overall row
    if overall_score is not None:
        lines.append(mid)
        emoji = "✅" if overall_score >= 0.7 else "⚠️" if overall_score >= 0.5 else "❌"
        label = f"  **OVERALL**".ljust(col1_w)
        score = f"  **{overall_score:.2f}**".ljust(col2_w)
        bar = render_progress_bar(overall_score, 1.0, width=18, show_pct=False)
        rating = f"  {bar} {emoji}".ljust(col3_w)
        lines.append(f"│{label}│{score}│{rating}│")

    lines.append(bot)

    if decision:
        emoji = DECISION_EMOJI.get(decision, "❓")
        lines.append(f"\n**Decision:** {emoji} **{decision}**")

    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════


def render_status_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    status_col: int | None = None,
) -> str:
    """
    Render a markdown table with optional status emoji in a column.

    Args:
        title: Section title
        headers: Column headers
        rows: List of row data (list of strings)
        status_col: If set, applies status emoji lookup to that column index
    """
    if not rows:
        return ""

    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")

    for row in rows:
        cells = list(row)
        if status_col is not None and 0 <= status_col < len(cells):
            status_key = cells[status_col].lower().strip()
            emoji = STATUS_EMOJI.get(status_key, "")
            if emoji:
                cells[status_col] = f"{emoji} {cells[status_col]}"
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════


def render_agent_activity(agents: dict[str, dict[str, Any]]) -> str:
    """
    Render agent activity status panel.

    Args:
        agents: Dict of agent_name -> {"status": str, "last_run": str|None, "details": dict}
    """
    lines = ["## 🤖 Agent Activity", ""]

    lines.append("```")

    for name, info in agents.items():
        status = info.get("status", "pending")
        emoji = STATUS_EMOJI.get(status, "⚪")

        display_name = name.replace("_", " ").title()
        last_run = info.get("last_run", "")
        time_str = f"  ({last_run})" if last_run else ""

        # Extract key details
        details = info.get("details", {})
        detail_parts = []
        for k, v in details.items():
            if isinstance(v, (int, float)) and v > 0:
                detail_parts.append(f"{k}: {v}")

        detail_str = f"  │ {', '.join(detail_parts)}" if detail_parts else ""

        lines.append(f"  {emoji} {display_name.ljust(25)} {status.upper().ljust(10)}{time_str}{detail_str}")

    lines.extend(["```", ""])
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# RISK / VETO DISPLAY
# ══════════════════════════════════════════════════════════════════════════════


def render_risk_limits(limits: dict[str, str]) -> str:
    """Render risk limits as a formatted panel."""
    lines = ["## 🛡️ Risk Limits", "", "```"]

    for label, value in limits.items():
        lines.append(f"  {label.ljust(25)} {value}")

    lines.extend(["```", ""])
    return "\n".join(lines)


def render_veto_section(
    hypothesis_id: str,
    status: str,
    vetos: list[dict[str, str]] | None = None,
    warnings: list[str] | None = None,
    portfolio_impact: dict[str, Any] | None = None,
) -> str:
    """Render a hypothesis risk assessment section."""
    emoji = STATUS_EMOJI.get(status.lower(), "⚪")
    lines = [f"### {emoji} {hypothesis_id}: **{status.upper()}**", ""]

    if vetos:
        lines.append("**Vetos:**")
        for veto in vetos:
            severity = veto.get("severity", "critical")
            sev_emoji = "🚫" if severity == "critical" else "⚠️"
            lines.append(f"  {sev_emoji} **{veto.get('type', 'Unknown')}** — {veto.get('reason', '')}")
        lines.append("")

    if warnings:
        lines.append("**Warnings:**")
        for w in warnings:
            lines.append(f"  ⚠️ {w}")
        lines.append("")

    if portfolio_impact:
        lines.append("**Portfolio Impact:**")
        lines.append("```")
        for k, v in portfolio_impact.items():
            display_key = k.replace("_", " ").title()
            if isinstance(v, float) and v < 1:
                lines.append(f"  {display_key.ljust(25)} {v:.1%}")
            else:
                lines.append(f"  {display_key.ljust(25)} {v}")
        lines.append("```")
        lines.append("")

    lines.append(DIVIDER_LIGHT)
    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHTS / ACTION ITEMS
# ══════════════════════════════════════════════════════════════════════════════


def render_insights(
    title: str,
    insights: list[dict[str, Any]],
) -> str:
    """
    Render prioritized insights/action items.

    Args:
        title: Section title
        insights: List of dicts with keys: priority, category, action (or insight)
    """
    if not insights:
        return ""

    lines = [f"## 💡 {title}", ""]

    for i, insight in enumerate(insights, 1):
        priority = insight.get("priority", "low")
        emoji = PRIORITY_EMOJI.get(priority, "🔵")
        category = insight.get("category", "general").upper()
        action = insight.get("action", insight.get("insight", ""))

        lines.append(f"{i}. {emoji} **[{category}]** {action}")

    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def get_status_emoji(status: str) -> str:
    """Get emoji for a status string."""
    return STATUS_EMOJI.get(status.lower().strip(), "⚪")


def format_metric(value: Any, fmt: str = "") -> str:
    """Format a metric value safely."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        if fmt == "pct":
            return f"{float(value):.1%}"
        elif fmt == "f2":
            return f"{float(value):.2f}"
        elif fmt == "f3":
            return f"{float(value):.3f}"
        elif fmt == "f4":
            return f"{float(value):.4f}"
        elif fmt == "usd":
            return f"${float(value):.4f}"
        elif fmt == "int":
            return str(int(value))
        return str(value)
    except (ValueError, TypeError):
        return str(value)


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def render_section_divider(title: str = "") -> str:
    """Render a visual section divider."""
    if title:
        return f"\n{DIVIDER_LIGHT}\n\n## {title}\n"
    return f"\n{DIVIDER_LIGHT}\n"
