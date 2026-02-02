"""Agent reporting utilities for standardized execution reports."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AgentReport:
    """Standardized agent execution report.

    Provides consistent reporting format for all agent types with
    markdown generation and file persistence.
    """

    agent_name: str
    start_time: datetime
    end_time: datetime
    status: str  # "success", "failed", "running"
    results: dict[str, Any]
    errors: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report with institutional formatting."""
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, get_status_emoji,
        )

        duration = self.end_time - self.start_time
        duration_seconds = duration.total_seconds()

        parts = []

        # ── Header ──
        parts.append(render_header(
            title=f"{self.agent_name} Execution Report",
            report_type="agent-execution",
            date_str=self.start_time.strftime("%Y-%m-%d"),
        ))

        # ── KPI Dashboard ──
        status_emoji = get_status_emoji(self.status)
        error_count = len(self.errors)
        result_count = len(self.results)

        parts.append(render_kpi_dashboard([
            {"icon": status_emoji, "label": "Status", "value": self.status.upper(), "detail": ""},
            {"icon": "⏱️", "label": "Duration", "value": _format_duration(duration_seconds), "detail": ""},
            {"icon": "📊", "label": "Results", "value": result_count, "detail": "fields"},
            {"icon": "❌" if error_count > 0 else "✅", "label": "Errors", "value": error_count, "detail": ""},
        ]))

        # ── Alert banner for errors ──
        if self.errors:
            parts.append(render_alert_banner(
                [f"{error_count} error(s) occurred during execution"],
                severity="critical",
            ))

        # ── Execution Details ──
        parts.append("## ⏱️ Execution Details\n")
        parts.append("```")
        parts.append(f"  Agent:     {self.agent_name}")
        parts.append(f"  Status:    {status_emoji} {self.status.upper()}")
        parts.append(f"  Start:     {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        parts.append(f"  End:       {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        parts.append(f"  Duration:  {_format_duration(duration_seconds)}")
        parts.append("```\n")

        # ── Results ──
        if self.results:
            parts.append("## 📊 Results\n")
            parts.append(_format_dict(self.results))
            parts.append("")

        # ── Errors ──
        if self.errors:
            parts.append("## 🚨 Errors\n")
            for i, error in enumerate(self.errors, 1):
                parts.append(f"{i}. 🔴 {error}")
            parts.append("")

        # ── Footer ──
        parts.append(render_footer(
            agent_name=self.agent_name,
            timestamp=self.end_time,
            duration_seconds=duration_seconds,
        ))

        return "\n".join(parts)

    def save_to_file(self, path: Path) -> None:
        """Save report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown())


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _format_dict(d: dict[str, Any], indent: int = 0) -> str:
    """Format dictionary as markdown."""
    lines = []
    prefix = "  " * indent

    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}- **{key}**:")
            lines.append(_format_dict(value, indent + 1))
        else:
            lines.append(f"{prefix}- **{key}**: {value}")

    return "\n".join(lines)
