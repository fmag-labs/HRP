"""
Validation report generation.

Creates comprehensive validation reports in markdown format.
"""

from datetime import date
from typing import Any

from loguru import logger


def generate_validation_report(data: dict[str, Any]) -> str:
    """
    Generate comprehensive validation report in markdown.

    Args:
        data: Dictionary with validation data including:
            - hypothesis_id
            - metrics
            - significance_test
            - robustness
            - validation_passed
            - confidence_score

    Returns:
        Markdown-formatted validation report
    """
    from hrp.agents.report_formatting import (
        render_header, render_footer, render_kpi_dashboard,
        render_alert_banner, render_health_gauges, render_progress_bar,
        render_section_divider, render_insights,
    )

    hypothesis_id = data["hypothesis_id"]
    metrics = data["metrics"]
    sig_test = data.get("significance_test", {})
    robustness = data.get("robustness", {})
    passed = data.get("validation_passed", False)
    confidence = data.get("confidence_score", 0.0)

    status = "VALIDATED" if passed else "REJECTED"
    status_emoji = "✅" if passed else "❌"

    parts = []

    # ── Header ──
    parts.append(render_header(
        title=f"Validation Report: {hypothesis_id}",
        report_type="validation",
        date_str=date.today().isoformat(),
        subtitle=f"{status_emoji} Status: **{status}** | Confidence: **{confidence:.2f}**",
    ))

    # ── KPI Dashboard ──
    sharpe = metrics.get("sharpe", 0)
    max_dd = metrics.get("max_drawdown", 0)
    win_rate = metrics.get("win_rate", 0)
    num_checks_passed = sum(1 for r in robustness.values() if r == "PASS")
    total_checks = len(robustness) if robustness else 0

    parts.append(render_kpi_dashboard([
        {"icon": status_emoji, "label": "Status", "value": status, "detail": f"conf: {confidence:.2f}"},
        {"icon": "📈", "label": "Sharpe", "value": f"{sharpe:.2f}", "detail": "OOS"},
        {"icon": "📉", "label": "Max DD", "value": f"{max_dd:.1%}", "detail": "limit: 25%"},
        {"icon": "🎯", "label": "Win Rate", "value": f"{win_rate:.1%}", "detail": "limit: 40%"},
    ]))

    # ── Alert banner ──
    if passed:
        parts.append(render_alert_banner(
            ["Strategy VALIDATED — approved for paper trading",
             "📌 Monitor for 30 days before live deployment consideration"],
            severity="info",
        ))
    else:
        parts.append(render_alert_banner(
            ["Strategy REJECTED — does not meet validation criteria",
             "📌 Review failed metrics and robustness checks below"],
            severity="critical",
        ))

    # ── Health Gauges ──
    parts.append(render_health_gauges([
        {"label": "Confidence Score", "value": confidence * 100, "max_val": 100,
         "trend": "up" if confidence > 0.7 else "down"},
        {"label": "Robustness Checks", "value": num_checks_passed, "max_val": max(total_checks, 1),
         "trend": "up" if num_checks_passed == total_checks else "down"},
    ]))

    # ── Performance Metrics ──
    parts.append(render_section_divider("📈 Performance Metrics (Out-of-Sample)"))

    def _check(val, threshold, op=">"):
        if op == ">":
            return "✅" if val > threshold else "❌"
        elif op == "<":
            return "✅" if val < threshold else "❌"
        elif op == ">=":
            return "✅" if val >= threshold else "❌"
        return "—"

    cagr = metrics.get("cagr", 0)
    profit_factor = metrics.get("profit_factor", 0)
    num_trades = metrics.get("num_trades", 0)
    oos_days = metrics.get("oos_period_days", 0)

    parts.append("| Metric | Value | Threshold | Result | Bar |")
    parts.append("|--------|-------|-----------|--------|-----|")
    parts.append(f"| **Sharpe Ratio** | {sharpe:.2f} | > 0.5 | {_check(sharpe, 0.5)} | {render_progress_bar(max(sharpe, 0), 2.0, width=10, show_pct=False)} |")
    parts.append(f"| **CAGR** | {cagr:.1%} | — | — | {render_progress_bar(max(cagr, 0), 0.5, width=10, show_pct=False)} |")
    parts.append(f"| **Max Drawdown** | {max_dd:.1%} | < 25% | {_check(max_dd, 0.25, '<')} | {render_progress_bar(max_dd, 0.5, width=10, show_pct=False)} |")
    parts.append(f"| **Win Rate** | {win_rate:.1%} | > 40% | {_check(win_rate, 0.40)} | {render_progress_bar(win_rate, 1.0, width=10, show_pct=False)} |")
    parts.append(f"| **Profit Factor** | {profit_factor:.2f} | > 1.2 | {_check(profit_factor, 1.2)} | {render_progress_bar(max(profit_factor, 0), 3.0, width=10, show_pct=False)} |")
    parts.append(f"| **Trade Count** | {num_trades} | ≥ 100 | {_check(num_trades, 100, '>=')} | {render_progress_bar(num_trades, 500, width=10, show_pct=False)} |")
    parts.append(f"| **OOS Period** | {oos_days} days | ≥ 730 | {_check(oos_days, 730, '>=')} | {render_progress_bar(oos_days, 1460, width=10, show_pct=False)} |")
    parts.append("")

    # ── Statistical Significance ──
    parts.append(render_section_divider("📊 Statistical Significance"))

    if sig_test:
        t_stat = sig_test.get("t_statistic", 0)
        p_val = sig_test.get("p_value", 1)
        significant = sig_test.get("significant", False)
        excess_ret = sig_test.get("excess_return_annualized", 0)

        parts.append("```")
        parts.append(f"  Excess Return (ann.)   {excess_ret:+.1%}")
        parts.append(f"  t-statistic            {t_stat:.2f}")
        parts.append(f"  p-value                {p_val:.4f}")
        parts.append(f"  Significant (α=0.05)   {'✅ YES' if significant else '❌ NO'}")
        parts.append("```")
        parts.append("")
    else:
        parts.append("> _No significance test performed_\n")

    # ── Robustness ──
    parts.append(render_section_divider("🛡️ Robustness Checks"))

    if robustness:
        parts.append("| Check | Result |")
        parts.append("|-------|--------|")
        for check_name, result in robustness.items():
            emoji = "✅" if result == "PASS" else "❌"
            parts.append(f"| {check_name.replace('_', ' ').title()} | {emoji} **{result}** |")
        parts.append("")
    else:
        parts.append("> _No robustness checks performed_\n")

    # ── Recommendation ──
    parts.append(render_section_divider("💡 Recommendation"))

    if passed:
        parts.append(render_insights("Next Steps", [
            {"priority": "high", "category": "deployment", "action": "Deploy to paper trading account"},
            {"priority": "medium", "category": "monitoring", "action": "Monitor live performance vs backtest"},
            {"priority": "medium", "category": "review", "action": "Review after 30 days minimum"},
            {"priority": "low", "category": "deployment", "action": "Consider live deployment if performance holds"},
        ]))
    else:
        parts.append(render_insights("Options", [
            {"priority": "high", "category": "research", "action": "Revise strategy and re-test"},
            {"priority": "medium", "category": "research", "action": "Investigate failed criteria"},
            {"priority": "low", "category": "research", "action": "Archive hypothesis as rejected"},
            {"priority": "low", "category": "research", "action": "Consider alternative approaches"},
        ]))

    # ── Footer ──
    parts.append(render_footer(agent_name="validation-system"))

    report = "\n".join(parts)
    logger.info(f"Generated validation report for {hypothesis_id}: {status}")

    return report


class ValidationReport:
    """Class for managing validation reports."""
    
    def __init__(self, hypothesis_id: str):
        self.hypothesis_id = hypothesis_id
    
    def generate(self, data: dict[str, Any]) -> str:
        """Generate report for this hypothesis."""
        data["hypothesis_id"] = self.hypothesis_id
        return generate_validation_report(data)
    
    def save(self, filepath: str, data: dict[str, Any]):
        """Generate and save report to file."""
        report = self.generate(data)
        
        with open(filepath, "w") as f:
            f.write(report)
        
        logger.info(f"Saved validation report to {filepath}")
