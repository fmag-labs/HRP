"""Consumer-mode Today page for HRP."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st


OPEN_STATUSES = ["active", "pending_approval"]
CLOSED_STATUSES = ["closed_profit", "closed_loss", "closed_stopped", "expired"]


def render(api: Any) -> None:
    """Render the consumer-facing daily advisory view."""
    st.title("Today")
    st.caption("Local research brief. Recommendations are decision support, not financial advice.")

    _render_status_strip(api)
    st.divider()
    _render_latest_recommendations(api)
    st.divider()
    _render_open_risk(api)
    st.divider()
    _render_track_record(api)
    st.divider()
    _render_daily_run_status(api)


def _render_status_strip(api: Any) -> None:
    latest_price_date = _fetch_scalar(api, "SELECT MAX(date) FROM prices")
    rec_count = _fetch_scalar(
        api,
        "SELECT COUNT(*) FROM recommendations WHERE status IN ('active', 'pending_approval')",
    ) or 0
    last_rec_at = _fetch_scalar(api, "SELECT MAX(created_at) FROM recommendations")

    freshness_label, freshness_help = _freshness(latest_price_date)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data", freshness_label, help=freshness_help)
    col2.metric("Open Ideas", int(rec_count))
    col3.metric("Last Brief", _format_datetime(last_rec_at))
    col4.metric("Trading", _trading_mode())


def _render_latest_recommendations(api: Any) -> None:
    st.subheader("Latest Recommendations")

    recommendations = _fetch_frame(
        api,
        """
        SELECT recommendation_id, created_at, symbol, action, confidence,
               signal_strength, entry_price, target_price, stop_price,
               position_pct, thesis_plain, risk_plain, status
        FROM recommendations
        WHERE status IN ('active', 'pending_approval')
        ORDER BY created_at DESC, signal_strength DESC
        LIMIT 5
        """,
    )

    if recommendations.empty:
        st.info("No active recommendations yet. Run the daily consumer refresh after setup.")
        return

    for _, rec in recommendations.iterrows():
        _render_recommendation(rec)


def _render_recommendation(rec: pd.Series) -> None:
    confidence = rec.get("confidence", "MEDIUM")
    confidence_color = {
        "HIGH": "#15803d",
        "MEDIUM": "#b45309",
        "LOW": "#b91c1c",
    }.get(confidence, "#4b5563")

    with st.container(border=True):
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown(f"### {rec.get('action', 'BUY')} {rec.get('symbol', '')}")
            st.caption(f"{rec.get('recommendation_id', '')} | {rec.get('status', '')}")
        with header_right:
            st.markdown(
                f"<div style='text-align:right; color:{confidence_color}; "
                f"font-weight:700;'>{confidence}</div>",
                unsafe_allow_html=True,
            )

        thesis = rec.get("thesis_plain") or "No thesis text available."
        risk = rec.get("risk_plain") or "No risk scenario available."
        st.write(thesis)

        price_col, target_col, stop_col, size_col = st.columns(4)
        price_col.metric("Entry", _money(rec.get("entry_price")))
        target_col.metric("Target", _money(rec.get("target_price")))
        stop_col.metric("Stop", _money(rec.get("stop_price")))
        size_col.metric("Size", _pct(rec.get("position_pct")))

        with st.expander("Risk scenario"):
            st.write(risk)


def _render_open_risk(api: Any) -> None:
    st.subheader("Open Risk")
    rows = _fetch_frame(
        api,
        """
        SELECT symbol, action, confidence, entry_price, target_price, stop_price,
               position_pct, time_horizon_days, created_at
        FROM recommendations
        WHERE status IN ('active', 'pending_approval')
        ORDER BY position_pct DESC, created_at DESC
        LIMIT 20
        """,
    )

    if rows.empty:
        st.info("No open recommendation risk to review.")
        return

    display = rows.copy()
    for col in ["entry_price", "target_price", "stop_price"]:
        display[col] = display[col].apply(_money)
    display["position_pct"] = display["position_pct"].apply(_pct)
    display = display.rename(
        columns={
            "symbol": "Symbol",
            "action": "Action",
            "confidence": "Confidence",
            "entry_price": "Entry",
            "target_price": "Target",
            "stop_price": "Stop",
            "position_pct": "Size",
            "time_horizon_days": "Horizon",
        }
    )
    st.dataframe(
        display[["Symbol", "Action", "Confidence", "Entry", "Target", "Stop", "Size", "Horizon"]],
        use_container_width=True,
        hide_index=True,
    )


def _render_track_record(api: Any) -> None:
    st.subheader("Track Record")
    history = _fetch_frame(
        api,
        """
        SELECT symbol, realized_return, status, closed_at
        FROM recommendations
        WHERE status IN ('closed_profit', 'closed_loss', 'closed_stopped', 'expired')
          AND realized_return IS NOT NULL
        ORDER BY closed_at DESC
        LIMIT 200
        """,
    )

    if history.empty:
        st.info("No closed recommendations yet. Track record will appear after outcomes close.")
        return

    returns = history["realized_return"].astype(float)
    wins = returns[returns > 0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Rate", f"{len(wins) / len(returns):.0%}")
    col2.metric("Average Return", f"{returns.mean():+.1%}")
    col3.metric("Closed Ideas", len(returns))
    col4.metric("Best", f"{returns.max():+.1%}")


def _render_daily_run_status(api: Any) -> None:
    st.subheader("Daily Local Run")

    logs = _fetch_frame(
        api,
        """
        SELECT source_id, started_at, completed_at, status, records_inserted, error_message
        FROM ingestion_log
        ORDER BY started_at DESC
        LIMIT 8
        """,
    )

    if logs.empty:
        st.info("No local job history yet. Use Open HRP or Enable Daily HRP from the project folder.")
        return

    display = logs.rename(
        columns={
            "source_id": "Job",
            "started_at": "Started",
            "completed_at": "Finished",
            "status": "Status",
            "records_inserted": "Rows",
            "error_message": "Error",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _fetch_scalar(api: Any, query: str) -> Any:
    try:
        row = api.fetchone_readonly(query)
        return row[0] if row else None
    except Exception:
        return None


def _fetch_frame(api: Any, query: str) -> pd.DataFrame:
    try:
        return api.query_readonly(query)
    except Exception:
        return pd.DataFrame()


def _freshness(latest_price_date: Any) -> tuple[str, str]:
    if latest_price_date is None:
        return "No data", "No price data has been loaded yet."

    if isinstance(latest_price_date, str):
        latest = date.fromisoformat(latest_price_date[:10])
    elif isinstance(latest_price_date, datetime):
        latest = latest_price_date.date()
    else:
        latest = latest_price_date

    age = (date.today() - latest).days
    if age <= 1:
        return "Fresh", f"Latest price date: {latest}"
    if age <= 5:
        return f"{age}d old", f"Latest price date: {latest}"
    return "Stale", f"Latest price date: {latest}"


def _format_datetime(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, datetime):
        if value.date() == date.today():
            return value.strftime("%H:%M")
        return value.strftime("%b %d")
    text = str(value)
    return text[:16] if text else "None"


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def _trading_mode() -> str:
    import os

    if os.getenv("HRP_TRADING_DRY_RUN", "true").lower() == "true":
        return "Dry run"
    return "Live enabled"
