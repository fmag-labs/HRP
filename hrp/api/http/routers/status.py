"""App status endpoint: data freshness + counts so the UI never shows a silent
empty chart. Reuses PlatformAPI.get_data_health_summary()."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from hrp.api.http.deps import get_api
from hrp.api.http.schemas import DataFreshness, Status

router = APIRouter(prefix="/status", tags=["status"])


def _count(api, sql: str) -> int:
    """Return the integer scalar from a COUNT-style query (0 if empty)."""
    df = api.query_readonly(sql, [])
    return 0 if df is None or df.empty else int(df.iloc[0, 0] or 0)


@router.get("", response_model=Status)
def get_status(api=Depends(get_api)) -> Status:
    summary = api.get_data_health_summary()
    freshness = summary.get("data_freshness", {})
    symbol_count = int(summary.get("symbol_count") or 0)
    is_fresh = bool(freshness.get("is_fresh", False))
    days_stale = freshness.get("days_stale")

    rec_count = _count(api, "SELECT COUNT(*) FROM recommendations")
    pos_count = _count(api, "SELECT COUNT(*) FROM live_positions")

    has_data = symbol_count > 0 and freshness.get("last_date") is not None
    ok = has_data and is_fresh

    if not has_data:
        message = "No market data loaded yet — run a data refresh to get started."
    elif not is_fresh:
        n = days_stale if days_stale is not None else "several"
        message = f"Market data is {n} days stale — run a refresh for current prices."
    else:
        message = "Data is up to date."

    return Status(
        ok=ok,
        message=message,
        data=DataFreshness(
            last_date=freshness.get("last_date"),
            days_stale=days_stale,
            is_fresh=is_fresh,
        ),
        symbol_count=symbol_count,
        recommendation_count=rec_count,
        position_count=pos_count,
    )
