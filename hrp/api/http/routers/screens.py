"""Stock-screen endpoints: rank the universe by a feature at the latest date.

Config-driven — each screen is one feature + direction, run through a single
parametrized query against the (long-format) features store. Powered entirely by
the existing feature store; no new data sources.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from hrp.api.http.deps import df_to_records, get_api
from hrp.api.http.schemas import ScreenInfo, ScreenResult, ScreenRow

router = APIRouter(prefix="/screens", tags=["screens"])

# A symbol only qualifies for a time-series screen if its price history is
# contiguous up to the latest date (prior bar within this many days). This keeps
# gap artifacts out of momentum/RSI/etc. when data is stale or incomplete.
CONTINUITY_MAX_GAP_DAYS = 7

# key -> screen definition. `direction` is a fixed whitelist (never user input),
# so it is safe to interpolate; feature/limit are bound as query params.
# require_continuity: True for rolling/time-series features (corrupted by price
# gaps); False for point-in-time fundamentals.
SCREENS: dict[str, dict] = {
    "momentum": {
        "title": "Momentum",
        "subtitle": "Strongest 20-day price momentum",
        "feature": "momentum_20d",
        "direction": "DESC",
        "value_label": "Momentum 20d",
        "positive_only": False,
        "require_continuity": True,
    },
    "unusual-volume": {
        "title": "Unusual Volume",
        "subtitle": "Highest volume vs. its recent average",
        "feature": "volume_ratio",
        "direction": "DESC",
        "value_label": "Volume Ratio",
        "positive_only": False,
        "require_continuity": True,
    },
    "oversold": {
        "title": "Oversold",
        "subtitle": "Lowest 14-day RSI — potential mean-reversion",
        "feature": "rsi_14d",
        "direction": "ASC",
        "value_label": "RSI 14d",
        "positive_only": False,
        "require_continuity": True,
    },
    "strong-trend": {
        "title": "Strong Trend",
        "subtitle": "Highest ADX — strongest directional trend",
        "feature": "adx_14d",
        "direction": "DESC",
        "value_label": "ADX 14d",
        "positive_only": False,
        "require_continuity": True,
    },
    "above-trend": {
        "title": "Above Trend",
        "subtitle": "Furthest above the 200-day moving average",
        "feature": "price_to_sma_200d",
        "direction": "DESC",
        "value_label": "Price / 200d SMA",
        "positive_only": True,
        "require_continuity": True,
    },
    "low-volatility": {
        "title": "Low Volatility",
        "subtitle": "Calmest names by 20-day volatility",
        "feature": "volatility_20d",
        "direction": "ASC",
        "value_label": "Volatility 20d",
        "positive_only": True,
        "require_continuity": True,
    },
    "value": {
        "title": "Value",
        "subtitle": "Lowest P/E — cheapest on earnings",
        "feature": "pe_ratio",
        "direction": "ASC",
        "value_label": "P/E",
        "positive_only": True,  # ignore negative/zero P/E (loss-makers)
        "require_continuity": False,
    },
    "dividends": {
        "title": "Dividends",
        "subtitle": "Highest dividend yield",
        "feature": "dividend_yield",
        "direction": "DESC",
        "value_label": "Yield",
        "positive_only": True,
        "require_continuity": False,
    },
}


@router.get("", response_model=list[ScreenInfo])
def list_screens() -> list[ScreenInfo]:
    return [
        ScreenInfo(
            key=key,
            title=cfg["title"],
            subtitle=cfg["subtitle"],
            value_label=cfg["value_label"],
        )
        for key, cfg in SCREENS.items()
    ]


@router.get("/{screen}", response_model=ScreenResult)
def run_screen(screen: str, limit: int = 25, api=Depends(get_api)) -> ScreenResult:
    cfg = SCREENS.get(screen)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown screen: {screen}")

    limit = max(1, min(limit, 100))
    feature = cfg["feature"]
    direction = "ASC" if cfg["direction"] == "ASC" else "DESC"
    value_filter = "AND f.value > 0" if cfg["positive_only"] else ""

    # Continuity guard: for time-series features, only rank symbols whose price
    # history is contiguous up to the latest date (prior bar within N days), so a
    # data gap can't manufacture a spurious momentum/RSI/etc. value.
    continuity_filter = ""
    if cfg.get("require_continuity"):
        continuity_filter = f"""
          AND f.symbol IN (
              SELECT p.symbol FROM prices p
              WHERE p.date < (SELECT m FROM latest)
              GROUP BY p.symbol
              HAVING date_diff('day', MAX(p.date), (SELECT m FROM latest))
                     <= {CONTINUITY_MAX_GAP_DAYS}
          )"""

    sql = f"""
        WITH latest AS (
            SELECT MAX(date) AS m FROM features
            WHERE feature_name = ? AND version = 'v1'
        )
        SELECT f.symbol, s.name, s.sector, f.value, f.date AS as_of
        FROM features f
        JOIN symbols s ON f.symbol = s.symbol
        WHERE f.feature_name = ? AND f.version = 'v1'
          AND f.date = (SELECT m FROM latest)
          AND f.value IS NOT NULL
          {value_filter}
          {continuity_filter}
        ORDER BY f.value {direction}
        LIMIT ?
    """
    df = api.query_readonly(sql, (feature, feature, limit))
    records = df_to_records(df)

    rows = [
        ScreenRow(
            rank=i + 1,
            symbol=r.get("symbol"),
            name=r.get("name"),
            sector=r.get("sector"),
            value=r.get("value"),
        )
        for i, r in enumerate(records)
    ]
    as_of = records[0].get("as_of") if records else None

    return ScreenResult(
        screen=screen,
        title=cfg["title"],
        subtitle=cfg["subtitle"],
        value_label=cfg["value_label"],
        as_of=as_of,
        rows=rows,
    )
