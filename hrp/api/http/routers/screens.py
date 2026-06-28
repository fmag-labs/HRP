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

# key -> screen definition. `direction` is a fixed whitelist (never user input),
# so it is safe to interpolate; feature/limit are bound as query params.
SCREENS: dict[str, dict] = {
    "momentum": {
        "title": "Momentum",
        "subtitle": "Strongest 20-day price momentum",
        "feature": "momentum_20d",
        "direction": "DESC",
        "value_label": "Momentum 20d",
        "positive_only": False,
    },
    "value": {
        "title": "Value",
        "subtitle": "Lowest P/E — cheapest on earnings",
        "feature": "pe_ratio",
        "direction": "ASC",
        "value_label": "P/E",
        "positive_only": True,  # ignore negative/zero P/E (loss-makers)
    },
    "unusual-volume": {
        "title": "Unusual Volume",
        "subtitle": "Highest volume vs. its recent average",
        "feature": "volume_ratio",
        "direction": "DESC",
        "value_label": "Volume Ratio",
        "positive_only": False,
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
