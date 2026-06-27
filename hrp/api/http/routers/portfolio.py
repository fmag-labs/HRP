"""Portfolio endpoint: live positions + total value."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from hrp.api.http.deps import df_to_records, get_api, to_jsonable
from hrp.api.http.schemas import Portfolio, Position

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("", response_model=Portfolio)
def get_portfolio(api=Depends(get_api)) -> Portfolio:
    """Live-priced personal book: positions and aggregate NAV."""
    positions = [Position(**r) for r in df_to_records(api.get_live_positions())]
    total_value = float(to_jsonable(api.get_portfolio_value()) or 0.0)
    return Portfolio(
        total_value=total_value,
        position_count=len(positions),
        positions=positions,
    )
