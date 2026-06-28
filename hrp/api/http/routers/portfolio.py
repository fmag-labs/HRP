"""Portfolio endpoint: live positions + total value."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from hrp.api.http.deps import df_to_records, get_api, to_jsonable
from hrp.api.http.schemas import Portfolio, Position

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _position_from_row(row: dict) -> Position:
    """Map a live_positions row to the API contract (entry_price -> avg_cost)."""
    return Position(
        symbol=row.get("symbol"),
        quantity=row.get("quantity"),
        avg_cost=row.get("entry_price"),
        current_price=row.get("current_price"),
        market_value=row.get("market_value"),
        unrealized_pnl=row.get("unrealized_pnl"),
        unrealized_pnl_pct=row.get("unrealized_pnl_pct"),
    )


@router.get("", response_model=Portfolio)
def get_portfolio(api=Depends(get_api)) -> Portfolio:
    """Live-priced personal book: positions and aggregate NAV."""
    positions = [_position_from_row(r) for r in df_to_records(api.get_live_positions())]
    total_value = float(to_jsonable(api.get_portfolio_value()) or 0.0)
    return Portfolio(
        total_value=total_value,
        position_count=len(positions),
        positions=positions,
    )
