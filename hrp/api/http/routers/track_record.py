"""Track-record endpoint: win rate, returns, alpha vs benchmark by period."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query

from hrp.api.http.deps import df_to_records, get_api
from hrp.api.http.schemas import TrackRecordPeriod

router = APIRouter(prefix="/track-record", tags=["track-record"])


@router.get("", response_model=list[TrackRecordPeriod])
def get_track_record(
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    api=Depends(get_api),
) -> list[TrackRecordPeriod]:
    end = end or date.today()
    start = start or (end - timedelta(days=365))
    df = api.get_track_record(start, end)
    return [_period_from_row(r) for r in df_to_records(df)]


def _period_from_row(row: dict) -> TrackRecordPeriod:
    """Map a track_record row, deriving closed count and win rate."""
    profitable = row.get("profitable")
    unprofitable = row.get("unprofitable")
    closed = None
    win_rate = None
    if profitable is not None and unprofitable is not None:
        closed = int(profitable) + int(unprofitable)
        win_rate = (profitable / closed) if closed else None
    return TrackRecordPeriod(
        period_start=row.get("period_start"),
        period_end=row.get("period_end"),
        total_recommendations=row.get("total_recommendations"),
        profitable=profitable,
        unprofitable=unprofitable,
        closed_recommendations=closed,
        win_rate=win_rate,
        avg_return=row.get("avg_return"),
        avg_win=row.get("avg_win"),
        avg_loss=row.get("avg_loss"),
        best_pick=row.get("best_pick"),
        worst_pick=row.get("worst_pick"),
        benchmark_return=row.get("benchmark_return"),
        excess_return=row.get("excess_return"),
    )
