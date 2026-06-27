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
    return [TrackRecordPeriod(**r) for r in df_to_records(df)]
