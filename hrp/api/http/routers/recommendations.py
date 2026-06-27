"""Recommendation endpoints (conviction list, detail, approve/reject)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from hrp.api.http.deps import df_to_records, get_api
from hrp.api.http.schemas import (
    ApprovalResult,
    ApproveRequest,
    Provenance,
    RecommendationDetail,
    RecommendationSummary,
    RejectRequest,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _detail_from_row(row: dict[str, Any]) -> RecommendationDetail:
    """Map a recommendation DB row to the output contract."""
    return RecommendationDetail(
        recommendation_id=row.get("recommendation_id"),
        symbol=row.get("symbol"),
        action=row.get("action"),
        confidence=row.get("confidence"),
        signal_strength=row.get("signal_strength"),
        entry_price=row.get("entry_price"),
        target_price=row.get("target_price"),
        stop_price=row.get("stop_price"),
        position_pct=row.get("position_pct"),
        thesis=row.get("thesis_plain") or row.get("thesis"),
        risks=row.get("risk_plain") or row.get("risks"),
        time_horizon_days=row.get("time_horizon_days"),
        status=row.get("status"),
        created_at=str(row["created_at"]) if row.get("created_at") is not None else None,
        closed_at=str(row["closed_at"]) if row.get("closed_at") is not None else None,
        provenance=Provenance(
            hypothesis_id=row.get("hypothesis_id"),
            model_name=row.get("model_name"),
            validation_status=row.get("validation_status"),
        ),
    )


@router.get("", response_model=list[RecommendationSummary])
def list_recommendations(
    status: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
    api=Depends(get_api),
) -> list[RecommendationSummary]:
    """The conviction list: active/pending/closed recommendations."""
    df = api.get_recommendations(status=status, symbol=symbol, limit=limit)
    return [RecommendationSummary(**r) for r in df_to_records(df)]


@router.get("/history", response_model=list[RecommendationSummary])
def recommendation_history(limit: int = 100, api=Depends(get_api)) -> list[RecommendationSummary]:
    df = api.get_recommendation_history(limit=limit)
    return [RecommendationSummary(**r) for r in df_to_records(df)]


@router.post("/approve-all", response_model=list[ApprovalResult])
def approve_all(body: ApproveRequest, api=Depends(get_api)) -> list[ApprovalResult]:
    results = api.approve_all_recommendations(actor=body.actor, dry_run=body.dry_run)
    return [ApprovalResult(**r) for r in results]


@router.get("/{recommendation_id}", response_model=RecommendationDetail)
def get_recommendation(recommendation_id: str, api=Depends(get_api)) -> RecommendationDetail:
    row = api.get_recommendation_by_id(recommendation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return _detail_from_row(row)


@router.post("/{recommendation_id}/approve", response_model=ApprovalResult)
def approve(recommendation_id: str, body: ApproveRequest, api=Depends(get_api)) -> ApprovalResult:
    result = api.approve_recommendation(recommendation_id, actor=body.actor, dry_run=body.dry_run)
    return ApprovalResult(**result)


@router.post("/{recommendation_id}/reject", response_model=ApprovalResult)
def reject(recommendation_id: str, body: RejectRequest, api=Depends(get_api)) -> ApprovalResult:
    result = api.reject_recommendation(recommendation_id, actor=body.actor, reason=body.reason)
    return ApprovalResult(**result)
