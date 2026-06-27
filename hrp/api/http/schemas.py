"""Pydantic response models for the HRP HTTP API.

These implement the *Recommendation Output Contract* from the consumer platform
plan: every pick serializes to a thesis, trade parameters, risks, supporting
data, and validation provenance. Models are intentionally lenient (most fields
optional) so real DuckDB rows serialize without coupling the HTTP layer to an
exact column set.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    """Why a pick exists — HRP's differentiator over hand-curated tips."""

    hypothesis_id: str | None = None
    model_name: str | None = None
    validation_status: str | None = None


class RecommendationSummary(BaseModel):
    """Compact recommendation row for list/history endpoints."""

    model_config = ConfigDict(extra="ignore")

    recommendation_id: str
    symbol: str
    action: str | None = None
    confidence: str | None = None
    signal_strength: float | None = None
    entry_price: float | None = None
    close_price: float | None = None
    realized_return: float | None = None
    status: str | None = None
    created_at: str | None = None
    closed_at: str | None = None


class RecommendationDetail(BaseModel):
    """Full recommendation implementing the output contract."""

    model_config = ConfigDict(extra="ignore")

    recommendation_id: str
    symbol: str
    action: str | None = None
    confidence: str | None = None
    signal_strength: float | None = None
    entry_price: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    position_pct: float | None = None
    thesis: str | None = Field(default=None, description="Plain-English thesis")
    risks: str | None = Field(default=None, description="Plain-English key risks")
    time_horizon_days: int | None = None
    status: str | None = None
    created_at: str | None = None
    closed_at: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class ApprovalResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recommendation_id: str | None = None
    action: str | None = None
    actor: str | None = None
    order_id: str | None = None
    reason: str | None = None
    message: str | None = None


class ApproveRequest(BaseModel):
    actor: str = "user"
    dry_run: bool = True


class RejectRequest(BaseModel):
    actor: str = "user"
    reason: str = ""


class Position(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str | None = None
    quantity: float | None = None
    avg_cost: float | None = None
    current_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None


class Portfolio(BaseModel):
    total_value: float
    position_count: int
    positions: list[Position]


class TrackRecordPeriod(BaseModel):
    model_config = ConfigDict(extra="ignore")

    period_start: str | None = None
    period_end: str | None = None
    total_recommendations: int | None = None
    closed_recommendations: int | None = None
    win_rate: float | None = None
    avg_return: float | None = None
    total_return: float | None = None
    benchmark_return: float | None = None
    excess_return: float | None = None
    sharpe_ratio: float | None = None
