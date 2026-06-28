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
    avg_cost: float | None = None  # mapped from live_positions.entry_price
    current_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None


class Portfolio(BaseModel):
    total_value: float
    position_count: int
    positions: list[Position]


class TrackRecordPeriod(BaseModel):
    """Mirrors the persisted ``track_record`` table; win_rate/closed are derived."""

    model_config = ConfigDict(extra="ignore")

    period_start: str | None = None
    period_end: str | None = None
    total_recommendations: int | None = None
    profitable: int | None = None
    unprofitable: int | None = None
    closed_recommendations: int | None = None  # derived: profitable + unprofitable
    win_rate: float | None = None  # derived: profitable / closed
    avg_return: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    best_pick: str | None = None
    worst_pick: str | None = None
    benchmark_return: float | None = None
    excess_return: float | None = None


class Settings(BaseModel):
    """Consumer settings, backed by the active ``user_profiles`` row."""

    model_config = ConfigDict(extra="ignore")

    profile_id: str | None = None
    name: str = "Default"
    risk_tolerance: int = 3  # 1 (conservative) .. 5 (aggressive)
    portfolio_value: float = 100000.0
    max_positions: int = 20
    max_position_pct: float = 0.10
    excluded_symbols: list[str] = Field(default_factory=list)
    excluded_sectors: list[str] = Field(default_factory=list)
    preferred_horizon: str = "medium"  # short | medium | long


class SettingsUpdate(BaseModel):
    name: str | None = None
    risk_tolerance: int | None = None
    portfolio_value: float | None = None
    max_positions: int | None = None
    max_position_pct: float | None = None
    excluded_symbols: list[str] | None = None
    excluded_sectors: list[str] | None = None
    preferred_horizon: str | None = None


class ScreenInfo(BaseModel):
    key: str
    title: str
    subtitle: str
    value_label: str


class ScreenRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int
    symbol: str
    name: str | None = None
    sector: str | None = None
    value: float | None = None


class ScreenResult(BaseModel):
    screen: str
    title: str
    subtitle: str
    value_label: str
    as_of: str | None = None
    rows: list[ScreenRow]


class DataFreshness(BaseModel):
    last_date: str | None = None
    days_stale: int | None = None
    is_fresh: bool = False


class Status(BaseModel):
    """Lightweight app status so the UI can warn instead of showing empty charts."""

    ok: bool
    message: str
    data: DataFreshness
    symbol_count: int = 0
    recommendation_count: int = 0
    position_count: int = 0


class ModelInfo(BaseModel):
    key: str
    label: str
    model: str
    available: bool


class AssistantQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: str | None = None  # registry key (claude|gpt|glm); None = default


class AssistantAnswer(BaseModel):
    answer: str
    remaining_today: int
    grounded_on: list[str] = Field(default_factory=list)
    model: str | None = None


class ConsultQuery(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    model: str | None = None


class ConsultAnswer(BaseModel):
    answer: str
    model: str
