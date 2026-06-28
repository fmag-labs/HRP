"""Vault Assistant endpoint: Claude grounded on the user's HRP vault data.

Mirrors The Assembly's assistant: answers cite vault data only, are rate-limited
per day, and carry a not-investment-advice disclaimer. Degrades gracefully when
``ANTHROPIC_API_KEY`` is unset (503) or the daily limit is hit (429).
"""

from __future__ import annotations

import os
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from hrp import llm
from hrp.api.http.deps import get_api
from hrp.api.http.schemas import AssistantAnswer, AssistantQuery, ModelInfo
from hrp.utils.rate_limiter import RateLimiter

router = APIRouter(prefix="/assistant", tags=["assistant"])

DAILY_LIMIT = int(os.getenv("HRP_ASSISTANT_DAILY_LIMIT", "30"))
# Token-bucket approximating a per-day quota (refills continuously over 24h).
RATE_LIMITER = RateLimiter(max_calls=DAILY_LIMIT, period=86400.0)

SYSTEM_PROMPT = (
    "You are the HRP Vault Assistant. Answer using ONLY the vault data provided "
    "below (the user's recommendations, portfolio, and track record). If the data "
    "does not contain the answer, say so plainly. Be concise and specific. "
    "Always end with: 'Not investment advice.'"
)


def build_context(api) -> tuple[str, list[str]]:
    """Assemble a compact grounding context from the user's vault data."""
    parts: list[str] = []
    sources: list[str] = []

    recs = api.get_recommendations(limit=15)
    if recs is not None and not recs.empty:
        sources.append("recommendations")
        parts.append("CURRENT RECOMMENDATIONS:\n" + recs.to_string(index=False))

    positions = api.get_live_positions()
    if positions is not None and not positions.empty:
        sources.append("portfolio")
        parts.append("PORTFOLIO POSITIONS:\n" + positions.to_string(index=False))

    end = date.today()
    tr = api.get_track_record(end - timedelta(days=365), end)
    if tr is not None and not tr.empty:
        sources.append("track_record")
        parts.append("TRACK RECORD:\n" + tr.to_string(index=False))

    return ("\n\n".join(parts) or "No vault data available."), sources


def answer(api, question: str, model_key: str) -> tuple[str, list[str]]:
    """Answer with the chosen model, grounded on the user's vault data."""
    context, sources = build_context(api)
    text = llm.complete(model_key, SYSTEM_PROMPT, f"{context}\n\nQUESTION: {question}")
    return text, sources


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    """List selectable models and whether each provider is configured."""
    return [ModelInfo(**m) for m in llm.list_models()]


@router.post("/query", response_model=AssistantAnswer)
def query(body: AssistantQuery, api=Depends(get_api)) -> AssistantAnswer:
    model_key = body.model or llm.default_model()
    try:
        llm.get_spec(model_key)
    except llm.LLMError:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    if not llm.is_available(model_key):
        raise HTTPException(
            status_code=503,
            detail=f"{model_key} is not configured (missing API key)",
        )
    if not RATE_LIMITER.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Daily assistant limit reached")
    try:
        text, sources = answer(api, body.question, model_key)
    except Exception as exc:  # log full detail server-side; return a clean message
        logger.error(f"Assistant query failed: {exc!r}")
        raise HTTPException(
            status_code=502,
            detail="The assistant is temporarily unavailable. Please try again later.",
        )
    return AssistantAnswer(
        answer=text,
        remaining_today=int(RATE_LIMITER.available_tokens),
        grounded_on=sources,
        model=model_key,
    )
