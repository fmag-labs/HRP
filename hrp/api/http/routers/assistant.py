"""Vault Assistant endpoint: Claude grounded on the user's HRP vault data.

Mirrors The Assembly's assistant: answers cite vault data only, are rate-limited
per day, and carry a not-investment-advice disclaimer. Degrades gracefully when
``ANTHROPIC_API_KEY`` is unset (503) or the daily limit is hit (429).
"""

from __future__ import annotations

import os
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException

from hrp.api.http.deps import get_api
from hrp.api.http.schemas import AssistantAnswer, AssistantQuery
from hrp.utils.rate_limiter import RateLimiter

router = APIRouter(prefix="/assistant", tags=["assistant"])

DAILY_LIMIT = int(os.getenv("HRP_ASSISTANT_DAILY_LIMIT", "30"))
ASSISTANT_MODEL = os.getenv("HRP_ASSISTANT_MODEL", "claude-sonnet-4-20250514")
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


def _make_client():
    import anthropic

    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def answer(api, question: str) -> tuple[str, list[str]]:
    """Call Claude with the grounding context. Returns (answer_text, sources)."""
    context, sources = build_context(api)
    client = _make_client()
    message = client.messages.create(
        model=ASSISTANT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{context}\n\nQUESTION: {question}"}],
    )
    return message.content[0].text, sources


@router.post("/query", response_model=AssistantAnswer)
def query(body: AssistantQuery, api=Depends(get_api)) -> AssistantAnswer:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Assistant unavailable: ANTHROPIC_API_KEY not configured",
        )
    if not RATE_LIMITER.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Daily assistant limit reached")
    try:
        text, sources = answer(api, body.question)
    except Exception as exc:  # surface upstream/model errors clearly
        raise HTTPException(status_code=502, detail=f"Assistant error: {exc}")
    return AssistantAnswer(
        answer=text,
        remaining_today=int(RATE_LIMITER.available_tokens),
        grounded_on=sources,
    )
