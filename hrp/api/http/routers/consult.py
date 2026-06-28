"""Ad-hoc multi-LLM consult endpoint.

Unlike the Vault Assistant, this is NOT grounded on vault data — it's a plain
"ask any configured model a question" tool, shared with the ``hrp consult`` CLI.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from loguru import logger

from hrp import llm
from hrp.api.http.schemas import ConsultAnswer, ConsultQuery, ModelInfo
from hrp.utils.rate_limiter import RateLimiter

router = APIRouter(prefix="/consult", tags=["consult"])

DAILY_LIMIT = int(os.getenv("HRP_CONSULT_DAILY_LIMIT", "50"))
RATE_LIMITER = RateLimiter(max_calls=DAILY_LIMIT, period=86400.0)
SYSTEM_PROMPT = "You are a helpful, concise expert assistant."


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [ModelInfo(**m) for m in llm.list_models()]


@router.post("", response_model=ConsultAnswer)
def consult(body: ConsultQuery) -> ConsultAnswer:
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
        raise HTTPException(status_code=429, detail="Daily consult limit reached")
    try:
        text = llm.complete(model_key, SYSTEM_PROMPT, body.question)
    except Exception as exc:
        logger.error(f"Consult failed: {exc!r}")
        raise HTTPException(
            status_code=502,
            detail="The model is temporarily unavailable. Please try again later.",
        )
    return ConsultAnswer(answer=text, model=model_key)
