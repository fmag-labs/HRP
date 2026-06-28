"""Bearer-token authentication for the HTTP API.

Local-first by default: if ``HRP_API_TOKEN`` is unset, auth is disabled (the app
runs on your Mac for a single user). Set the env var to require
``Authorization: Bearer <token>`` on every protected route — for any deployment
beyond localhost.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def require_token(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing the bearer token when one is configured."""
    expected = os.getenv("HRP_API_TOKEN")
    if not expected:
        return  # auth disabled (local-first single-user mode)
    if authorization != f"Bearer {expected}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
