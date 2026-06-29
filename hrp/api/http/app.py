"""FastAPI application factory for the HRP consumer HTTP API."""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hrp import __version__
from hrp.api.http.auth import require_token
from hrp.api.http.routers import (
    assistant,
    consult,
    portfolio,
    recommendations,
    screens,
    settings,
    status,
    track_record,
)

API_PREFIX = "/api"


def create_app() -> FastAPI:
    """Build the HRP HTTP API app (mirrors hrp.ops.server.create_app)."""
    app = FastAPI(
        title="HRP API",
        description="Advisory HTTP/JSON API for the HRP consumer front-end",
        version=__version__,
    )

    # CORS for the SPA front-end (different origin/port). Configurable via
    # HRP_API_CORS_ORIGINS (comma-separated); defaults to the Next.js dev server.
    origins = [
        o.strip()
        for o in os.getenv("HRP_API_CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(f"{API_PREFIX}/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    protected = [Depends(require_token)]
    for module in (
        recommendations,
        portfolio,
        track_record,
        settings,
        assistant,
        status,
        screens,
        consult,
    ):
        app.include_router(module.router, prefix=API_PREFIX, dependencies=protected)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8090) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
