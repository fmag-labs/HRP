"""FastAPI application factory for the HRP consumer HTTP API."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from hrp.api.http.auth import require_token
from hrp.api.http.routers import portfolio, recommendations, track_record

API_PREFIX = "/api"


def create_app() -> FastAPI:
    """Build the HRP HTTP API app (mirrors hrp.ops.server.create_app)."""
    app = FastAPI(
        title="HRP API",
        description="Advisory HTTP/JSON API for the HRP consumer front-end",
        version="1.0.0",
    )

    @app.get(f"{API_PREFIX}/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    protected = [Depends(require_token)]
    for module in (recommendations, portfolio, track_record):
        app.include_router(module.router, prefix=API_PREFIX, dependencies=protected)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8090) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
