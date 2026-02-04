"""FastAPI ops server with health endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create FastAPI application for ops endpoints."""
    app = FastAPI(
        title="HRP Ops",
        description="Health and metrics endpoints for HRP",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Liveness probe - returns 200 if API is responsive."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }

    return app


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the ops server with uvicorn."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)
