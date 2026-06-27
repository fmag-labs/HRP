"""HRP HTTP/JSON API.

FastAPI layer over ``hrp.api.platform.PlatformAPI`` that exposes the advisory
surface (recommendations, portfolio, track record) as JSON for the consumer
front-end. See ``docs/plans/2026-06-27-hrp-consumer-platform-plan.md``.
"""

from hrp.api.http.app import create_app, run_server

__all__ = ["create_app", "run_server"]
