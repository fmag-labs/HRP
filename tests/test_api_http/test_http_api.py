"""Tests for the HRP HTTP/JSON API (hrp.api.http).

Uses a stub PlatformAPI injected via dependency_overrides so the HTTP layer,
serialization, routing, and auth are exercised without a database.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hrp.api.http.app import create_app
from hrp.api.http.deps import get_api


class StubAPI:
    """Minimal stand-in for PlatformAPI returning canned data."""

    def get_recommendations(self, status=None, symbol=None, limit=100):
        return pd.DataFrame(
            [
                {
                    "recommendation_id": "REC-1",
                    "symbol": "NVDA",
                    "action": "BUY",
                    "confidence": "HIGH",
                    "signal_strength": 0.82,
                    "entry_price": 178.5,
                    "close_price": None,
                    "realized_return": np.nan,  # exercises NaN -> null
                    "status": status or "active",
                    "created_at": "2026-06-20T00:00:00",
                    "closed_at": None,
                }
            ]
        )

    def get_recommendation_history(self, limit=100):
        return self.get_recommendations()

    def get_recommendation_by_id(self, recommendation_id):
        if recommendation_id != "REC-1":
            return None
        return {
            "recommendation_id": "REC-1",
            "symbol": "NVDA",
            "action": "BUY",
            "confidence": "HIGH",
            "signal_strength": 0.82,
            "entry_price": 178.5,
            "target_price": 220.0,
            "stop_price": 160.0,
            "position_pct": 0.05,
            "thesis_plain": "Momentum + AI demand",
            "risk_plain": "Valuation, cyclicality",
            "time_horizon_days": 60,
            "status": "active",
            "hypothesis_id": "HYP-2026-007",
            "model_name": "lightgbm_v3",
            "created_at": "2026-06-20",
            "closed_at": None,
        }

    def approve_recommendation(self, recommendation_id, actor="user", dry_run=True):
        return {
            "recommendation_id": recommendation_id,
            "action": "approved",
            "actor": actor,
            "order_id": None if dry_run else "ORD-9",
            "message": "ok",
        }

    def reject_recommendation(self, recommendation_id, actor="user", reason=""):
        return {
            "recommendation_id": recommendation_id,
            "action": "rejected",
            "actor": actor,
            "reason": reason,
            "message": "ok",
        }

    def approve_all_recommendations(self, actor="user", dry_run=True):
        return [self.approve_recommendation("REC-1", actor, dry_run)]

    def get_live_positions(self, as_of_date=None):
        # Real live_positions columns: entry_price (not avg_cost).
        return pd.DataFrame(
            [
                {
                    "symbol": "NVDA",
                    "quantity": 50,
                    "entry_price": 178.5,
                    "current_price": 192.53,
                    "market_value": 9626.5,
                    "cost_basis": 8925.0,
                    "unrealized_pnl": 701.5,
                    "unrealized_pnl_pct": 0.0786,
                }
            ]
        )

    def get_portfolio_value(self):
        return Decimal("9626.50")

    def get_track_record(self, start_date, end_date):
        # Real track_record columns: profitable/unprofitable (no win_rate).
        return pd.DataFrame(
            [
                {
                    "period_start": str(start_date),
                    "period_end": str(end_date),
                    "total_recommendations": 10,
                    "profitable": 4,
                    "unprofitable": 2,
                    "avg_return": 0.041,
                    "avg_win": 0.08,
                    "avg_loss": -0.03,
                    "best_pick": "NVDA",
                    "worst_pick": "INTC",
                    "benchmark_return": 0.05,
                    "excess_return": 0.07,
                }
            ]
        )

    def get_data_health_summary(self):
        return {
            "symbol_count": 20,
            "data_freshness": {
                "last_date": "2026-06-27",
                "days_stale": 1,
                "is_fresh": True,
            },
        }

    # --- generic DB access used by settings endpoints ---
    def query_readonly(self, sql, params=None):
        return pd.DataFrame()  # default: no active profile

    def execute_write(self, sql, params=None):
        return 1


@pytest.fixture
def client():
    app = create_app()
    app.dependency_overrides[get_api] = lambda: StubAPI()
    return TestClient(app)


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestRecommendations:
    def test_list(self, client):
        r = client.get("/api/recommendations")
        assert r.status_code == 200
        body = r.json()
        assert body[0]["symbol"] == "NVDA"
        assert body[0]["realized_return"] is None  # NaN serialized to null

    def test_list_with_filters(self, client):
        r = client.get("/api/recommendations", params={"status": "active", "limit": 5})
        assert r.status_code == 200
        assert r.json()[0]["status"] == "active"

    def test_history(self, client):
        r = client.get("/api/recommendations/history")
        assert r.status_code == 200
        assert r.json()[0]["recommendation_id"] == "REC-1"

    def test_detail_maps_contract(self, client):
        r = client.get("/api/recommendations/REC-1")
        assert r.status_code == 200
        body = r.json()
        assert body["thesis"] == "Momentum + AI demand"
        assert body["risks"] == "Valuation, cyclicality"
        assert body["target_price"] == 220.0
        assert body["provenance"]["hypothesis_id"] == "HYP-2026-007"
        assert body["provenance"]["model_name"] == "lightgbm_v3"

    def test_detail_404(self, client):
        r = client.get("/api/recommendations/REC-MISSING")
        assert r.status_code == 404

    def test_approve(self, client):
        r = client.post("/api/recommendations/REC-1/approve", json={"dry_run": False})
        assert r.status_code == 200
        assert r.json()["order_id"] == "ORD-9"

    def test_reject(self, client):
        r = client.post("/api/recommendations/REC-1/reject", json={"reason": "too risky"})
        assert r.status_code == 200
        assert r.json()["reason"] == "too risky"

    def test_approve_all(self, client):
        r = client.post("/api/recommendations/approve-all", json={})
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert r.json()[0]["action"] == "approved"


class TestPortfolio:
    def test_portfolio(self, client):
        r = client.get("/api/portfolio")
        assert r.status_code == 200
        body = r.json()
        assert body["total_value"] == 9626.5
        assert body["position_count"] == 1
        pos = body["positions"][0]
        assert pos["symbol"] == "NVDA"
        # entry_price mapped to avg_cost (the caveat-1 bug fix)
        assert pos["avg_cost"] == 178.5
        assert pos["unrealized_pnl_pct"] == 0.0786


class TestTrackRecord:
    def test_track_record_derives_win_rate(self, client):
        r = client.get("/api/track-record")
        assert r.status_code == 200
        row = r.json()[0]
        # win_rate and closed are derived from profitable/unprofitable
        assert row["closed_recommendations"] == 6
        assert row["win_rate"] == pytest.approx(4 / 6)
        assert row["best_pick"] == "NVDA"

    def test_track_record_with_dates(self, client):
        r = client.get(
            "/api/track-record",
            params={"start": "2026-01-01", "end": "2026-06-01"},
        )
        assert r.status_code == 200
        assert r.json()[0]["period_start"] == "2026-01-01"


class TestSettings:
    def test_defaults_when_no_profile(self, client):
        # StubAPI.query_readonly returns empty -> real get_active_profile -> defaults
        r = client.get("/api/settings")
        assert r.status_code == 200
        body = r.json()
        assert body["risk_tolerance"] == 3
        assert body["max_positions"] == 20
        assert body["excluded_symbols"] == []

    def test_get_populated_profile(self, client, monkeypatch):
        from hrp.api.http.routers import settings as settings_mod

        monkeypatch.setattr(
            settings_mod,
            "get_active_profile",
            lambda api: {
                "profile_id": "P1",
                "name": "Me",
                "risk_tolerance": 4,
                "portfolio_value": 250000.0,
                "max_positions": 12,
                "max_position_pct": 0.08,
                "excluded_symbols": "TSLA,GME",
                "excluded_sectors": "Financials",
                "preferred_horizon": "long",
            },
        )
        r = client.get("/api/settings")
        assert r.status_code == 200
        body = r.json()
        assert body["risk_tolerance"] == 4
        assert body["excluded_symbols"] == ["TSLA", "GME"]
        assert body["preferred_horizon"] == "long"

    def test_update_settings(self, client, monkeypatch):
        from hrp.api.http.routers import settings as settings_mod

        captured = {}

        def fake_upsert(api, updates):
            captured.update(updates)
            return {
                "profile_id": "P1",
                "name": "Me",
                "risk_tolerance": updates.get("risk_tolerance", 3),
                "portfolio_value": 100000.0,
                "max_positions": 20,
                "max_position_pct": 0.10,
                "excluded_symbols": updates.get("excluded_symbols", ""),
                "excluded_sectors": "",
                "preferred_horizon": "medium",
            }

        monkeypatch.setattr(settings_mod, "upsert_profile", fake_upsert)
        r = client.put(
            "/api/settings",
            json={"risk_tolerance": 5, "excluded_symbols": ["TSLA"]},
        )
        assert r.status_code == 200
        assert r.json()["risk_tolerance"] == 5
        # list field CSV-encoded for the VARCHAR column
        assert captured["excluded_symbols"] == "TSLA"


class TestAssistant:
    def test_unavailable_without_key(self, client, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        r = client.post("/api/assistant/query", json={"question": "What do I own?"})
        assert r.status_code == 503

    def test_answers_with_default_model(self, client, monkeypatch):
        from hrp.api.http.routers import assistant as assistant_mod
        from hrp.utils.rate_limiter import RateLimiter

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(assistant_mod, "RATE_LIMITER", RateLimiter(5, 3600.0))
        monkeypatch.setattr(
            assistant_mod, "answer", lambda api, q, m: ("You own NVDA.", ["portfolio"])
        )
        r = client.post("/api/assistant/query", json={"question": "What do I own?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "You own NVDA."
        assert body["grounded_on"] == ["portfolio"]
        assert body["model"] == "claude"  # default
        assert body["remaining_today"] >= 0

    def test_select_model_is_passed_through(self, client, monkeypatch):
        from hrp.api.http.routers import assistant as assistant_mod
        from hrp.utils.rate_limiter import RateLimiter

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(assistant_mod, "RATE_LIMITER", RateLimiter(5, 3600.0))
        captured = {}

        def fake_answer(api, q, m):
            captured["model"] = m
            return ("hi from gpt", [])

        monkeypatch.setattr(assistant_mod, "answer", fake_answer)
        r = client.post("/api/assistant/query", json={"question": "hi", "model": "gpt"})
        assert r.status_code == 200
        assert r.json()["model"] == "gpt"
        assert captured["model"] == "gpt"

    def test_unknown_model_400(self, client):
        r = client.post("/api/assistant/query", json={"question": "hi", "model": "x"})
        assert r.status_code == 400

    def test_models_endpoint(self, client):
        r = client.get("/api/assistant/models")
        assert r.status_code == 200
        assert {m["key"] for m in r.json()} == {"claude", "gpt", "glm"}

    def test_rate_limited(self, client, monkeypatch):
        from hrp.api.http.routers import assistant as assistant_mod
        from hrp.utils.rate_limiter import RateLimiter

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(assistant_mod, "RATE_LIMITER", RateLimiter(1, 3600.0))
        monkeypatch.setattr(assistant_mod, "answer", lambda api, q, m: ("ok", []))
        first = client.post("/api/assistant/query", json={"question": "hi"})
        second = client.post("/api/assistant/query", json={"question": "hi"})
        assert first.status_code == 200
        assert second.status_code == 429

    def test_upstream_error_is_sanitized(self, client, monkeypatch):
        from hrp.api.http.routers import assistant as assistant_mod
        from hrp.utils.rate_limiter import RateLimiter

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(assistant_mod, "RATE_LIMITER", RateLimiter(5, 3600.0))

        def boom(api, q, m):
            raise RuntimeError("credit balance too low; request_id=req_secret123")

        monkeypatch.setattr(assistant_mod, "answer", boom)
        r = client.post("/api/assistant/query", json={"question": "hi"})
        assert r.status_code == 502
        # raw upstream text must NOT leak to the consumer
        assert "request_id" not in r.text
        assert "credit balance" not in r.text
        assert "temporarily unavailable" in r.json()["detail"]


class TestConsult:
    def test_models(self, client):
        r = client.get("/api/consult/models")
        assert r.status_code == 200
        assert {m["key"] for m in r.json()} == {"claude", "gpt", "glm"}

    def test_consult_answer(self, client, monkeypatch):
        import hrp.llm as llm_mod
        from hrp.api.http.routers import consult as consult_mod
        from hrp.utils.rate_limiter import RateLimiter

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(consult_mod, "RATE_LIMITER", RateLimiter(5, 3600.0))
        monkeypatch.setattr(llm_mod, "complete", lambda k, s, u: f"[{k}] answer")
        r = client.post("/api/consult", json={"question": "2+2?", "model": "gpt"})
        assert r.status_code == 200
        body = r.json()
        assert body["model"] == "gpt"
        assert body["answer"] == "[gpt] answer"

    def test_consult_unknown_model_400(self, client):
        r = client.post("/api/consult", json={"question": "hi", "model": "nope"})
        assert r.status_code == 400


class TestStatus:
    def test_status_ok(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["data"]["is_fresh"] is True
        assert body["symbol_count"] == 20
        assert "up to date" in body["message"].lower()

    def test_status_stale(self, client, monkeypatch):
        from hrp.api.http.routers import status as status_mod  # noqa: F401

        def stale_summary(self):
            return {
                "symbol_count": 20,
                "data_freshness": {
                    "last_date": "2026-06-20",
                    "days_stale": 6,
                    "is_fresh": False,
                },
            }

        monkeypatch.setattr(StubAPI, "get_data_health_summary", stale_summary)
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "stale" in body["message"].lower()
        assert body["data"]["days_stale"] == 6

    def test_status_no_data(self, client, monkeypatch):
        def empty_summary(self):
            return {
                "symbol_count": 0,
                "data_freshness": {"last_date": None, "days_stale": None, "is_fresh": False},
            }

        monkeypatch.setattr(StubAPI, "get_data_health_summary", empty_summary)
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "no market data" in body["message"].lower()


class TestScreens:
    def test_list_screens(self, client):
        r = client.get("/api/screens")
        assert r.status_code == 200
        keys = {s["key"] for s in r.json()}
        assert {"momentum", "value", "unusual-volume"} <= keys

    def test_run_screen_ranks_rows(self, client, monkeypatch):
        def fake_query(self, sql, params=None):
            assert "momentum_20d" in (params or ())  # screen feature is bound
            assert "FROM prices p" in sql  # continuity guard applied (time-series)
            return pd.DataFrame(
                [
                    {
                        "symbol": "NVDA",
                        "name": "NVIDIA",
                        "sector": "Tech",
                        "value": 0.21,
                        "as_of": "2026-06-26",
                    },
                    {
                        "symbol": "MELI",
                        "name": "MercadoLibre",
                        "sector": "Cons",
                        "value": 0.15,
                        "as_of": "2026-06-26",
                    },
                ]
            )

        monkeypatch.setattr(StubAPI, "query_readonly", fake_query)
        r = client.get("/api/screens/momentum", params={"limit": 10})
        assert r.status_code == 200
        body = r.json()
        assert body["screen"] == "momentum"
        assert body["as_of"] == "2026-06-26"
        assert body["rows"][0] == {
            "rank": 1,
            "symbol": "NVDA",
            "name": "NVIDIA",
            "sector": "Tech",
            "value": 0.21,
        }
        assert body["rows"][1]["rank"] == 2

    def test_fundamental_screen_has_no_continuity_guard(self, client, monkeypatch):
        captured = {}

        def fake_query(self, sql, params=None):
            captured["sql"] = sql
            return pd.DataFrame(
                [
                    {
                        "symbol": "AAPL",
                        "name": "Apple",
                        "sector": "Tech",
                        "value": 15.2,
                        "as_of": "2026-06-26",
                    }
                ]
            )

        monkeypatch.setattr(StubAPI, "query_readonly", fake_query)
        r = client.get("/api/screens/value")
        assert r.status_code == 200
        # point-in-time fundamental screens must NOT apply the price-continuity guard
        assert "FROM prices p" not in captured["sql"]

    def test_unknown_screen_404(self, client):
        r = client.get("/api/screens/does-not-exist")
        assert r.status_code == 404


class TestCors:
    def test_cors_header_for_configured_origin(self, client):
        # SPA on a different origin must receive an allow-origin header.
        r = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


class TestAuth:
    def test_no_token_required_when_unset(self, client, monkeypatch):
        monkeypatch.delenv("HRP_API_TOKEN", raising=False)
        assert client.get("/api/portfolio").status_code == 200

    def test_rejects_missing_token(self, client, monkeypatch):
        monkeypatch.setenv("HRP_API_TOKEN", "secret")
        assert client.get("/api/portfolio").status_code == 401

    def test_rejects_bad_token(self, client, monkeypatch):
        monkeypatch.setenv("HRP_API_TOKEN", "secret")
        r = client.get("/api/portfolio", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_accepts_valid_token(self, client, monkeypatch):
        monkeypatch.setenv("HRP_API_TOKEN", "secret")
        r = client.get("/api/portfolio", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200

    def test_health_never_requires_token(self, client, monkeypatch):
        monkeypatch.setenv("HRP_API_TOKEN", "secret")
        assert client.get("/api/health").status_code == 200
