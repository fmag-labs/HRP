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
        return pd.DataFrame(
            [
                {
                    "symbol": "NVDA",
                    "quantity": 50,
                    "avg_cost": 178.5,
                    "current_price": 192.53,
                    "market_value": 9626.5,
                    "unrealized_pnl": 701.5,
                }
            ]
        )

    def get_portfolio_value(self):
        return Decimal("9626.50")

    def get_track_record(self, start_date, end_date):
        return pd.DataFrame(
            [
                {
                    "period_start": str(start_date),
                    "period_end": str(end_date),
                    "total_recommendations": 10,
                    "closed_recommendations": 6,
                    "win_rate": 0.67,
                    "avg_return": 0.041,
                    "total_return": 0.12,
                    "benchmark_return": 0.05,
                    "excess_return": 0.07,
                    "sharpe_ratio": 1.3,
                }
            ]
        )


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
        assert body["positions"][0]["symbol"] == "NVDA"


class TestTrackRecord:
    def test_track_record(self, client):
        r = client.get("/api/track-record")
        assert r.status_code == 200
        assert r.json()[0]["win_rate"] == 0.67

    def test_track_record_with_dates(self, client):
        r = client.get(
            "/api/track-record",
            params={"start": "2026-01-01", "end": "2026-06-01"},
        )
        assert r.status_code == 200
        assert r.json()[0]["period_start"] == "2026-01-01"


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
