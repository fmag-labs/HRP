"""Test ops server endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for ops server."""
    from hrp.ops.server import create_app

    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_timestamp(self, client):
        """Health endpoint should include timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
