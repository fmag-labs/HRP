"""Test startup validation."""

import os
import pytest


class TestValidateStartup:
    """Tests for validate_startup function."""

    def test_returns_empty_list_in_development(self, monkeypatch):
        """No errors in development mode."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "development")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from hrp.utils.startup import validate_startup
        errors = validate_startup()
        assert errors == []

    def test_requires_anthropic_key_in_production(self, monkeypatch):
        """Production requires ANTHROPIC_API_KEY."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "production")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from hrp.utils.startup import validate_startup
        errors = validate_startup()
        # Note: error messages use human-friendly name from secrets.py
        assert any("Anthropic" in e for e in errors)

    def test_no_errors_when_secrets_present(self, monkeypatch):
        """No errors when required secrets are set."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "production")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Also set other required production secrets from secrets.py
        monkeypatch.setenv("RESEND_API_KEY", "test-key")
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")

        from hrp.utils.startup import validate_startup
        errors = validate_startup()
        assert errors == []


class TestFailFastStartup:
    """Tests for fail_fast_startup function."""

    def test_raises_on_missing_secrets(self, monkeypatch):
        """Should raise RuntimeError when secrets missing in production."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "production")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from hrp.utils.startup import fail_fast_startup
        with pytest.raises(RuntimeError, match="Startup validation failed"):
            fail_fast_startup()

    def test_passes_in_development(self, monkeypatch):
        """Should not raise in development."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "development")

        from hrp.utils.startup import fail_fast_startup
        fail_fast_startup()  # Should not raise
