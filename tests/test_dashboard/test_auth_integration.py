"""Integration tests for dashboard authentication."""

import pytest


def test_dashboard_auth_config_accessible(monkeypatch):
    """Test that auth config can be loaded in dashboard context."""
    monkeypatch.setenv("HRP_AUTH_ENABLED", "true")
    monkeypatch.setenv("HRP_AUTH_COOKIE_KEY", "test_key_32_characters_long!!!")

    from hrp.dashboard.auth import AuthConfig

    config = AuthConfig.from_env()
    assert config.enabled is True


def test_auth_required_matches_environment(monkeypatch):
    """Test that auth_required property works with environment."""
    from hrp.utils.config import reset_config, get_config, Environment

    # Test production requires auth
    monkeypatch.setenv("HRP_ENVIRONMENT", "production")
    reset_config()
    config = get_config()
    assert config.auth_required is True
    assert config.environment == Environment.PRODUCTION

    # Test development doesn't require auth
    monkeypatch.setenv("HRP_ENVIRONMENT", "development")
    reset_config()
    config = get_config()
    assert config.auth_required is False
    assert config.environment == Environment.DEVELOPMENT

    reset_config()


def test_get_authenticator_integration(monkeypatch, tmp_path):
    """Test full authenticator setup."""
    from hrp.dashboard.auth import AuthConfig, get_authenticator, add_user

    # Create a test users file with a user
    users_file = tmp_path / "users.yaml"
    add_user(users_file, "testuser", "test@example.com", "Test User", "password123")

    # Create config with valid settings
    config = AuthConfig(
        enabled=True,
        users_file=users_file,
        cookie_key="test_key_32_characters_long_enough",
        cookie_name="test_auth",
        cookie_expiry_days=1,
    )

    # Should return an authenticator instance
    authenticator = get_authenticator(config)
    assert authenticator is not None
