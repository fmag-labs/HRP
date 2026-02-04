"""Tests for secrets validation module."""

import pytest

from hrp.utils.secrets import (
    REQUIRED_SECRETS,
    SecretDefinition,
    log_secrets_status,
    validate_secrets,
)


class TestSecretDefinition:
    """Tests for SecretDefinition dataclass."""

    def test_secret_definition_structure(self):
        """SecretDefinition should have expected fields."""
        secret = SecretDefinition(
            name="Test API Key",
            env_var="TEST_API_KEY",
            required_in=["production"],
            description="For testing"
        )
        assert secret.name == "Test API Key"
        assert secret.env_var == "TEST_API_KEY"
        assert secret.required_in == ["production"]
        assert secret.description == "For testing"

    def test_secret_definition_multiple_environments(self):
        """SecretDefinition should support multiple environments."""
        secret = SecretDefinition(
            name="Multi-env Key",
            env_var="MULTI_KEY",
            required_in=["staging", "production"],
            description="Required in both"
        )
        assert "staging" in secret.required_in
        assert "production" in secret.required_in


class TestRequiredSecrets:
    """Tests for the REQUIRED_SECRETS list."""

    def test_required_secrets_list_not_empty(self):
        """REQUIRED_SECRETS should not be empty."""
        assert len(REQUIRED_SECRETS) > 0

    def test_required_secrets_have_valid_structure(self):
        """Each secret in REQUIRED_SECRETS should have required fields."""
        for secret in REQUIRED_SECRETS:
            assert isinstance(secret, SecretDefinition)
            assert secret.name
            assert secret.env_var
            assert isinstance(secret.required_in, list)
            assert secret.description

    def test_anthropic_api_key_in_list(self):
        """Anthropic API key should be in required secrets."""
        env_vars = [s.env_var for s in REQUIRED_SECRETS]
        assert "ANTHROPIC_API_KEY" in env_vars


class TestValidateSecrets:
    """Tests for validate_secrets function."""

    def test_validate_secrets_returns_missing(self, monkeypatch):
        """validate_secrets should return missing secrets for production."""
        # Clear all secret env vars
        for secret in REQUIRED_SECRETS:
            monkeypatch.delenv(secret.env_var, raising=False)

        valid, missing = validate_secrets("production")
        assert valid is False
        assert len(missing) > 0

    def test_validate_secrets_passes_when_set(self, monkeypatch):
        """validate_secrets should pass when all required secrets are set."""
        # Set all required production secrets
        for secret in REQUIRED_SECRETS:
            if "production" in secret.required_in:
                monkeypatch.setenv(secret.env_var, "test_value")

        valid, missing = validate_secrets("production")
        assert valid is True
        assert len(missing) == 0

    def test_validate_secrets_dev_requires_less(self, monkeypatch):
        """Development should require fewer secrets than production."""
        # Clear all secrets
        for secret in REQUIRED_SECRETS:
            monkeypatch.delenv(secret.env_var, raising=False)

        # Development should require fewer secrets
        valid_dev, missing_dev = validate_secrets("development")
        valid_prod, missing_prod = validate_secrets("production")
        assert len(missing_dev) <= len(missing_prod)

    def test_validate_secrets_empty_string_counts_as_missing(self, monkeypatch):
        """Empty string values should count as missing."""
        for secret in REQUIRED_SECRETS:
            if "production" in secret.required_in:
                monkeypatch.setenv(secret.env_var, "")

        valid, missing = validate_secrets("production")
        assert valid is False

    def test_validate_secrets_whitespace_only_counts_as_missing(self, monkeypatch):
        """Whitespace-only values should count as missing."""
        for secret in REQUIRED_SECRETS:
            if "production" in secret.required_in:
                monkeypatch.setenv(secret.env_var, "   ")

        valid, missing = validate_secrets("production")
        assert valid is False

    def test_validate_secrets_staging_env(self, monkeypatch):
        """Staging environment should check staging-required secrets."""
        # Clear all secrets
        for secret in REQUIRED_SECRETS:
            monkeypatch.delenv(secret.env_var, raising=False)

        valid, missing = validate_secrets("staging")
        # Staging should require at least some secrets
        staging_secrets = [s for s in REQUIRED_SECRETS if "staging" in s.required_in]
        if staging_secrets:
            assert len(missing) > 0


class TestLogSecretsStatus:
    """Tests for log_secrets_status function."""

    def test_log_secrets_status_runs_without_error_missing(self, monkeypatch):
        """log_secrets_status should run without error when secrets missing."""
        # Clear all secrets
        for secret in REQUIRED_SECRETS:
            monkeypatch.delenv(secret.env_var, raising=False)

        # Should not raise
        log_secrets_status("production")

    def test_log_secrets_status_runs_without_error_configured(self, monkeypatch):
        """log_secrets_status should run without error when all secrets configured."""
        # Set all required production secrets
        for secret in REQUIRED_SECRETS:
            if "production" in secret.required_in:
                monkeypatch.setenv(secret.env_var, "test_value")

        # Should not raise
        log_secrets_status("production")
