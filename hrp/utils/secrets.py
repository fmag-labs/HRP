"""
Secrets management utilities for HRP.

Provides validation and status reporting for required secrets by environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger


@dataclass
class SecretDefinition:
    """Definition of a required secret."""

    name: str
    env_var: str
    required_in: list[str]  # ['staging', 'production']
    description: str


REQUIRED_SECRETS = [
    SecretDefinition(
        name="Anthropic API Key",
        env_var="ANTHROPIC_API_KEY",
        required_in=["staging", "production"],
        description="Required for Claude agents"
    ),
    SecretDefinition(
        name="Resend API Key",
        env_var="RESEND_API_KEY",
        required_in=["production"],
        description="Required for email notifications"
    ),
    SecretDefinition(
        name="Polygon API Key",
        env_var="POLYGON_API_KEY",
        required_in=["staging", "production"],
        description="Required for market data"
    ),
]


def validate_secrets(environment: str) -> tuple[bool, list[str]]:
    """
    Validate required secrets are set for the given environment.

    Args:
        environment: Environment name ('development', 'staging', 'production')

    Returns:
        Tuple of (all_valid, list_of_missing_secret_names)
    """
    missing = []

    for secret in REQUIRED_SECRETS:
        if environment in secret.required_in:
            value = os.getenv(secret.env_var)
            if not value or not value.strip():
                missing.append(secret.name)

    return len(missing) == 0, missing


def log_secrets_status(environment: str) -> None:
    """
    Log warnings for missing secrets at startup.

    Args:
        environment: Current environment name
    """
    valid, missing = validate_secrets(environment)

    if not valid:
        logger.warning(f"Missing secrets for {environment} environment:")
        for name in missing:
            logger.warning(f"  - {name}")
    else:
        logger.info(f"All required secrets configured for {environment}")
