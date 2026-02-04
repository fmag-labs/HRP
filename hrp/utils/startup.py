"""Startup validation for HRP.

Provides fail-fast validation of required configuration and secrets.
"""

from __future__ import annotations

import os

from loguru import logger

from hrp.utils.secrets import validate_secrets


def validate_startup() -> list[str]:
    """
    Validate all required config at startup.

    Returns:
        List of error messages. Empty if all valid.
    """
    errors = []
    env = os.getenv("HRP_ENVIRONMENT", "development")

    # Validate secrets for the current environment
    valid, missing = validate_secrets(env)
    if not valid:
        for name in missing:
            errors.append(f"Missing required secret: {name}")

    return errors


def fail_fast_startup() -> None:
    """
    Validate startup and raise if invalid.

    Call this at application entry points (API, dashboard, CLI).

    Raises:
        RuntimeError: If required configuration is missing.
    """
    errors = validate_startup()
    if errors:
        error_msg = "Startup validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.debug("Startup validation passed")
