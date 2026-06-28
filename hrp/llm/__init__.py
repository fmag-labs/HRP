"""Shared multi-provider LLM layer.

A small registry of model "specs" plus a single ``complete()`` entry point, used
by the Vault Assistant, the ``/api/consult`` endpoint, and ``hrp consult``.

Adding a provider = adding one registry entry. Model ids, endpoints, and API-key
env vars are all overridable via ``HRP_LLM_*`` env vars, so e.g. the exact GLM
model id or Z.ai endpoint can be set without code changes. Z.ai (GLM) is
OpenAI-compatible, so it rides on the OpenAI SDK with a custom ``base_url``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class LLMError(Exception):
    """Unknown model or unsupported provider."""


class LLMUnavailableError(LLMError):
    """The requested model's provider is not configured (missing API key)."""


@dataclass(frozen=True)
class LLMSpec:
    key: str
    label: str
    sdk: str  # "anthropic" | "openai"
    model: str
    api_key_env: str
    base_url: str | None = None  # for OpenAI-compatible providers (e.g. Z.ai)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def registry() -> dict[str, LLMSpec]:
    """Built fresh per call so env overrides (and tests) are always honored."""
    return {
        "claude": LLMSpec(
            key="claude",
            label="Claude (Anthropic)",
            sdk="anthropic",
            model=_env("HRP_LLM_CLAUDE_MODEL", "claude-sonnet-4-6"),
            api_key_env="ANTHROPIC_API_KEY",
        ),
        "gpt": LLMSpec(
            key="gpt",
            label="GPT (OpenAI)",
            sdk="openai",
            model=_env("HRP_LLM_GPT_MODEL", "gpt-4o"),
            api_key_env="OPENAI_API_KEY",
        ),
        "glm": LLMSpec(
            key="glm",
            label="GLM (Z.ai)",
            sdk="openai",
            model=_env("HRP_LLM_GLM_MODEL", "glm-4.5"),
            api_key_env="ZAI_API_KEY",
            base_url=_env("HRP_LLM_GLM_BASE_URL", "https://api.z.ai/api/paas/v4"),
        ),
    }


def default_model() -> str:
    return os.getenv("HRP_LLM_DEFAULT", "claude")


def get_spec(key: str) -> LLMSpec:
    reg = registry()
    if key not in reg:
        raise LLMError(f"Unknown model: {key}")
    return reg[key]


def is_available(key: str) -> bool:
    try:
        spec = get_spec(key)
    except LLMError:
        return False
    return bool(os.getenv(spec.api_key_env))


def list_models() -> list[dict]:
    """All registered models with an ``available`` flag (True when key is set)."""
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "model": spec.model,
            "available": bool(os.getenv(spec.api_key_env)),
        }
        for spec in registry().values()
    ]


def complete(model_key: str, system: str, user: str, max_tokens: int = 1024) -> str:
    """Call the chosen model with a system + user message; return the text reply.

    Raises LLMError (unknown model), LLMUnavailableError (provider not configured), or
    propagates the SDK's error for upstream/transport failures.
    """
    spec = get_spec(model_key)
    api_key = os.getenv(spec.api_key_env)
    if not api_key:
        raise LLMUnavailableError(f"{spec.label} is not configured ({spec.api_key_env} unset)")

    if spec.sdk == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=spec.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

    if spec.sdk == "openai":
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=spec.base_url)
        response = client.chat.completions.create(
            model=spec.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    raise LLMError(f"Unsupported provider sdk: {spec.sdk}")
