"""Tests for the shared multi-provider LLM registry (hrp.llm)."""

import pytest

import hrp.llm as llm


def test_list_models_has_three_providers():
    keys = {m["key"] for m in llm.list_models()}
    assert keys == {"claude", "gpt", "glm"}


def test_get_spec_unknown_raises():
    with pytest.raises(llm.LLMError):
        llm.get_spec("nope")


def test_is_available_reflects_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert llm.is_available("gpt") is True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm.is_available("gpt") is False
    assert llm.is_available("nope") is False


def test_complete_unknown_model_raises():
    with pytest.raises(llm.LLMError):
        llm.complete("nope", "s", "u")


def test_complete_unavailable_raises(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(llm.LLMUnavailableError):
        llm.complete("glm", "s", "u")


def test_env_overrides_model_id(monkeypatch):
    monkeypatch.setenv("HRP_LLM_GPT_MODEL", "gpt-custom")
    assert llm.get_spec("gpt").model == "gpt-custom"


def test_glm_uses_openai_sdk_with_base_url():
    spec = llm.get_spec("glm")
    assert spec.sdk == "openai"
    assert spec.base_url  # Z.ai OpenAI-compatible endpoint
