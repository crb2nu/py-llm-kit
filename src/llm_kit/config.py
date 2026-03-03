"""LLM configuration settings for OpenAI-compatible endpoints.

This module intentionally avoids `pydantic-settings` to keep the dependency
surface small for services and CI environments that use an internal package
index mirror.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value != "" else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str

    vision_model: str
    textgen_model: str
    agent_model: str

    default_temperature: float
    default_max_tokens: int

    agent_temperature: float
    agent_max_tokens: int

    cloud_fallback_enabled: bool
    openai_api_key: str | None
    openai_model: str

    request_timeout: int


def _load_settings() -> LLMSettings:
    base_url = (
        _env("LLM_BASE_URL", "http://litellm.ai.svc:8000/v1") or "http://litellm.ai.svc:8000/v1"
    )
    api_key = _env("LLM_API_KEY", "sk-local") or "sk-local"

    vision_model = _env("LLM_VISION_MODEL", "vision") or "vision"
    textgen_model = _env("LLM_TEXTGEN_MODEL", "textgen") or "textgen"
    agent_model = _env("LLM_AGENT_MODEL", "agent") or "agent"

    default_temperature = _env_float("LLM_DEFAULT_TEMPERATURE", 0.7)
    default_max_tokens = _env_int("LLM_DEFAULT_MAX_TOKENS", 2048)

    agent_temperature = _env_float("LLM_AGENT_TEMPERATURE", 0.3)
    agent_max_tokens = _env_int("LLM_AGENT_MAX_TOKENS", 4096)

    cloud_fallback_enabled = _env_bool("LLM_CLOUD_FALLBACK", False) or _env_bool(
        "LLM_CLOUD_FALLBACK_ENABLED", False
    )
    openai_api_key = _env("OPENAI_API_KEY", None) or _env("LLM_OPENAI_API_KEY", None)
    openai_model = _env("LLM_OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

    request_timeout = _env_int("LLM_REQUEST_TIMEOUT", 120)

    return LLMSettings(
        base_url=base_url,
        api_key=api_key,
        vision_model=vision_model,
        textgen_model=textgen_model,
        agent_model=agent_model,
        default_temperature=default_temperature,
        default_max_tokens=default_max_tokens,
        agent_temperature=agent_temperature,
        agent_max_tokens=agent_max_tokens,
        cloud_fallback_enabled=cloud_fallback_enabled,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        request_timeout=request_timeout,
    )


@lru_cache
def get_llm_settings() -> LLMSettings:
    return _load_settings()


def get_effective_base_url() -> str:
    return os.getenv("LLM_BASE_URL", get_llm_settings().base_url)


def is_local_llm_available() -> bool:
    settings = get_llm_settings()
    return "localhost" in settings.base_url or "litellm" in settings.base_url
