"""LLM client factories for LangChain/LangGraph integration."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

from llm_kit.config import get_llm_settings


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_llm_settings()
    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.request_timeout,
    )


def get_vision_model(**kwargs: Any) -> ChatOpenAI:
    settings = get_llm_settings()
    return ChatOpenAI(
        base_url=settings.base_url,
        model=settings.vision_model,
        api_key=settings.api_key,
        temperature=kwargs.pop("temperature", settings.default_temperature),
        max_tokens=kwargs.pop("max_tokens", settings.default_max_tokens),
        timeout=settings.request_timeout,
        **kwargs,
    )


def get_textgen_model(**kwargs: Any) -> ChatOpenAI:
    settings = get_llm_settings()
    return ChatOpenAI(
        base_url=settings.base_url,
        model=settings.textgen_model,
        api_key=settings.api_key,
        temperature=kwargs.pop("temperature", settings.default_temperature),
        max_tokens=kwargs.pop("max_tokens", settings.default_max_tokens),
        timeout=settings.request_timeout,
        **kwargs,
    )


def get_agent_model(**kwargs: Any) -> ChatOpenAI:
    settings = get_llm_settings()
    return ChatOpenAI(
        base_url=settings.base_url,
        model=settings.agent_model,
        api_key=settings.api_key,
        temperature=kwargs.pop("temperature", settings.agent_temperature),
        max_tokens=kwargs.pop("max_tokens", settings.agent_max_tokens),
        timeout=settings.request_timeout,
        **kwargs,
    )
