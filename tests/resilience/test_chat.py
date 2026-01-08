"""Tests for resilience integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage

from llm_kit.resilience.chat import ResilientChatModel
from resilience import CircuitBreaker, CircuitBreakerConfig, Retry, RetryConfig, CircuitBreakerOpenError


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value="mock response")
    return model


@pytest.mark.asyncio
async def test_resilient_chat_model_happy_path(mock_model):
    """Test standard invocation works."""
    model = ResilientChatModel(mock_model)
    result = await model.invoke("test")
    assert result == "mock response"
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_resilient_chat_model_circuit_breaker(mock_model):
    """Test circuit breaker integration."""
    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
    model = ResilientChatModel(mock_model, circuit_breaker=breaker)

    # Fail once
    mock_model.ainvoke.side_effect = ValueError("error")
    with pytest.raises(ValueError):
        await model.invoke("test")

    # Circuit should be open
    with pytest.raises(CircuitBreakerOpenError):
        await model.invoke("test")


@pytest.mark.asyncio
async def test_resilient_chat_model_retry(mock_model):
    """Test retry integration."""
    retry = Retry(RetryConfig(max_attempts=2))
    model = ResilientChatModel(mock_model, retry=retry)

    # Fail once then succeed
    mock_model.ainvoke.side_effect = [ValueError("error"), "success"]
    
    result = await model.invoke("test")
    assert result == "success"
    assert mock_model.ainvoke.call_count == 2
