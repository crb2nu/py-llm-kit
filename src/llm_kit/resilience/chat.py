"""Resilience wrappers for LangChain models."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from observability.tracing import instrument_llm
from resilience import CircuitBreaker, Retry, execute_with_retry


class ResilientChatModel:
    """Wrapper for ChatModel that adds CircuitBreaker, Retry, and Observability.
    
    This is not a full LangChain Runnable, but wraps the `invoke` method.
    """

    def __init__(
        self,
        model: BaseChatModel,
        circuit_breaker: CircuitBreaker | None = None,
        retry: Retry | None = None,
        name: str = "llm",
    ):
        self.model = model
        self.circuit_breaker = circuit_breaker
        self.retry = retry
        self.name = name

    @instrument_llm(model_key="model_name")
    async def invoke(self, input: list[BaseMessage] | str, **kwargs: Any) -> Any:
        """Invoke the model with resilience."""
        
        async def call_model() -> Any:
            return await self.model.ainvoke(input, **kwargs)

        # Apply Retry
        if self.retry:
            async def operation() -> Any:
                if self.circuit_breaker:
                    return await self.circuit_breaker.execute(call_model)
                return await call_model()
                
            return await self.retry.execute(operation)
        
        # Apply Circuit Breaker (no retry)
        if self.circuit_breaker:
            return await self.circuit_breaker.execute(call_model)
            
        # Direct call
        return await call_model()

    def with_structured_output(self, schema: Any) -> Any:
        """Pass through structured output config."""
        return self.model.with_structured_output(schema)
