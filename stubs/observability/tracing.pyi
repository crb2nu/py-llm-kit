from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def instrument_llm(model_key: str = "model") -> Callable[[F], F]: ...