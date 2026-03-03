"""Speculative decoding workflow with LangGraph."""

from __future__ import annotations

from inspect import isawaitable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from llm_kit.clients import get_agent_model, get_resilient_model, get_textgen_model


class SpecDecodeState(TypedDict):
    task: str
    system_prompt: str | None
    draft: str
    feedback: str
    iteration: int
    result: str
    approved: bool


def create_spec_decode_graph(max_iterations: int = 2) -> Any:
    draft_model = get_resilient_model(get_textgen_model(), name="draft_model")
    verify_model = get_resilient_model(get_agent_model(), name="verify_model")

    async def _call_model(model: Any, messages: list[dict[str, str]]) -> Any:
        # Test doubles may expose only sync `invoke`, while real models expose async call paths.
        instance_ainvoke = getattr(getattr(model, "__dict__", {}), "get", lambda _k: None)(
            "ainvoke"
        )
        supports_ainvoke = callable(instance_ainvoke) or callable(
            getattr(type(model), "ainvoke", None)
        )
        if supports_ainvoke:
            result = model.ainvoke(messages)
        else:
            result = model.invoke(messages)
        if isawaitable(result):
            return await result
        return result

    async def draft_node(state: SpecDecodeState) -> dict:
        system = state.get("system_prompt") or "You are a helpful assistant. Be concise."

        if state.get("feedback"):
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Original task: {state['task']}\n\n"
                        f"Your previous draft:\n{state['draft']}\n\n"
                        f"Feedback from reviewer:\n{state['feedback']}\n\n"
                        "Please revise your response to address the feedback."
                    ),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": state["task"]},
            ]

        response = await _call_model(draft_model, messages)
        return {
            "draft": response.content,
            "iteration": state.get("iteration", 0) + 1,
        }

    class VerifyOutput(BaseModel):
        status: Literal["APPROVED", "REVISE"] = Field(
            description="Is the draft approved or does it need revision?"
        )
        feedback: str = Field(
            description="If REVISE, provide concise feedback. If APPROVED, leave empty."
        )

    async def verify_node(state: SpecDecodeState) -> dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "Review the draft response. verification output must follow the schema."
                ),
            },
            {
                "role": "user",
                "content": (f"Task: {state['task']}\n\nDraft Response:\n{state['draft']}"),
            },
        ]

        structured_llm = verify_model.with_structured_output(VerifyOutput)
        response = await _call_model(structured_llm, messages)

        if response.status == "APPROVED":
            return {
                "approved": True,
                "result": state["draft"],
                "feedback": "",
            }
        return {
            "approved": False,
            "feedback": response.feedback,
        }

    def should_continue(state: SpecDecodeState) -> str:
        if state.get("approved", False):
            return "end"
        if state.get("iteration", 0) >= max_iterations:
            return "accept"
        return "revise"

    def accept_node(state: SpecDecodeState) -> dict:
        return {"result": state["draft"], "approved": True}

    workflow = StateGraph(SpecDecodeState)
    workflow.add_node("draft", draft_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("accept", accept_node)
    workflow.set_entry_point("draft")
    workflow.add_edge("draft", "verify")
    workflow.add_conditional_edges(
        "verify",
        should_continue,
        {
            "revise": "draft",
            "accept": "accept",
            "end": END,
        },
    )
    workflow.add_edge("accept", END)
    return workflow.compile()


_default_workflow = None


def get_spec_decode_chain():
    global _default_workflow
    if _default_workflow is None:
        _default_workflow = create_spec_decode_graph()
    return _default_workflow


async def spec_decode(
    task: str,
    system_prompt: str | None = None,
    max_iterations: int = 2,
) -> str:
    chain = create_spec_decode_graph(max_iterations)
    result = await chain.ainvoke(
        {
            "task": task,
            "system_prompt": system_prompt,
            "draft": "",
            "feedback": "",
            "iteration": 0,
            "result": "",
            "approved": False,
        }
    )
    return result["result"]
