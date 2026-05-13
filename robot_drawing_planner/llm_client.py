"""LangChain ChatOpenAI client construction and structured parsing."""

from __future__ import annotations

from typing import Any, Protocol

from robot_drawing_planner.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    get_openai_model,
    require_openai_api_key,
)
from robot_drawing_planner.prompts import SYSTEM_PROMPT
from robot_drawing_planner.schemas import ParsedGoal


class Invokable(Protocol):
    """Small protocol used by real LangChain runnables and tests."""

    def invoke(self, input: Any) -> Any:
        ...


def build_chat_model() -> Any:
    """Build a deterministic ChatOpenAI client for live structured parsing."""

    require_openai_api_key()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=get_openai_model(),
        temperature=0,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )


def build_structured_parser(llm: Any | None = None) -> Invokable:
    """Return a LangChain runnable that emits ParsedGoal instances."""

    chat_model = llm if llm is not None else build_chat_model()
    return chat_model.with_structured_output(ParsedGoal, method="json_schema")


def parse_goal(command: str, parser: Invokable | None = None) -> ParsedGoal:
    """Parse a natural language command into a ParsedGoal."""

    if not command.strip():
        raise ValueError("Drawing command must not be empty.")

    structured_parser = parser if parser is not None else build_structured_parser()
    result = structured_parser.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", command),
        ]
    )
    if isinstance(result, ParsedGoal):
        return result
    return ParsedGoal.model_validate(result)

