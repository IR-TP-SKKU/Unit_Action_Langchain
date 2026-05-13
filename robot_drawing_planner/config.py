"""Runtime configuration for the planner."""

from __future__ import annotations

import os

from robot_drawing_planner.schemas import Board

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BOARD = Board()


def get_openai_model() -> str:
    """Return the configured OpenAI model name."""

    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def require_openai_api_key() -> str:
    """Return OPENAI_API_KEY or raise the project-required message."""

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Make sure it is exported in ~/.zshrc."
        )
    return key

