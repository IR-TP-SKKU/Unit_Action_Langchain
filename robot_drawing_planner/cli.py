"""Command line interface for generating drawing plans."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from robot_drawing_planner.agent_planner import plan_drawing_agentic
from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.llm_client import get_llm
from robot_drawing_planner.planner import build_plan_from_parsed_goal, plan_drawing
from robot_drawing_planner.schemas import Measurement, ParsedGoal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a natural language drawing command into robot primitive JSON. "
            "Default mode uses LLM agentic planning over Unit Action tools."
        )
    )
    parser.add_argument("command", help="Natural language drawing command.")
    parser.add_argument("--out", help="Optional path where JSON plan will be written.")
    parser.add_argument("--model", help="Optional OpenAI model override.")
    parser.add_argument(
        "--mode",
        choices=["agentic", "template"],
        default="agentic",
        help=(
            "Planning mode. Default 'agentic' uses LLM agentic planning over Unit Action "
            "tools. 'template' uses the old ParsedGoal/template compiler baseline."
        ),
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help=(
            "Use deterministic demo fallback for development/testing only; this never "
            "calls OpenAI and does not exercise production agentic planning."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument("--config", help="Optional JSON PlannerConfig file.")
    parser.add_argument(
        "--out-only",
        action="store_true",
        help="Write JSON to --out without printing it to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        if args.no_api:
            parsed = demo_parse_command(args.command)
            plan = build_plan_from_parsed_goal(parsed, config=config)
        elif args.mode == "template":
            llm = get_llm(args.model)
            plan = plan_drawing(args.command, config=config, llm=llm)
        else:
            llm = get_llm(args.model)
            plan = plan_drawing_agentic(args.command, config=config, llm=llm)

        json_text = json.dumps(
            plan.model_dump(mode="json"),
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )
        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_text + "\n", encoding="utf-8")
        if not args.out_only:
            print(json_text)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def demo_parse_command(command: str) -> ParsedGoal:
    """Small deterministic parser for tests and demos; production uses LangChain."""

    text = command.strip()
    lowered = text.lower()
    shape_type = _infer_shape_type(text)
    measurement = _extract_measurement(text)
    letter = _extract_letter(text) if shape_type == "letter" else None
    position_hint = _extract_position_hint(text)

    kwargs: dict[str, Any] = {
        "shape_type": shape_type,
        "center": None,
        "radius": None,
        "side_length": None,
        "size": None,
        "orientation_deg": 0.0,
        "letter": letter,
        "position_hint": position_hint,
        "raw_command": command,
    }

    if shape_type == "circle":
        if measurement is None:
            kwargs["radius"] = Measurement(value=5, unit="cm")
        elif _mentions_radius(text):
            kwargs["radius"] = measurement
        elif _mentions_diameter_or_size(text):
            kwargs["size"] = measurement
        else:
            kwargs["radius"] = measurement
    elif shape_type in {"square", "triangle"}:
        if measurement is None:
            kwargs["side_length"] = Measurement(value=10, unit="cm")
        elif _mentions_size(text) and not _mentions_side_length(text):
            kwargs["size"] = measurement
        else:
            kwargs["side_length"] = measurement
    elif shape_type == "letter":
        kwargs["size"] = measurement or Measurement(value=10, unit="cm")

    return ParsedGoal(**kwargs)


def _load_config(path: str | None) -> PlannerConfig:
    if path is None:
        return PlannerConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlannerConfig.model_validate(payload)


def _infer_shape_type(command: str) -> str:
    lowered = command.lower()
    if any(token in command for token in ["원", "동그라미"]) or "circle" in lowered:
        return "circle"
    if any(token in command for token in ["네모", "사각형", "정사각형"]) or "square" in lowered:
        return "square"
    if any(token in command for token in ["세모", "삼각형", "정삼각형"]) or "triangle" in lowered:
        return "triangle"
    if any(token in command for token in ["글자", "문자", "알파벳"]) or re.search(r"\bletter\b", lowered):
        return "letter"
    if re.search(r"\b[AHLTO]\b", command.upper()):
        return "letter"
    raise ValueError("Could not infer supported shape in --no-api mode.")


def _extract_letter(command: str) -> str:
    match = re.search(r"(?<![A-Z])([AHLTO])(?![A-Z])", command.upper())
    if match:
        return match.group(1)
    for letter in ["A", "H", "L", "T", "O"]:
        if f"letter {letter}".lower() in command.lower():
            return letter
    raise ValueError("--no-api letter commands must include one of A, H, L, T, O.")


def _extract_position_hint(command: str) -> str | None:
    lowered = command.lower()
    if "중앙" in command:
        return "중앙"
    if "center" in lowered:
        return "center"
    if "middle" in lowered:
        return "middle"
    return None


def _extract_measurement(command: str) -> Measurement | None:
    match = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)(?=$|[\s,.;:)\]]|[가-힣])",
        command,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit not in {"m", "cm", "mm"}:
        return None
    return Measurement(value=value, unit=unit)


def _mentions_radius(command: str) -> bool:
    lowered = command.lower()
    return "반지름" in command or "radius" in lowered


def _mentions_diameter_or_size(command: str) -> bool:
    lowered = command.lower()
    return any(token in command for token in ["지름", "크기"]) or any(
        token in lowered for token in ["diameter", "size"]
    )


def _mentions_side_length(command: str) -> bool:
    lowered = command.lower()
    return any(token in command for token in ["한 변", "변 길이"]) or any(
        token in lowered for token in ["side length", "side"]
    )


def _mentions_size(command: str) -> bool:
    lowered = command.lower()
    return "크기" in command or "size" in lowered


if __name__ == "__main__":
    raise SystemExit(main())
