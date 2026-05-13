"""Command line interface for generating drawing plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_drawing_planner.planner import plan_from_text
from robot_drawing_planner.schemas import Board


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a natural language drawing command into robot primitive JSON."
    )
    parser.add_argument("command", nargs="+", help="Natural language drawing command.")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path where the JSON primitive plan will be written.",
    )
    parser.add_argument("--board-width-m", type=float, default=0.40)
    parser.add_argument("--board-height-m", type=float, default=0.30)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of indented JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = " ".join(args.command)
    board = Board(width_m=args.board_width_m, height_m=args.board_height_m)
    plan = plan_from_text(command, board=board)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.compact else 2
    output_path.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=indent) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

