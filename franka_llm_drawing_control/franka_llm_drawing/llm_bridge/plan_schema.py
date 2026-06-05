"""Small dataclasses for the LLM planner handoff format.

The existing LLM planner may use Pydantic internally. This offline control
package keeps its boundary lightweight and only depends on a JSON-compatible
mapping with an ``actions`` list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

SUPPORTED_ACTION_NAMES = {
    "move_to_start",
    "align_pen_orientation",
    "pen_down",
    "draw_line_to",
    "draw_line",
    "draw_arc",
    "pen_up",
}

UNIT_TO_METERS = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}


@dataclass(frozen=True)
class Point3D:
    """A point expressed in SI meters after parsing."""

    x: float
    y: float
    z: float = 0.0
    unit: str = "m"

    def to_array(self) -> np.ndarray:
        """Return the point as a shape ``(3,)`` NumPy array in meters."""

        scale = unit_scale(self.unit)
        return np.array([self.x, self.y, self.z], dtype=float) * scale


@dataclass(frozen=True)
class PrimitiveAction:
    """One symbolic drawing primitive from the LLM planner."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    frame: str = "board"
    stroke_id: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PrimitiveAction":
        if "name" not in data:
            raise ValueError("PrimitiveAction requires a 'name' field.")
        params = data.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError("PrimitiveAction.params must be a mapping.")
        return cls(
            name=str(data["name"]),
            params=dict(params),
            frame=str(data.get("frame", "board")),
            stroke_id=(
                None if data.get("stroke_id") is None else str(data.get("stroke_id"))
            ),
        )


@dataclass(frozen=True)
class DrawingPlan:
    """A JSON handoff plan containing board-frame primitive actions."""

    actions: list[PrimitiveAction]
    schema_version: str = "1.0"
    source_command: str = ""
    goal: dict[str, Any] | None = None
    strokes: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DrawingPlan":
        actions_raw = data.get("actions")
        if not isinstance(actions_raw, list):
            raise ValueError("DrawingPlan JSON requires an 'actions' list.")
        return cls(
            actions=[PrimitiveAction.from_mapping(item) for item in actions_raw],
            schema_version=str(data.get("schema_version", "1.0")),
            source_command=str(data.get("source_command", "")),
            goal=dict(data["goal"]) if isinstance(data.get("goal"), Mapping) else None,
            strokes=[dict(item) for item in data.get("strokes", [])],
            diagnostics=(
                dict(data["diagnostics"])
                if isinstance(data.get("diagnostics"), Mapping)
                else {}
            ),
        )


def unit_scale(unit: str) -> float:
    """Return a multiplier from the given length unit to meters."""

    try:
        return UNIT_TO_METERS[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported length unit: {unit!r}.") from exc


def point3d_from_mapping(data: Mapping[str, Any], default_z: float = 0.0) -> Point3D:
    """Parse a JSON point mapping into a ``Point3D``.

    Missing ``z`` is accepted because some planner internals represent strokes
    as 2D board coordinates.
    """

    if "x" not in data or "y" not in data:
        raise ValueError("Point mapping requires 'x' and 'y'.")
    return Point3D(
        x=float(data["x"]),
        y=float(data["y"]),
        z=float(data.get("z", default_z)),
        unit=str(data.get("unit", "m")),
    )
