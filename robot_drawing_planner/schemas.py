"""Pydantic schemas for drawing goals and robot primitive actions."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictBaseModel(BaseModel):
    """Base model that rejects unplanned low-level robot fields."""

    model_config = ConfigDict(extra="forbid")


class Point2D(StrictBaseModel):
    """A point on the drawing board, expressed in board-frame meters."""

    x_m: float = Field(description="X coordinate in the drawing board frame, meters.")
    y_m: float = Field(description="Y coordinate in the drawing board frame, meters.")


class Board(StrictBaseModel):
    """Known planar drawing board dimensions."""

    width_m: float = Field(default=0.40, gt=0, description="Board width in meters.")
    height_m: float = Field(default=0.30, gt=0, description="Board height in meters.")
    origin: Literal["center"] = Field(
        default="center",
        description="The board coordinate origin. This package uses the board center.",
    )


class ParsedGoal(StrictBaseModel):
    """Structured goal returned by LangChain structured output."""

    object_type: str = Field(
        description="One of circle, square, triangle, or letter.",
        examples=["circle", "square", "triangle", "letter"],
    )
    letter: str | None = Field(
        default=None,
        description="Required only when object_type is letter. Supported: A, H, L, T, O.",
        examples=["A"],
    )
    size: float = Field(
        gt=0,
        description=(
            "Nominal drawing size. For circle/O this is diameter; square side; "
            "triangle side; letters height."
        ),
    )
    unit: str = Field(default="m", description="Unit for size: m, cm, or mm.")
    center_x: float = Field(
        default=0.0,
        description="Desired center x coordinate on the drawing board.",
    )
    center_y: float = Field(
        default=0.0,
        description="Desired center y coordinate on the drawing board.",
    )
    center_unit: str = Field(
        default="m",
        description="Unit for center_x and center_y: m, cm, or mm.",
    )

    @field_validator("object_type")
    @classmethod
    def normalize_object_type(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("letter")
    @classmethod
    def normalize_letter(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None

    @field_validator("unit", "center_unit")
    @classmethod
    def normalize_unit(cls, value: str) -> str:
        return value.strip().lower()


class DrawingGoal(StrictBaseModel):
    """Normalized drawing goal consumed by deterministic geometry."""

    kind: Literal["circle", "square", "triangle", "letter"]
    letter: str | None = Field(default=None, description="Supported letter when kind=letter.")
    size_m: float = Field(gt=0, description="Nominal drawing size in meters.")
    center: Point2D


class MoveToStartAction(StrictBaseModel):
    action: Literal["move_to_start"] = "move_to_start"
    target: Point2D


class AlignPenOrientationAction(StrictBaseModel):
    action: Literal["align_pen_orientation"] = "align_pen_orientation"
    orientation: Literal["board_normal"] = "board_normal"


class PenDownAction(StrictBaseModel):
    action: Literal["pen_down"] = "pen_down"
    contact: Literal["drawing_board"] = "drawing_board"


class DrawLineAction(StrictBaseModel):
    action: Literal["draw_line"] = "draw_line"
    start: Point2D
    end: Point2D


class DrawArcAction(StrictBaseModel):
    action: Literal["draw_arc"] = "draw_arc"
    start: Point2D
    end: Point2D
    center: Point2D
    radius_m: float = Field(gt=0)
    start_angle_rad: float
    end_angle_rad: float
    clockwise: bool = False


class PenUpAction(StrictBaseModel):
    action: Literal["pen_up"] = "pen_up"


RobotPrimitiveAction = Annotated[
    Union[
        MoveToStartAction,
        AlignPenOrientationAction,
        PenDownAction,
        DrawLineAction,
        DrawArcAction,
        PenUpAction,
    ],
    Field(discriminator="action"),
]


class PrimitivePlan(StrictBaseModel):
    """Handoff JSON for the kinematics and Isaac Sim teammate."""

    schema_version: Literal["1.0"] = "1.0"
    coordinate_frame: Literal["drawing_board_2d_m"] = "drawing_board_2d_m"
    board: Board = Field(default_factory=Board)
    goal: DrawingGoal
    actions: list[RobotPrimitiveAction]

