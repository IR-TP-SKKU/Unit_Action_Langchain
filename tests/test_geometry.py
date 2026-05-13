import math

import pytest

from robot_drawing_planner.geometry import generate_strokes
from robot_drawing_planner.schemas import ArcStroke, LineStroke, NormalizedGoal, Point2D


def goal(shape_type, size_m=0.10, letter=None):
    return NormalizedGoal(
        shape_type=shape_type,
        letter=letter,
        radius_m=size_m / 2.0 if shape_type in {"circle", "letter"} and letter == "O" else None,
        side_length_m=size_m if shape_type in {"square", "triangle"} else None,
        size_m=size_m,
        orientation_rad=0.0,
        center=Point2D(x=0.0, y=0.0),
        assumptions=[],
        warnings=[],
    )


def test_circle_generates_one_full_arc():
    circle_goal = NormalizedGoal(
        shape_type="circle",
        center=Point2D(x=0.0, y=0.0),
        radius_m=0.05,
        side_length_m=None,
        size_m=0.10,
        orientation_rad=0.0,
        letter=None,
        assumptions=[],
        warnings=[],
    )
    strokes = generate_strokes(circle_goal)
    assert len(strokes) == 1
    assert isinstance(strokes[0], ArcStroke)
    assert strokes[0].radius_m == pytest.approx(0.05)
    assert strokes[0].end_angle_rad == pytest.approx(2.0 * math.pi)
    assert strokes[0].direction == "ccw"


def test_square_generates_four_lines():
    strokes = generate_strokes(goal("square"))
    assert len(strokes) == 4
    assert all(isinstance(stroke, LineStroke) for stroke in strokes)
    assert strokes[0].start.x == pytest.approx(-0.05)
    assert strokes[0].start.y == pytest.approx(0.05)
    assert strokes[-1].end == strokes[0].start


def test_triangle_generates_three_lines():
    strokes = generate_strokes(goal("triangle"))
    assert len(strokes) == 3
    assert all(isinstance(stroke, LineStroke) for stroke in strokes)
    assert strokes[-1].end == strokes[0].start


@pytest.mark.parametrize("letter,expected_count", [("A", 3), ("H", 3), ("L", 2), ("T", 2)])
def test_supported_line_letters(letter, expected_count):
    strokes = generate_strokes(goal("letter", letter=letter))
    assert len(strokes) == expected_count
    assert all(isinstance(stroke, LineStroke) for stroke in strokes)


def test_letter_o_uses_circular_arc():
    strokes = generate_strokes(goal("letter", letter="O"))
    assert len(strokes) == 1
    assert isinstance(strokes[0], ArcStroke)

