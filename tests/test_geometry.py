import math

import pytest

from robot_drawing_planner.geometry import ArcStroke, LineStroke, generate_strokes
from robot_drawing_planner.schemas import DrawingGoal, Point2D


def goal(kind, size_m=0.10, letter=None):
    return DrawingGoal(
        kind=kind,
        letter=letter,
        size_m=size_m,
        center=Point2D(x_m=0.0, y_m=0.0),
    )


def test_circle_generates_one_full_arc():
    strokes = generate_strokes(goal("circle"))
    assert len(strokes) == 1
    assert isinstance(strokes[0], ArcStroke)
    assert strokes[0].radius_m == pytest.approx(0.05)
    assert strokes[0].end_angle_rad == pytest.approx(2.0 * math.pi)


def test_square_generates_four_lines():
    strokes = generate_strokes(goal("square"))
    assert len(strokes) == 4
    assert all(isinstance(stroke, LineStroke) for stroke in strokes)
    assert strokes[0].start.x_m == pytest.approx(-0.05)
    assert strokes[0].start.y_m == pytest.approx(0.05)
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

