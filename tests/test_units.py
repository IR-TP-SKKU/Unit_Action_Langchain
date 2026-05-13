import pytest

from robot_drawing_planner.units import coordinate_to_meters, to_meters


def test_to_meters_supported_units():
    assert to_meters(1, "m") == pytest.approx(1.0)
    assert to_meters(12, "cm") == pytest.approx(0.12)
    assert to_meters(25, "mm") == pytest.approx(0.025)


def test_coordinate_to_meters_allows_signed_values():
    assert coordinate_to_meters(-10, "cm") == pytest.approx(-0.10)


def test_to_meters_rejects_non_positive_size():
    with pytest.raises(ValueError, match="positive"):
        to_meters(0, "cm")


def test_to_meters_rejects_unsupported_unit():
    with pytest.raises(ValueError, match="Unsupported unit"):
        to_meters(1, "inch")

