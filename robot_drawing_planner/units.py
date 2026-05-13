"""Unit conversion helpers."""

from __future__ import annotations

SUPPORTED_UNITS_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
}


def normalize_unit(unit: str) -> str:
    """Normalize a length unit and verify that it is supported."""

    normalized = unit.strip().lower()
    if normalized not in SUPPORTED_UNITS_TO_METERS:
        supported = ", ".join(sorted({"m", "cm", "mm"}))
        raise ValueError(f"Unsupported unit '{unit}'. Supported units: {supported}.")
    return normalized


def to_meters(value: float, unit: str) -> float:
    """Convert a positive scalar length to meters."""

    if value <= 0:
        raise ValueError("Size must be positive.")
    return value * SUPPORTED_UNITS_TO_METERS[normalize_unit(unit)]


def coordinate_to_meters(value: float, unit: str) -> float:
    """Convert a signed coordinate to meters."""

    return value * SUPPORTED_UNITS_TO_METERS[normalize_unit(unit)]

