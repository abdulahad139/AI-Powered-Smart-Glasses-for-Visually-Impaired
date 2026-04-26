# =============================================================================
# Distance Estimation — Monocular distance calculation using width ratios
# =============================================================================
#
# Uses per-class width_ratio (calibrated via distance_calibration.py).
# Formula: distance_m = FOCAL_LENGTH / (pixel_width * width_ratio)
#
# =============================================================================

from config import (
    FOCAL_LENGTH,
    CLASS_WIDTH_RATIOS,
    DISTANCE_VERY_CLOSE,
)


def estimate_distance(label: str, pixel_width: float) -> float | None:
    """
    Return estimated distance in metres for a detected object.

    Uses the monocular formula:
        distance = FOCAL_LENGTH / (pixel_width * width_ratio)

    Returns None if the class is not calibrated or pixel_width is invalid.
    """
    if pixel_width <= 0:
        return None
    entry = CLASS_WIDTH_RATIOS.get(label.lower())
    if entry is None:
        return None
    ratio = entry.get("width_ratio", 0)
    if ratio <= 0:
        return None
    return FOCAL_LENGTH / (pixel_width * ratio)


def format_distance_spoken(distance_m: float | None) -> str:
    """
    Return a natural spoken distance phrase.

    Examples:
        0.6  → "less than one metre away"
        1.2  → "about one metre away"
        2.4  → "about 2 metres away"
        6.1  → "about 6 metres away"
    """
    if distance_m is None:
        return ""
    if distance_m < 1.0:
        return "less than one metre away"
    rounded = round(distance_m)
    if rounded == 1:
        return "about one metre away"
    return f"about {rounded} metres away"


def format_distance_label(distance_m: float | None) -> str:
    """Short on-screen label, e.g. '1.2m'. Used in bounding box text only."""
    if distance_m is None:
        return ""
    return f"{distance_m:.1f}m"


def is_very_close(distance_m: float | None) -> bool:
    """Check if distance is within the very close threshold."""
    if distance_m is None:
        return False
    return distance_m <= DISTANCE_VERY_CLOSE