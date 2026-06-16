"""Geographic helpers: great-circle distance and timezone offsets for travel
fatigue adjustments (Layer 3)."""
from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points (degrees)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def approx_timezone_offset_hours(lon: float) -> float:
    """Rough timezone offset (hours from UTC) from longitude.

    Good enough for a fatigue penalty proportional to east/west travel; we do
    not need civil timezone boundaries, just the magnitude of the body-clock
    shift between two venues.
    """
    return lon / 15.0
