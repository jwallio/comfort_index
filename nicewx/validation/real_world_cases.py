"""Default real-world validation cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RealWorldCase:
    """A point forecast validation case."""

    case_name: str
    lat: float
    lon: float
    date: date
    expected_label: str | None = None


DEFAULT_REAL_WORLD_CASES: tuple[RealWorldCase, ...] = (
    RealWorldCase("Raleigh", 35.78, -78.64, date(2026, 3, 24), "pleasant"),
    RealWorldCase("Miami", 25.76, -80.19, date(2026, 3, 24), "fair"),
    RealWorldCase("Denver", 39.74, -104.99, date(2026, 3, 24), "pleasant"),
    RealWorldCase("Seattle", 47.61, -122.33, date(2026, 3, 24), "poor"),
    RealWorldCase("Phoenix", 33.45, -112.07, date(2026, 3, 24), "pleasant"),
    RealWorldCase("San Diego", 32.72, -117.16, date(2026, 3, 24), "ideal"),
)
