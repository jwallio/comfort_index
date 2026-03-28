"""Configurable benchmark case list for proxy model verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from comfortwx.config import OPENMETEO_VERIFICATION_BENCHMARK_LEAD_DAYS


@dataclass(frozen=True)
class VerificationBenchmarkCase:
    region_name: str
    valid_date: date
    forecast_lead_days: int = 1


VERIFICATION_BENCHMARK_TIER_DEFAULT: str = "default"
VERIFICATION_BENCHMARK_TIER_FOCUSED_MAE: str = "focused-mae"
VERIFICATION_BENCHMARK_TIER_FULL_SEASONAL: str = "full-seasonal"
VERIFICATION_BENCHMARK_TIER_NDFD_WEST_COAST: str = "ndfd-west-coast"
VERIFICATION_BENCHMARK_TIERS: tuple[str, ...] = (
    VERIFICATION_BENCHMARK_TIER_DEFAULT,
    VERIFICATION_BENCHMARK_TIER_FOCUSED_MAE,
    VERIFICATION_BENCHMARK_TIER_FULL_SEASONAL,
    VERIFICATION_BENCHMARK_TIER_NDFD_WEST_COAST,
)

_DEFAULT_BENCHMARK_REGIONS: tuple[str, ...] = ("southeast", "southwest", "plains", "northeast")
_DEFAULT_BENCHMARK_DATES: tuple[date, ...] = (date(2026, 1, 15), date(2026, 3, 20))
_FOCUSED_MAE_BENCHMARK_REGIONS: tuple[str, ...] = ("southeast", "plains", "northeast")
_FOCUSED_MAE_BENCHMARK_DATES: tuple[date, ...] = (
    date(2025, 1, 15),
    date(2025, 3, 20),
    date(2025, 4, 20),
    date(2025, 6, 20),
    date(2025, 7, 20),
    date(2025, 8, 20),
    date(2025, 10, 20),
    date(2025, 11, 15),
)
_FULL_SEASONAL_BENCHMARK_REGIONS: tuple[str, ...] = (
    "west_coast",
    "southwest",
    "rockies",
    "plains",
    "southeast",
    "northeast",
    "great_lakes",
)
_FULL_SEASONAL_BENCHMARK_DATES: tuple[date, ...] = (
    date(2025, 1, 15),
    date(2025, 3, 20),
    date(2025, 5, 15),
    date(2025, 7, 20),
    date(2025, 9, 20),
    date(2025, 11, 15),
)
_NDFD_WEST_COAST_BENCHMARK_REGIONS: tuple[str, ...] = ("west_coast",)
_NDFD_WEST_COAST_BENCHMARK_DATES: tuple[date, ...] = (
    date(2024, 3, 20),
    date(2024, 5, 15),
    date(2024, 7, 20),
    date(2024, 8, 20),
)


def _build_cases(
    *,
    regions: tuple[str, ...],
    dates: tuple[date, ...],
    lead_days: tuple[int, ...] = OPENMETEO_VERIFICATION_BENCHMARK_LEAD_DAYS,
) -> tuple[VerificationBenchmarkCase, ...]:
    return tuple(
        VerificationBenchmarkCase(
            region_name=region_name,
            valid_date=valid_date,
            forecast_lead_days=lead_day,
        )
        for region_name in regions
        for valid_date in dates
        for lead_day in lead_days
    )

DEFAULT_VERIFICATION_BENCHMARK_CASES: tuple[VerificationBenchmarkCase, ...] = _build_cases(
    regions=_DEFAULT_BENCHMARK_REGIONS,
    dates=_DEFAULT_BENCHMARK_DATES,
)
FOCUSED_MAE_VERIFICATION_BENCHMARK_CASES: tuple[VerificationBenchmarkCase, ...] = _build_cases(
    regions=_FOCUSED_MAE_BENCHMARK_REGIONS,
    dates=_FOCUSED_MAE_BENCHMARK_DATES,
)
FULL_SEASONAL_VERIFICATION_BENCHMARK_CASES: tuple[VerificationBenchmarkCase, ...] = _build_cases(
    regions=_FULL_SEASONAL_BENCHMARK_REGIONS,
    dates=_FULL_SEASONAL_BENCHMARK_DATES,
)
NDFD_WEST_COAST_VERIFICATION_BENCHMARK_CASES: tuple[VerificationBenchmarkCase, ...] = _build_cases(
    regions=_NDFD_WEST_COAST_BENCHMARK_REGIONS,
    dates=_NDFD_WEST_COAST_BENCHMARK_DATES,
    lead_days=(1,),
)

VERIFICATION_BENCHMARK_CASE_SETS: dict[str, tuple[VerificationBenchmarkCase, ...]] = {
    VERIFICATION_BENCHMARK_TIER_DEFAULT: DEFAULT_VERIFICATION_BENCHMARK_CASES,
    VERIFICATION_BENCHMARK_TIER_FOCUSED_MAE: FOCUSED_MAE_VERIFICATION_BENCHMARK_CASES,
    VERIFICATION_BENCHMARK_TIER_FULL_SEASONAL: FULL_SEASONAL_VERIFICATION_BENCHMARK_CASES,
    VERIFICATION_BENCHMARK_TIER_NDFD_WEST_COAST: NDFD_WEST_COAST_VERIFICATION_BENCHMARK_CASES,
}


def get_benchmark_case_set(tier: str) -> tuple[VerificationBenchmarkCase, ...]:
    normalized_tier = tier.strip().lower()
    if normalized_tier not in VERIFICATION_BENCHMARK_CASE_SETS:
        raise ValueError(
            f"Unknown verification benchmark tier '{tier}'. "
            f"Available tiers: {', '.join(VERIFICATION_BENCHMARK_TIERS)}."
        )
    return VERIFICATION_BENCHMARK_CASE_SETS[normalized_tier]
