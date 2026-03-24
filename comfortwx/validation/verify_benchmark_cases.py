"""Configurable benchmark case list for proxy model verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class VerificationBenchmarkCase:
    region_name: str
    valid_date: date


DEFAULT_VERIFICATION_BENCHMARK_CASES: tuple[VerificationBenchmarkCase, ...] = (
    VerificationBenchmarkCase(region_name="southeast", valid_date=date(2026, 1, 15)),
    VerificationBenchmarkCase(region_name="southeast", valid_date=date(2026, 3, 20)),
    VerificationBenchmarkCase(region_name="southwest", valid_date=date(2026, 1, 15)),
    VerificationBenchmarkCase(region_name="southwest", valid_date=date(2026, 3, 20)),
    VerificationBenchmarkCase(region_name="plains", valid_date=date(2026, 1, 15)),
    VerificationBenchmarkCase(region_name="plains", valid_date=date(2026, 3, 20)),
    VerificationBenchmarkCase(region_name="northeast", valid_date=date(2026, 1, 15)),
    VerificationBenchmarkCase(region_name="northeast", valid_date=date(2026, 3, 20)),
)

