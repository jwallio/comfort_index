from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from nicewx.validation.verify_benchmark import (
    _resolved_cases,
    format_benchmark_table,
    run_verification_benchmark,
)
from nicewx.validation.verify_benchmark_cases import DEFAULT_VERIFICATION_BENCHMARK_CASES, VerificationBenchmarkCase


def test_resolved_cases_cover_required_regions_and_multiple_dates() -> None:
    regions = {case.region_name for case in DEFAULT_VERIFICATION_BENCHMARK_CASES}
    dates = {case.valid_date for case in DEFAULT_VERIFICATION_BENCHMARK_CASES}

    assert {"southeast", "southwest", "plains", "northeast"}.issubset(regions)
    assert len(dates) >= 2

    override_cases = _resolved_cases(date(2026, 3, 20))
    assert {case.valid_date for case in override_cases} == {date(2026, 3, 20)}
    assert {case.region_name for case in override_cases} == {"southeast", "southwest", "plains", "northeast"}


def test_run_verification_benchmark_collects_summary_rows(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_verification(**kwargs):
        valid_date = kwargs["valid_date"]
        region_name = kwargs["region_name"]
        return {
            "forecast_score_map": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_forecast.png",
            "analysis_score_map": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_analysis.png",
            "score_difference_map": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_diff.png",
            "summary_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_summary.csv",
            "point_metrics_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_points.csv",
            "request_summary_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_request_summary.csv",
            "request_detail_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_request_detail.csv",
            "summary_record": {
                "valid_date": valid_date.isoformat(),
                "region_name": region_name,
                "score_bias_mean": 1.25,
                "score_mae": 5.5,
                "score_rmse": 7.1,
                "exact_category_agreement_fraction": 0.6,
                "near_category_agreement_fraction": 0.9,
            },
        }

    monkeypatch.setattr("nicewx.validation.verify_benchmark.run_verification", _fake_run_verification)
    frame = run_verification_benchmark(
        cases=[VerificationBenchmarkCase(region_name="southeast", valid_date=date(2026, 3, 20))],
        output_dir=tmp_path,
        mesh_profile="standard",
        forecast_model="gfs_seamless",
        forecast_run_hour_utc=12,
        forecast_lead_days=1,
    )

    assert len(frame) == 1
    assert frame.iloc[0]["region"] == "southeast"
    assert frame.iloc[0]["score_mae"] == 5.5
    assert "forecast_score_map_path" in frame.columns

    table = format_benchmark_table(frame)
    assert "southeast" in table
    assert "5.5" in table
