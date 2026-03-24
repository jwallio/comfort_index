from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from comfortwx.validation.verify_benchmark import (
    _apply_threshold_flags,
    _write_benchmark_charts,
    _write_benchmark_html_report,
    _write_region_summary,
    _write_region_summary_chart,
    _write_verification_site,
    _resolved_cases,
    format_benchmark_table,
    run_verification_benchmark,
)
from comfortwx.validation.verify_benchmark_cases import DEFAULT_VERIFICATION_BENCHMARK_CASES, VerificationBenchmarkCase


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
            "absolute_error_map": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_abs_error.png",
            "category_disagreement_map": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_category_disagreement.png",
            "missed_high_comfort_map": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_missed.png",
            "false_high_comfort_map": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_false.png",
            "summary_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_summary.csv",
            "point_metrics_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_points.csv",
            "component_metrics_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_components.csv",
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
                "temp_mae": 1.5,
                "dewpoint_mae": 2.0,
                "cloud_mae": 1.2,
                "precip_mae": 0.8,
                "reliability_score_mae": 3.5,
                "disruption_penalty_mae": 1.1,
            },
        }

    monkeypatch.setattr("comfortwx.validation.verify_benchmark.run_verification", _fake_run_verification)
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
    assert "absolute_error_map_path" in frame.columns
    assert "component_metrics_csv_path" in frame.columns
    assert bool(frame.iloc[0]["passes_benchmark_thresholds"]) is True

    table = format_benchmark_table(frame)
    assert "southeast" in table
    assert "5.5" in table


def test_apply_threshold_flags_marks_warn_cases() -> None:
    frame = pd.DataFrame(
        [
            {
                "region": "southeast",
                "date": "2026-03-20",
                "status": "ok",
                "score_bias_mean": 1.0,
                "score_mae": 6.0,
                "score_rmse": 7.0,
                "exact_category_agreement_fraction": 0.7,
                "near_category_agreement_fraction": 0.92,
            },
            {
                "region": "southwest",
                "date": "2026-03-20",
                "status": "ok",
                "score_bias_mean": 6.5,
                "score_mae": 9.0,
                "score_rmse": 10.0,
                "exact_category_agreement_fraction": 0.45,
                "near_category_agreement_fraction": 0.8,
            },
        ]
    )

    flagged = _apply_threshold_flags(frame)
    assert list(flagged["benchmark_threshold_status"]) == ["pass", "warn"]
    assert list(flagged["passes_benchmark_thresholds"]) == [True, False]


def test_write_benchmark_charts_and_html_report(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "region": "southeast",
                "date": "2026-01-15",
                "status": "ok",
                "score_bias_mean": 1.0,
                "score_mae": 5.0,
                "score_rmse": 6.5,
                "exact_category_agreement_fraction": 0.62,
                "near_category_agreement_fraction": 0.91,
                "missed_high_comfort_cell_count": 4,
                "false_high_comfort_cell_count": 2,
                "temp_mae": 1.5,
                "dewpoint_mae": 2.1,
                "cloud_mae": 1.8,
                "precip_mae": 0.9,
                "reliability_score_mae": 3.0,
                "disruption_penalty_mae": 1.2,
                "passes_benchmark_thresholds": True,
                "benchmark_threshold_status": "pass",
                "forecast_score_map_path": str(tmp_path / "forecast.png"),
                "analysis_score_map_path": str(tmp_path / "analysis.png"),
                "score_difference_map_path": str(tmp_path / "diff.png"),
                "absolute_error_map_path": str(tmp_path / "abs.png"),
                "category_disagreement_map_path": str(tmp_path / "category.png"),
                "missed_high_comfort_map_path": str(tmp_path / "missed.png"),
                "false_high_comfort_map_path": str(tmp_path / "false.png"),
                "component_metrics_csv_path": str(tmp_path / "components.csv"),
            },
            {
                "region": "southwest",
                "date": "2026-03-20",
                "status": "ok",
                "score_bias_mean": -2.0,
                "score_mae": 7.5,
                "score_rmse": 8.4,
                "exact_category_agreement_fraction": 0.54,
                "near_category_agreement_fraction": 0.9,
                "missed_high_comfort_cell_count": 8,
                "false_high_comfort_cell_count": 3,
                "temp_mae": 2.0,
                "dewpoint_mae": 2.8,
                "cloud_mae": 1.6,
                "precip_mae": 1.2,
                "reliability_score_mae": 4.0,
                "disruption_penalty_mae": 1.7,
                "passes_benchmark_thresholds": True,
                "benchmark_threshold_status": "pass",
                "forecast_score_map_path": str(tmp_path / "forecast2.png"),
                "analysis_score_map_path": str(tmp_path / "analysis2.png"),
                "score_difference_map_path": str(tmp_path / "diff2.png"),
                "absolute_error_map_path": str(tmp_path / "abs2.png"),
                "category_disagreement_map_path": str(tmp_path / "category2.png"),
                "missed_high_comfort_map_path": str(tmp_path / "missed2.png"),
                "false_high_comfort_map_path": str(tmp_path / "false2.png"),
                "component_metrics_csv_path": str(tmp_path / "components2.csv"),
            },
        ]
    )

    for filename in [
        "forecast.png",
        "analysis.png",
        "diff.png",
        "abs.png",
        "category.png",
        "missed.png",
        "false.png",
        "components.csv",
        "forecast2.png",
        "analysis2.png",
        "diff2.png",
        "abs2.png",
        "category2.png",
        "missed2.png",
        "false2.png",
        "components2.csv",
    ]:
        (tmp_path / filename).write_bytes(b"fake")

    charts = _write_benchmark_charts(summary, output_dir=tmp_path, stem="test")
    assert "mae_bar_chart" in charts
    assert all(path.exists() for path in charts.values())

    region_summary, region_summary_csv_path = _write_region_summary(summary, output_dir=tmp_path, stem="test")
    assert not region_summary.empty
    assert region_summary_csv_path is not None and region_summary_csv_path.exists()
    region_summary_chart = _write_region_summary_chart(region_summary, output_dir=tmp_path, stem="test")
    assert region_summary_chart is not None and region_summary_chart.exists()
    charts["region_summary_chart"] = region_summary_chart

    report_path = _write_benchmark_html_report(
        summary,
        charts=charts,
        region_summary=region_summary,
        region_summary_csv_path=region_summary_csv_path,
        output_dir=tmp_path,
        stem="test",
    )
    assert report_path.exists()
    html_text = report_path.read_text(encoding="utf-8")
    assert "Comfort Index Verification Benchmark" in html_text
    assert "diff.png" in html_text
    assert "Regional Rollup" in html_text

    summary_csv_path = tmp_path / "comfortwx_verify_benchmark_test.csv"
    summary.to_csv(summary_csv_path, index=False)

    site_index = _write_verification_site(
        summary=summary,
        summary_path=summary_csv_path,
        charts=charts,
        report_path=report_path,
        region_summary_csv_path=region_summary_csv_path,
        output_dir=tmp_path,
        stem="test",
    )
    assert site_index.exists()
    assert (tmp_path / "verification_site" / "latest" / "index.html").exists()
