from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from comfortwx.validation.verify_benchmark import (
    _apply_score_calibration,
    _apply_threshold_flags,
    _parse_aggregation_policies,
    _build_calibration_summary,
    _write_aggregation_policy_summary,
    _priority_table,
    _parse_lead_days,
    _write_benchmark_charts,
    _write_component_priority_chart,
    _write_component_priority_summary,
    _write_benchmark_html_report,
    _write_lead_summary,
    _write_lead_summary_chart,
    _write_priority_case_chart,
    _write_priority_case_summary,
    _write_region_lead_heatmap,
    _write_region_lead_summary,
    _write_region_summary,
    _write_region_summary_chart,
    _write_verification_site,
    _resolved_cases,
    format_benchmark_table,
    run_verification_benchmark,
)
from comfortwx.validation.verify_benchmark_cases import DEFAULT_VERIFICATION_BENCHMARK_CASES, VerificationBenchmarkCase
from comfortwx.validation.verify_benchmark_cases import (
    FOCUSED_MAE_VERIFICATION_BENCHMARK_CASES,
    FULL_SEASONAL_VERIFICATION_BENCHMARK_CASES,
    NDFD_WEST_COAST_VERIFICATION_BENCHMARK_CASES,
    VERIFICATION_BENCHMARK_TIER_DEFAULT,
    VERIFICATION_BENCHMARK_TIER_FOCUSED_MAE,
    VERIFICATION_BENCHMARK_TIER_FULL_SEASONAL,
    VERIFICATION_BENCHMARK_TIER_NDFD_WEST_COAST,
)


def test_resolved_cases_cover_required_regions_and_multiple_dates() -> None:
    regions = {case.region_name for case in DEFAULT_VERIFICATION_BENCHMARK_CASES}
    dates = {case.valid_date for case in DEFAULT_VERIFICATION_BENCHMARK_CASES}
    lead_days = {case.forecast_lead_days for case in DEFAULT_VERIFICATION_BENCHMARK_CASES}
    focused_mae_regions = {case.region_name for case in FOCUSED_MAE_VERIFICATION_BENCHMARK_CASES}
    focused_mae_dates = {case.valid_date for case in FOCUSED_MAE_VERIFICATION_BENCHMARK_CASES}
    full_seasonal_regions = {case.region_name for case in FULL_SEASONAL_VERIFICATION_BENCHMARK_CASES}
    ndfd_regions = {case.region_name for case in NDFD_WEST_COAST_VERIFICATION_BENCHMARK_CASES}
    ndfd_leads = {case.forecast_lead_days for case in NDFD_WEST_COAST_VERIFICATION_BENCHMARK_CASES}

    assert {"southeast", "southwest", "plains", "northeast"}.issubset(regions)
    assert len(dates) >= 2
    assert {1, 2, 3, 7}.issubset(lead_days)
    assert focused_mae_regions == {"southeast", "plains", "northeast"}
    assert len(focused_mae_dates) >= 8
    assert {"west_coast", "rockies", "great_lakes"}.issubset(full_seasonal_regions)
    assert ndfd_regions == {"west_coast"}
    assert ndfd_leads == {1}

    override_cases = _resolved_cases(date(2026, 3, 20), VERIFICATION_BENCHMARK_TIER_DEFAULT)
    assert {case.valid_date for case in override_cases} == {date(2026, 3, 20)}
    assert {case.region_name for case in override_cases} == {"southeast", "southwest", "plains", "northeast"}
    assert {case.forecast_lead_days for case in override_cases} == {1, 2, 3, 7}

    focused_mae_override_cases = _resolved_cases(date(2026, 3, 20), VERIFICATION_BENCHMARK_TIER_FOCUSED_MAE)
    assert {case.valid_date for case in focused_mae_override_cases} == {date(2026, 3, 20)}
    assert {case.region_name for case in focused_mae_override_cases} == {"southeast", "plains", "northeast"}
    assert {case.forecast_lead_days for case in focused_mae_override_cases} == {1, 2, 3, 7}

    full_seasonal_override_cases = _resolved_cases(date(2026, 3, 20), VERIFICATION_BENCHMARK_TIER_FULL_SEASONAL)
    assert {case.valid_date for case in full_seasonal_override_cases} == {date(2026, 3, 20)}
    assert {"west_coast", "rockies", "great_lakes"}.issubset({case.region_name for case in full_seasonal_override_cases})

    ndfd_override_cases = _resolved_cases(date(2024, 3, 20), VERIFICATION_BENCHMARK_TIER_NDFD_WEST_COAST)
    assert {case.valid_date for case in ndfd_override_cases} == {date(2024, 3, 20)}
    assert {case.region_name for case in ndfd_override_cases} == {"west_coast"}
    assert {case.forecast_lead_days for case in ndfd_override_cases} == {1}


def test_parse_lead_days_preserves_order_and_deduplicates() -> None:
    assert _parse_lead_days("1,2,3,7") == (1, 2, 3, 7)
    assert _parse_lead_days("1, 3, 3, 7") == (1, 3, 7)


def test_parse_aggregation_policies_preserves_order_and_deduplicates() -> None:
    assert _parse_aggregation_policies("baseline,experimental_regime_aware,baseline") == (
        "baseline",
        "experimental_regime_aware",
    )


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
            "forecast_daily_fields": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_forecast.nc",
            "analysis_daily_fields": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_analysis.nc",
            "summary_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_summary.csv",
            "point_metrics_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_points.csv",
            "component_metrics_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_components.csv",
            "request_summary_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_request_summary.csv",
            "request_detail_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_request_detail.csv",
            "summary_record": {
                "valid_date": valid_date.isoformat(),
                "region_name": region_name,
                "forecast_lead_days": kwargs["forecast_lead_days"],
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
        benchmark_tier=VERIFICATION_BENCHMARK_TIER_DEFAULT,
    )

    assert len(frame) == 1
    assert frame.iloc[0]["benchmark_tier"] == VERIFICATION_BENCHMARK_TIER_DEFAULT
    assert frame.iloc[0]["region"] == "southeast"
    assert frame.iloc[0]["forecast_lead_days"] == 1
    assert frame.iloc[0]["score_mae"] == 5.5
    assert "forecast_score_map_path" in frame.columns
    assert "absolute_error_map_path" in frame.columns
    assert "component_metrics_csv_path" in frame.columns
    assert bool(frame.iloc[0]["passes_benchmark_thresholds"]) is True

    table = format_benchmark_table(frame)
    assert "southeast" in table
    assert "5.5" in table


def test_run_verification_benchmark_collects_multiple_policies(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_verification(**kwargs):
        valid_date = kwargs["valid_date"]
        region_name = kwargs["region_name"]
        aggregation_policy = kwargs["aggregation_policy"]
        score_mae = 5.5 if aggregation_policy == "baseline" else 4.5
        return {
            "forecast_score_map": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_forecast.png",
            "analysis_score_map": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_analysis.png",
            "score_difference_map": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_diff.png",
            "absolute_error_map": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_abs_error.png",
            "category_disagreement_map": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_category_disagreement.png",
            "missed_high_comfort_map": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_missed.png",
            "false_high_comfort_map": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_false.png",
            "forecast_daily_fields": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_forecast.nc",
            "analysis_daily_fields": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_analysis.nc",
            "summary_csv": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_summary.csv",
            "point_metrics_csv": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_points.csv",
            "component_metrics_csv": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_components.csv",
            "request_summary_csv": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_request_summary.csv",
            "request_detail_csv": tmp_path / f"{region_name}_{aggregation_policy}_{valid_date:%Y%m%d}_request_detail.csv",
            "summary_record": {
                "valid_date": valid_date.isoformat(),
                "region_name": region_name,
                "forecast_lead_days": kwargs["forecast_lead_days"],
                "verification_aggregation_policy": aggregation_policy,
                "verification_aggregation_mode": "baseline" if aggregation_policy == "baseline" else "long_lead_soft",
                "score_bias_mean": 1.25,
                "score_mae": score_mae,
                "score_rmse": 7.1,
                "exact_category_agreement_fraction": 0.6,
                "near_category_agreement_fraction": 0.9,
            },
        }

    monkeypatch.setattr("comfortwx.validation.verify_benchmark.run_verification", _fake_run_verification)
    frame = run_verification_benchmark(
        cases=[VerificationBenchmarkCase(region_name="southeast", valid_date=date(2026, 3, 20))],
        output_dir=tmp_path,
        mesh_profile="standard",
        forecast_model="gfs_seamless",
        forecast_run_hour_utc=12,
        benchmark_tier=VERIFICATION_BENCHMARK_TIER_DEFAULT,
        aggregation_policies=("baseline", "experimental_regime_aware"),
    )

    assert len(frame) == 2
    assert set(frame["verification_aggregation_policy"]) == {"baseline", "experimental_regime_aware"}
    policy_summary, policy_summary_path = _write_aggregation_policy_summary(frame, output_dir=tmp_path, stem="test")
    assert not policy_summary.empty
    assert policy_summary_path is not None and policy_summary_path.exists()
    assert "score_mae_improvement_vs_baseline" in policy_summary.columns


def test_run_verification_benchmark_defers_uncached_cases_after_cap(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}

    def _fake_run_verification(**kwargs):
        calls["count"] += 1
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
            "forecast_daily_fields": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_forecast.nc",
            "analysis_daily_fields": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_analysis.nc",
            "summary_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_summary.csv",
            "point_metrics_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_points.csv",
            "component_metrics_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_components.csv",
            "request_summary_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_request_summary.csv",
            "request_detail_csv": tmp_path / f"{region_name}_{valid_date:%Y%m%d}_request_detail.csv",
            "summary_record": {
                "valid_date": valid_date.isoformat(),
                "region_name": region_name,
                "forecast_lead_days": kwargs["forecast_lead_days"],
                "score_bias_mean": 1.25,
                "score_mae": 5.5,
                "score_rmse": 7.1,
                "exact_category_agreement_fraction": 0.6,
                "near_category_agreement_fraction": 0.9,
            },
        }

    monkeypatch.setattr("comfortwx.validation.verify_benchmark.run_verification", _fake_run_verification)
    monkeypatch.setattr("comfortwx.validation.verify_benchmark.time.sleep", lambda _seconds: None)
    frame = run_verification_benchmark(
        cases=[
            VerificationBenchmarkCase(region_name="southeast", valid_date=date(2026, 3, 20)),
            VerificationBenchmarkCase(region_name="southwest", valid_date=date(2026, 3, 20)),
        ],
        output_dir=tmp_path,
        mesh_profile="standard",
        forecast_model="gfs_seamless",
        forecast_run_hour_utc=12,
        benchmark_tier=VERIFICATION_BENCHMARK_TIER_FULL_SEASONAL,
        max_fresh_cases=1,
        case_cooldown_seconds=0.0,
    )

    assert calls["count"] == 1
    assert list(frame["status"]) == ["ok", "deferred"]
    assert frame.iloc[1]["build_source"] == "deferred"


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
                "forecast_lead_days": 1,
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
                "forecast_lead_days": 3,
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
    lead_summary, lead_summary_csv_path = _write_lead_summary(summary, output_dir=tmp_path, stem="test")
    assert not lead_summary.empty
    assert lead_summary_csv_path is not None and lead_summary_csv_path.exists()
    lead_summary_chart = _write_lead_summary_chart(lead_summary, output_dir=tmp_path, stem="test")
    assert lead_summary_chart is not None and lead_summary_chart.exists()
    charts["lead_summary_chart"] = lead_summary_chart
    region_lead_summary, region_lead_summary_csv_path = _write_region_lead_summary(summary, output_dir=tmp_path, stem="test")
    assert not region_lead_summary.empty
    assert region_lead_summary_csv_path is not None and region_lead_summary_csv_path.exists()
    region_lead_heatmap = _write_region_lead_heatmap(region_lead_summary, output_dir=tmp_path, stem="test")
    assert region_lead_heatmap is not None and region_lead_heatmap.exists()
    charts["region_lead_heatmap"] = region_lead_heatmap
    component_priority_summary, component_priority_csv_path = _write_component_priority_summary(summary, output_dir=tmp_path, stem="test")
    assert not component_priority_summary.empty
    assert component_priority_csv_path is not None and component_priority_csv_path.exists()
    component_priority_chart = _write_component_priority_chart(component_priority_summary, output_dir=tmp_path, stem="test")
    assert component_priority_chart is not None and component_priority_chart.exists()
    charts["component_priority_chart"] = component_priority_chart
    priority_cases, priority_cases_csv_path = _write_priority_case_summary(summary, output_dir=tmp_path, stem="test")
    assert not priority_cases.empty
    assert priority_cases_csv_path is not None and priority_cases_csv_path.exists()
    priority_case_chart = _write_priority_case_chart(priority_cases, output_dir=tmp_path, stem="test")
    assert priority_case_chart is not None and priority_case_chart.exists()
    charts["priority_case_chart"] = priority_case_chart
    assert "driven" in _priority_table(priority_cases)

    report_path = _write_benchmark_html_report(
        summary,
        charts=charts,
        benchmark_tier=VERIFICATION_BENCHMARK_TIER_DEFAULT,
        region_summary=region_summary,
        region_summary_csv_path=region_summary_csv_path,
        lead_summary=lead_summary,
        lead_summary_csv_path=lead_summary_csv_path,
        region_lead_summary=region_lead_summary,
        region_lead_summary_csv_path=region_lead_summary_csv_path,
        component_priority_summary=component_priority_summary,
        component_priority_csv_path=component_priority_csv_path,
        priority_cases=priority_cases,
        priority_cases_csv_path=priority_cases_csv_path,
        calibration_summary=pd.DataFrame(),
        calibration_summary_csv_path=None,
        output_dir=tmp_path,
        stem="test",
    )
    assert report_path.exists()
    html_text = report_path.read_text(encoding="utf-8")
    assert "Comfort Index Verification Benchmark" in html_text
    assert "Tier: Default" in html_text
    assert "diff.png" in html_text
    assert "Regional Rollup" in html_text
    assert "Forecast Lead Rollup" in html_text
    assert "Improvement Priorities" in html_text
    assert "Component Priorities" in html_text

    summary_csv_path = tmp_path / "comfortwx_verify_benchmark_test.csv"
    summary.to_csv(summary_csv_path, index=False)

    site_index = _write_verification_site(
        summary=summary,
        summary_path=summary_csv_path,
        charts=charts,
        report_path=report_path,
        region_summary_csv_path=region_summary_csv_path,
        lead_summary_csv_path=lead_summary_csv_path,
        region_lead_summary_csv_path=region_lead_summary_csv_path,
        component_priority_csv_path=component_priority_csv_path,
        priority_cases_csv_path=priority_cases_csv_path,
        calibration_summary_csv_path=None,
        output_dir=tmp_path,
        stem="test",
    )
    assert site_index.exists()
    assert (tmp_path / "verification_site" / "latest" / "index.html").exists()


def test_write_verification_site_skips_blank_or_directory_like_paths(tmp_path: Path) -> None:
    report_path = tmp_path / "report.html"
    report_path.write_text("<html></html>", encoding="utf-8")
    summary_csv_path = tmp_path / "summary.csv"
    summary_csv_path.write_text("region,date\n", encoding="utf-8")
    chart_path = tmp_path / "chart.png"
    chart_path.write_bytes(b"fake")
    summary = pd.DataFrame(
        [
            {
                "region": "southeast",
                "date": "2026-03-20",
                "status": "error",
                "forecast_score_map_path": "",
                "analysis_score_map_path": ".",
                "score_difference_map_path": str(tmp_path),
            }
        ]
    )

    site_index = _write_verification_site(
        summary=summary,
        summary_path=summary_csv_path,
        charts={"chart": chart_path},
        report_path=report_path,
        region_summary_csv_path=None,
        lead_summary_csv_path=None,
        region_lead_summary_csv_path=None,
        component_priority_csv_path=None,
        priority_cases_csv_path=None,
        calibration_summary_csv_path=None,
        output_dir=tmp_path,
        stem="blank_path_guard",
    )

    assert site_index.exists()
    site_dir = tmp_path / "verification_site" / "blank_path_guard"
    assert (site_dir / "report.html").exists()
    assert (site_dir / "chart.png").exists()


def test_build_calibration_summary_improves_synthetic_bias_case(tmp_path: Path) -> None:
    lat = np.linspace(30.0, 34.5, 10)
    lon = np.linspace(-94.5, -90.0, 10)
    base_scores = np.linspace(60.0, 90.0, 100, dtype=np.float32).reshape(10, 10)
    forecast_a = xr.Dataset(
        data_vars={
            "daily_score": (("lat", "lon"), base_scores),
            "category_index": (("lat", "lon"), np.full((10, 10), 3, dtype=int)),
            "pristine_allowed": (("lat", "lon"), np.ones((10, 10), dtype=bool)),
        },
        coords={"lat": lat, "lon": lon},
    )
    analysis_a = xr.Dataset(
        data_vars={
            "daily_score": (("lat", "lon"), base_scores - 5.0),
            "category_index": (("lat", "lon"), np.full((10, 10), 3, dtype=int)),
            "pristine_allowed": (("lat", "lon"), np.ones((10, 10), dtype=bool)),
        },
        coords={"lat": lat, "lon": lon},
    )
    forecast_b = xr.Dataset(
        data_vars={
            "daily_score": (("lat", "lon"), base_scores + 2.0),
            "category_index": (("lat", "lon"), np.full((10, 10), 3, dtype=int)),
            "pristine_allowed": (("lat", "lon"), np.ones((10, 10), dtype=bool)),
        },
        coords={"lat": lat, "lon": lon},
    )
    analysis_b = xr.Dataset(
        data_vars={
            "daily_score": (("lat", "lon"), base_scores - 3.0),
            "category_index": (("lat", "lon"), np.full((10, 10), 3, dtype=int)),
            "pristine_allowed": (("lat", "lon"), np.ones((10, 10), dtype=bool)),
        },
        coords={"lat": lat, "lon": lon},
    )
    forecast_path_a = tmp_path / "forecast_a.nc"
    analysis_path_a = tmp_path / "analysis_a.nc"
    forecast_path_b = tmp_path / "forecast_b.nc"
    analysis_path_b = tmp_path / "analysis_b.nc"
    forecast_a.to_netcdf(forecast_path_a)
    analysis_a.to_netcdf(analysis_path_a)
    forecast_b.to_netcdf(forecast_path_b)
    analysis_b.to_netcdf(analysis_path_b)

    summary = pd.DataFrame(
        [
            {
                "region": "southeast",
                "date": "2026-01-15",
                "forecast_lead_days": 1,
                "status": "ok",
                "score_bias_mean": 5.0,
                "score_mae": 5.0,
                "score_rmse": 5.0,
                "exact_category_agreement_fraction": 0.5,
                "near_category_agreement_fraction": 1.0,
                "benchmark_threshold_status": "pass",
                "passes_benchmark_thresholds": True,
                "improvement_priority_score": 0.0,
                "forecast_daily_fields_path": str(forecast_path_a),
                "analysis_daily_fields_path": str(analysis_path_a),
            },
            {
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "status": "ok",
                "score_bias_mean": 5.0,
                "score_mae": 5.0,
                "score_rmse": 5.0,
                "exact_category_agreement_fraction": 0.5,
                "near_category_agreement_fraction": 1.0,
                "benchmark_threshold_status": "pass",
                "passes_benchmark_thresholds": True,
                "improvement_priority_score": 0.0,
                "forecast_daily_fields_path": str(forecast_path_b),
                "analysis_daily_fields_path": str(analysis_path_b),
            },
        ]
    )

    calibration_summary, calibration_csv = _build_calibration_summary(summary, output_dir=tmp_path, stem="synthetic")

    assert not calibration_summary.empty
    assert calibration_csv is not None and calibration_csv.exists()
    assert (calibration_summary["score_mae_improvement"] > 0).all()


def test_apply_score_calibration_recomputes_categories() -> None:
    forecast_daily = xr.Dataset(
        data_vars={
            "daily_score": (("lat", "lon"), np.array([[44.0, 74.0], [89.0, 95.0]], dtype=np.float32)),
            "category_index": (("lat", "lon"), np.array([[0, 2], [3, 4]], dtype=int)),
            "pristine_allowed": (("lat", "lon"), np.ones((2, 2), dtype=bool)),
        },
        coords={"lat": [30.0, 31.0], "lon": [-90.0, -89.0]},
    )

    calibrated = _apply_score_calibration(forecast_daily, slope=1.0, intercept=-5.0)

    assert float(calibrated["daily_score"].isel(lat=0, lon=0).values) == 39.0
    assert int(calibrated["category_index"].isel(lat=0, lon=1).values) == 2
    assert int(calibrated["category_index"].isel(lat=1, lon=1).values) == 4
