from __future__ import annotations

from datetime import date
import pandas as pd
from pathlib import Path

from comfortwx.validation.tune_daily_aggregation import (
    _hourly_cache_paths,
    _parse_candidate_modes,
    _parse_lead_days,
    evaluate_daily_aggregation_modes,
    build_policy_comparison,
    build_holdout_mode_selection,
    recommend_modes_by_lead,
    summarize_candidate_modes,
)
from comfortwx.validation.verify_benchmark_cases import VerificationBenchmarkCase


def test_parse_candidate_modes_preserves_order_and_deduplicates() -> None:
    assert _parse_candidate_modes("baseline, soft_reliability, baseline, long_lead_soft") == (
        "baseline",
        "soft_reliability",
        "long_lead_soft",
    )


def test_parse_lead_days_preserves_order_and_deduplicates() -> None:
    assert _parse_lead_days("1,2,2,7") == (1, 2, 7)


def test_hourly_cache_paths_include_cache_version(tmp_path: Path) -> None:
    forecast_path, analysis_path = _hourly_cache_paths(
        cache_dir=tmp_path,
        region_name="southeast",
        valid_date=date(2026, 3, 20),
        forecast_model="gfs_seamless",
        forecast_lead_days=1,
    )

    assert "v3_noaa_truth" in forecast_path.name
    assert "v3_noaa_truth" in analysis_path.name
    assert "noaa_urma_rtma" in forecast_path.name


def test_summarize_candidate_modes_and_recommendations() -> None:
    frame = pd.DataFrame(
        [
            {
                "case_label": "southeast 2026-03-20 D+1",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 1.0,
                "score_mae": 8.0,
                "score_rmse": 10.0,
                "exact_category_agreement_fraction": 0.60,
                "near_category_agreement_fraction": 0.88,
            },
            {
                "case_label": "southeast 2026-03-20 D+1",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "soft_reliability",
                "status": "ok",
                "score_bias_mean": 0.8,
                "score_mae": 7.0,
                "score_rmse": 9.0,
                "exact_category_agreement_fraction": 0.64,
                "near_category_agreement_fraction": 0.90,
            },
            {
                "case_label": "northeast 2026-03-20 D+1",
                "region": "northeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 1.2,
                "score_mae": 9.0,
                "score_rmse": 11.0,
                "exact_category_agreement_fraction": 0.55,
                "near_category_agreement_fraction": 0.86,
            },
            {
                "case_label": "northeast 2026-03-20 D+1",
                "region": "northeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "soft_reliability",
                "status": "ok",
                "score_bias_mean": 0.9,
                "score_mae": 7.8,
                "score_rmse": 9.8,
                "exact_category_agreement_fraction": 0.60,
                "near_category_agreement_fraction": 0.89,
            },
        ]
    )

    candidate_summary = summarize_candidate_modes(frame)
    assert not candidate_summary.empty
    recommended = recommend_modes_by_lead(candidate_summary)
    assert not recommended.empty
    assert recommended.iloc[0]["recommended_aggregation_mode"] == "soft_reliability"


def test_build_holdout_mode_selection_prefers_better_training_mode() -> None:
    frame = pd.DataFrame(
        [
            {
                "case_label": "southeast 2026-01-15 D+1",
                "region": "southeast",
                "date": "2026-01-15",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 1.0,
                "score_mae": 8.0,
                "score_rmse": 10.0,
                "exact_category_agreement_fraction": 0.60,
                "near_category_agreement_fraction": 0.88,
            },
            {
                "case_label": "southeast 2026-01-15 D+1",
                "region": "southeast",
                "date": "2026-01-15",
                "forecast_lead_days": 1,
                "aggregation_mode": "soft_reliability",
                "status": "ok",
                "score_bias_mean": 0.7,
                "score_mae": 7.0,
                "score_rmse": 9.0,
                "exact_category_agreement_fraction": 0.64,
                "near_category_agreement_fraction": 0.90,
            },
            {
                "case_label": "southeast 2026-03-20 D+1",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 1.1,
                "score_mae": 8.2,
                "score_rmse": 10.2,
                "exact_category_agreement_fraction": 0.59,
                "near_category_agreement_fraction": 0.87,
            },
            {
                "case_label": "southeast 2026-03-20 D+1",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "soft_reliability",
                "status": "ok",
                "score_bias_mean": 0.8,
                "score_mae": 7.1,
                "score_rmse": 9.1,
                "exact_category_agreement_fraction": 0.65,
                "near_category_agreement_fraction": 0.91,
            },
            {
                "case_label": "northeast 2026-01-15 D+1",
                "region": "northeast",
                "date": "2026-01-15",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 1.4,
                "score_mae": 9.3,
                "score_rmse": 11.4,
                "exact_category_agreement_fraction": 0.54,
                "near_category_agreement_fraction": 0.84,
            },
            {
                "case_label": "northeast 2026-01-15 D+1",
                "region": "northeast",
                "date": "2026-01-15",
                "forecast_lead_days": 1,
                "aggregation_mode": "soft_reliability",
                "status": "ok",
                "score_bias_mean": 1.0,
                "score_mae": 8.0,
                "score_rmse": 10.0,
                "exact_category_agreement_fraction": 0.60,
                "near_category_agreement_fraction": 0.88,
            },
        ]
    )

    holdout = build_holdout_mode_selection(frame)
    assert not holdout.empty
    assert set(holdout["selected_aggregation_mode"]) == {"soft_reliability"}
    assert (holdout["score_mae_improvement"] > 0).all()


def test_build_policy_comparison_compares_against_baseline() -> None:
    frame = pd.DataFrame(
        [
            {
                "case_label": "southeast 2026-03-20 D+1",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 1.2,
                "score_mae": 8.0,
                "score_rmse": 10.0,
                "exact_category_agreement_fraction": 0.60,
                "near_category_agreement_fraction": 0.88,
            },
            {
                "case_label": "southeast 2026-03-20 D+1",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "long_lead_soft",
                "status": "ok",
                "score_bias_mean": 1.0,
                "score_mae": 6.0,
                "score_rmse": 8.0,
                "exact_category_agreement_fraction": 0.64,
                "near_category_agreement_fraction": 0.90,
            },
            {
                "case_label": "southeast 2026-03-20 D+2",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 2,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 1.1,
                "score_mae": 9.0,
                "score_rmse": 11.0,
                "exact_category_agreement_fraction": 0.58,
                "near_category_agreement_fraction": 0.85,
            },
            {
                "case_label": "southeast 2026-03-20 D+2",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 2,
                "aggregation_mode": "long_lead_soft",
                "status": "ok",
                "score_bias_mean": 0.8,
                "score_mae": 7.0,
                "score_rmse": 9.0,
                "exact_category_agreement_fraction": 0.63,
                "near_category_agreement_fraction": 0.91,
            },
        ]
    )

    policy_case_scores, policy_summary = build_policy_comparison(frame)

    assert not policy_case_scores.empty
    assert not policy_summary.empty
    lead_one_soft = policy_summary.loc[
        (policy_summary["forecast_lead_days"] == 1)
        & (policy_summary["policy_name"] == "experimental_all_leads_soft")
    ].iloc[0]
    lead_two_lead_aware = policy_summary.loc[
        (policy_summary["forecast_lead_days"] == 2)
        & (policy_summary["policy_name"] == "experimental_lead_aware_soft")
    ].iloc[0]
    assert lead_one_soft["score_mae_improvement_vs_baseline"] == 2.0
    assert lead_two_lead_aware["score_mae_improvement_vs_baseline"] == 2.0


def test_build_policy_comparison_supports_region_overrides() -> None:
    frame = pd.DataFrame(
        [
            {
                "case_label": "southeast 2026-03-20 D+1",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 1.2,
                "score_mae": 8.0,
                "score_rmse": 10.0,
                "exact_category_agreement_fraction": 0.60,
                "near_category_agreement_fraction": 0.88,
            },
            {
                "case_label": "southeast 2026-03-20 D+1",
                "region": "southeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "long_lead_soft",
                "status": "ok",
                "score_bias_mean": 1.0,
                "score_mae": 6.0,
                "score_rmse": 8.0,
                "exact_category_agreement_fraction": 0.64,
                "near_category_agreement_fraction": 0.90,
            },
            {
                "case_label": "northeast 2026-03-20 D+1",
                "region": "northeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 0.5,
                "score_mae": 4.0,
                "score_rmse": 5.0,
                "exact_category_agreement_fraction": 0.88,
                "near_category_agreement_fraction": 1.00,
            },
            {
                "case_label": "northeast 2026-03-20 D+1",
                "region": "northeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "long_lead_soft",
                "status": "ok",
                "score_bias_mean": 0.7,
                "score_mae": 4.5,
                "score_rmse": 5.4,
                "exact_category_agreement_fraction": 0.86,
                "near_category_agreement_fraction": 0.98,
            },
        ]
    )

    _, policy_summary = build_policy_comparison(frame)

    southeast_blend = policy_summary.loc[
        (policy_summary["forecast_lead_days"] == 1)
        & (policy_summary["policy_name"] == "experimental_regional_blend")
    ].iloc[0]

    assert southeast_blend["mean_score_mae"] == 5.0
    assert southeast_blend["score_mae_improvement_vs_baseline"] == 1.0


def test_build_policy_comparison_supports_calendar_regimes() -> None:
    frame = pd.DataFrame(
        [
            {
                "case_label": "northeast 2025-01-15 D+1",
                "region": "northeast",
                "date": "2025-01-15",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 0.5,
                "score_mae": 5.0,
                "score_rmse": 6.0,
                "exact_category_agreement_fraction": 0.82,
                "near_category_agreement_fraction": 0.94,
            },
            {
                "case_label": "northeast 2025-01-15 D+1",
                "region": "northeast",
                "date": "2025-01-15",
                "forecast_lead_days": 1,
                "aggregation_mode": "long_lead_soft",
                "status": "ok",
                "score_bias_mean": 0.3,
                "score_mae": 4.0,
                "score_rmse": 5.0,
                "exact_category_agreement_fraction": 0.84,
                "near_category_agreement_fraction": 0.96,
            },
            {
                "case_label": "northeast 2026-03-20 D+1",
                "region": "northeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "baseline",
                "status": "ok",
                "score_bias_mean": 0.7,
                "score_mae": 6.0,
                "score_rmse": 7.0,
                "exact_category_agreement_fraction": 0.80,
                "near_category_agreement_fraction": 0.92,
            },
            {
                "case_label": "northeast 2026-03-20 D+1",
                "region": "northeast",
                "date": "2026-03-20",
                "forecast_lead_days": 1,
                "aggregation_mode": "long_lead_soft",
                "status": "ok",
                "score_bias_mean": 0.4,
                "score_mae": 3.5,
                "score_rmse": 4.4,
                "exact_category_agreement_fraction": 0.86,
                "near_category_agreement_fraction": 0.97,
            },
        ]
    )

    _, policy_summary = build_policy_comparison(
        frame,
        policy_definitions={
            "baseline": {
                "default": {1: "baseline"},
            },
            "seasonal_policy": {
                "default": {1: "baseline"},
                "calendar_regimes": {
                    "cool_season": {
                        "default": {1: "baseline"},
                    },
                    "warm_season": {
                        "default": {1: "long_lead_soft"},
                    },
                },
            },
        },
    )

    seasonal_row = policy_summary.loc[policy_summary["policy_name"] == "seasonal_policy"].iloc[0]
    assert seasonal_row["mean_score_mae"] == 4.25
    assert seasonal_row["score_mae_improvement_vs_baseline"] == 1.25


def test_evaluate_daily_aggregation_modes_defers_after_fresh_case_cap(monkeypatch, tmp_path: Path) -> None:
    def _fake_load_or_build_hourly_case(**_kwargs):
        return object(), object(), "fresh"

    def _fake_aggregate_daily_scores(_dataset, aggregation_mode="baseline"):
        return {"aggregation_mode": aggregation_mode}

    def _fake_daily_score_metrics(forecast_daily, _analysis_daily):
        return {
            "score_bias_mean": 0.0,
            "score_mae": 5.0 if forecast_daily["aggregation_mode"] == "baseline" else 4.0,
            "score_rmse": 6.0,
            "exact_category_agreement_fraction": 0.6,
            "near_category_agreement_fraction": 0.9,
        }

    monkeypatch.setattr(
        "comfortwx.validation.tune_daily_aggregation._load_or_build_hourly_case",
        _fake_load_or_build_hourly_case,
    )
    monkeypatch.setattr(
        "comfortwx.validation.tune_daily_aggregation.aggregate_daily_scores",
        _fake_aggregate_daily_scores,
    )
    monkeypatch.setattr(
        "comfortwx.validation.tune_daily_aggregation._daily_score_metrics",
        _fake_daily_score_metrics,
    )
    monkeypatch.setattr("comfortwx.validation.tune_daily_aggregation.time.sleep", lambda _seconds: None)

    frame = evaluate_daily_aggregation_modes(
        cases=[
            VerificationBenchmarkCase(region_name="southeast", valid_date=date(2026, 3, 20), forecast_lead_days=1),
            VerificationBenchmarkCase(region_name="southwest", valid_date=date(2026, 3, 20), forecast_lead_days=1),
        ],
        output_dir=tmp_path,
        mesh_profile="standard",
        forecast_model="gfs_seamless",
        forecast_run_hour_utc=12,
        candidate_modes=("baseline",),
        case_cache_mode="reuse",
        benchmark_tier="full-seasonal",
        max_fresh_cases=1,
        case_cooldown_seconds=0.0,
    )

    assert list(frame["status"]) == ["ok", "deferred"]
