from __future__ import annotations

import pandas as pd

from comfortwx.validation.tune_daily_aggregation import (
    _parse_candidate_modes,
    _parse_lead_days,
    build_holdout_mode_selection,
    recommend_modes_by_lead,
    summarize_candidate_modes,
)


def test_parse_candidate_modes_preserves_order_and_deduplicates() -> None:
    assert _parse_candidate_modes("baseline, soft_reliability, baseline, long_lead_soft") == (
        "baseline",
        "soft_reliability",
        "long_lead_soft",
    )


def test_parse_lead_days_preserves_order_and_deduplicates() -> None:
    assert _parse_lead_days("1,2,2,7") == (1, 2, 7)


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
