from __future__ import annotations

import pandas as pd
import xarray as xr

from nicewx.config import get_openmeteo_mesh_settings
from nicewx.validation.western_mesh_sensitivity import compare_mosaic_mesh_fields
from nicewx.validation.western_mosaic_method_sensitivity import compare_mosaic_methods
from nicewx.validation.western_fair_nice_audit import borderline_fair_nice_agreement, category_index_with_fair_nice_threshold
from nicewx.validation.western_seam_attribution import summarize_overlap_attribution
from nicewx.validation.western_threshold_sensitivity import crossed_thresholds, margin_stable_category_agreement


def test_get_openmeteo_mesh_settings_returns_fine_profile_for_western_regions() -> None:
    southwest_standard = get_openmeteo_mesh_settings("southwest", "standard")
    southwest_fine = get_openmeteo_mesh_settings("southwest", "fine")

    assert float(southwest_fine["lat_step"]) < float(southwest_standard["lat_step"])
    assert float(southwest_fine["lon_step"]) < float(southwest_standard["lon_step"])


def test_compare_mosaic_mesh_fields_reports_score_changes_on_fine_grid() -> None:
    standard_daily = xr.Dataset(
        {"daily_score": (("lat", "lon"), [[60.0, 70.0], [80.0, 90.0]])},
        coords={"lat": [30.0, 32.0], "lon": [-112.0, -110.0]},
    )
    fine_daily = xr.Dataset(
        {"daily_score": (("lat", "lon"), [[61.0, 66.0, 71.0], [72.0, 77.0, 82.0], [83.0, 88.0, 93.0]])},
        coords={"lat": [30.0, 31.0, 32.0], "lon": [-112.0, -111.0, -110.0]},
    )

    comparison = compare_mosaic_mesh_fields(standard_daily=standard_daily, fine_daily=fine_daily)

    assert comparison["compared_cell_count"] == 9
    assert comparison["mean_abs_daily_score_change"] > 0.0
    assert comparison["max_abs_daily_score_change"] >= comparison["mean_abs_daily_score_change"]


def test_compare_mosaic_methods_reports_overlap_category_flips() -> None:
    baseline_daily = xr.Dataset(
        {
            "daily_score": (("lat", "lon"), [[60.0, 70.0], [80.0, 90.0]]),
            "category_index": (("lat", "lon"), [[2.0, 2.0], [3.0, 4.0]]),
            "overlap_mask": (("lat", "lon"), [[1, 1], [1, 1]]),
        },
        coords={"lat": [30.0, 31.0], "lon": [-112.0, -111.0]},
    )
    candidate_daily = xr.Dataset(
        {
            "daily_score": (("lat", "lon"), [[50.0, 70.0], [78.0, 88.0]]),
            "category_index": (("lat", "lon"), [[1.0, 2.0], [3.0, 3.0]]),
            "overlap_mask": (("lat", "lon"), [[1, 1], [1, 1]]),
        },
        coords={"lat": [30.0, 31.0], "lon": [-112.0, -111.0]},
    )

    comparison = compare_mosaic_methods(baseline_daily=baseline_daily, candidate_daily=candidate_daily)

    assert comparison["compared_cell_count"] == 4
    assert comparison["mean_abs_daily_score_change"] > 0.0
    assert comparison["overlap_category_flip_count"] == 2


def test_summarize_overlap_attribution_identifies_dominant_driver() -> None:
    frame = pd.DataFrame(
        {
            "dominant_component": ["cloud_score", "cloud_score", "precip_score"],
            "southwest_temp_score": [30.0, 31.0, 32.0],
            "rockies_temp_score": [28.0, 29.0, 31.0],
            "temp_score_abs_diff": [2.0, 2.0, 1.0],
            "southwest_dewpoint_score": [18.0, 18.0, 17.0],
            "rockies_dewpoint_score": [16.0, 16.0, 16.0],
            "dewpoint_score_abs_diff": [2.0, 2.0, 1.0],
            "southwest_wind_score": [8.0, 8.0, 7.0],
            "rockies_wind_score": [6.0, 6.0, 6.0],
            "wind_score_abs_diff": [2.0, 2.0, 1.0],
            "southwest_cloud_score": [7.0, 8.0, 9.0],
            "rockies_cloud_score": [3.0, 4.0, 5.0],
            "cloud_score_abs_diff": [4.0, 4.0, 4.0],
            "southwest_precip_score": [12.0, 12.0, 12.0],
            "rockies_precip_score": [10.0, 10.0, 6.0],
            "precip_score_abs_diff": [2.0, 2.0, 6.0],
            "southwest_hazard_penalty": [1.0, 1.0, 1.0],
            "rockies_hazard_penalty": [2.0, 2.0, 3.0],
            "hazard_penalty_abs_diff": [1.0, 1.0, 2.0],
            "southwest_interaction_adjustment": [1.0, 1.0, 1.0],
            "rockies_interaction_adjustment": [0.0, 0.0, 0.0],
            "interaction_adjustment_abs_diff": [1.0, 1.0, 1.0],
            "southwest_reliability_score": [80.0, 81.0, 82.0],
            "rockies_reliability_score": [78.0, 79.0, 80.0],
            "reliability_score_abs_diff": [2.0, 2.0, 2.0],
            "southwest_disruption_penalty": [4.0, 4.0, 5.0],
            "rockies_disruption_penalty": [5.0, 5.0, 7.0],
            "disruption_penalty_abs_diff": [1.0, 1.0, 2.0],
            "southwest_daily_score": [70.0, 72.0, 73.0],
            "rockies_daily_score": [66.0, 68.0, 69.0],
            "daily_score_abs_diff": [4.0, 4.0, 4.0],
            "southwest_category_index": [2.0, 2.0, 2.0],
            "rockies_category_index": [2.0, 2.0, 1.0],
            "category_index_abs_diff": [0.0, 0.0, 1.0],
        }
    )

    summary = summarize_overlap_attribution(frame)

    assert summary["dominant_driver"] == "cloud_score"
    assert summary["secondary_driver"] == "precip_score"
    assert summary["driver_group"] == "cloud/precip"


def test_threshold_helpers_identify_crossings_and_margin_stability() -> None:
    assert crossed_thresholds(58.5, 60.8) == [60.0]
    assert margin_stable_category_agreement(58.9, 60.7, 1, 2, margin=2.0) is True
    assert margin_stable_category_agreement(56.0, 63.0, 1, 2, margin=2.0) is False


def test_fair_nice_helpers_support_comparison_only_low_end_diagnostics() -> None:
    assert category_index_with_fair_nice_threshold(44.4, 45.0) == 0
    assert category_index_with_fair_nice_threshold(44.4, 44.0) == 1
    assert borderline_fair_nice_agreement(44.1, 46.2, 0, 1, threshold=45.0, margin=2.0) is True
    assert borderline_fair_nice_agreement(41.0, 48.5, 0, 1, threshold=45.0, margin=2.0) is False
