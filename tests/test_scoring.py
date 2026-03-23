from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr

from nicewx.data.mock_data import generate_mock_hourly_grid
from nicewx.scoring.categories import category_name_from_value, categorize_scores
from nicewx.scoring.daily import aggregate_daily_scores
from nicewx.scoring.hourly import score_hourly_dataset


def test_category_thresholds() -> None:
    assert category_name_from_value(30) == "Poor"
    assert category_name_from_value(50) == "Fair"
    assert category_name_from_value(70) == "Pleasant"
    assert category_name_from_value(80) == "Ideal"
    assert category_name_from_value(95) == "Exceptional"


def test_mock_pipeline_ranges() -> None:
    hourly = generate_mock_hourly_grid(date(2026, 5, 18), lat_points=10, lon_points=12)
    scored = score_hourly_dataset(hourly)
    daily = aggregate_daily_scores(scored)

    assert set(["hourly_score", "temp_score", "hazard_penalty"]).issubset(scored.data_vars)
    assert set(["daily_score", "category_index"]).issubset(daily.data_vars)
    assert float(scored["hourly_score"].min()) >= 0.0
    assert float(scored["hourly_score"].max()) <= 100.0
    assert float(daily["daily_score"].min()) >= 0.0
    assert float(daily["daily_score"].max()) <= 100.0


def test_thunder_cap_limits_hourly_score() -> None:
    dataset = xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), np.array([[[90.0]]])),
            "dewpoint_f": (("time", "lat", "lon"), np.array([[[72.0]]])),
            "wind_mph": (("time", "lat", "lon"), np.array([[[10.0]]])),
            "gust_mph": (("time", "lat", "lon"), np.array([[[30.0]]])),
            "cloud_pct": (("time", "lat", "lon"), np.array([[[35.0]]])),
            "pop_pct": (("time", "lat", "lon"), np.array([[[80.0]]])),
            "qpf_in": (("time", "lat", "lon"), np.array([[[0.15]]])),
            "thunder": (("time", "lat", "lon"), np.array([[[True]]])),
        },
        coords={"time": xr.date_range("2026-06-01", periods=1, freq="1h"), "lat": [35.0], "lon": [-97.0]},
    )
    scored = score_hourly_dataset(dataset)
    assert float(scored["hourly_score"].values.squeeze()) <= 60.0


def test_pristine_gate_can_downgrade_to_perfect() -> None:
    score = xr.DataArray(np.array([[92.0]]), dims=("lat", "lon"), coords={"lat": [35.0], "lon": [-80.0]})
    pristine_allowed = xr.DataArray(
        np.array([[False]]),
        dims=("lat", "lon"),
        coords={"lat": [35.0], "lon": [-80.0]},
    )
    category_index = categorize_scores(score, pristine_allowed=pristine_allowed)
    assert int(category_index.values.squeeze()) == 3
    assert category_name_from_value(92.0, pristine_allowed=False) == "Ideal"


def test_optional_air_quality_fields_suppress_hourly_score() -> None:
    base_dataset = xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), np.array([[[74.0]]])),
            "dewpoint_f": (("time", "lat", "lon"), np.array([[[52.0]]])),
            "wind_mph": (("time", "lat", "lon"), np.array([[[6.0]]])),
            "gust_mph": (("time", "lat", "lon"), np.array([[[9.0]]])),
            "cloud_pct": (("time", "lat", "lon"), np.array([[[28.0]]])),
            "pop_pct": (("time", "lat", "lon"), np.array([[[5.0]]])),
            "qpf_in": (("time", "lat", "lon"), np.array([[[0.0]]])),
            "thunder": (("time", "lat", "lon"), np.array([[[False]]])),
        },
        coords={"time": xr.date_range("2026-06-01", periods=1, freq="1h"), "lat": [35.0], "lon": [-97.0]},
    )
    polluted_dataset = base_dataset.assign(
        aqi=(("time", "lat", "lon"), np.array([[[160.0]]])),
        visibility_mi=(("time", "lat", "lon"), np.array([[[2.0]]])),
    )

    clean_score = float(score_hourly_dataset(base_dataset)["hourly_score"].values.squeeze())
    polluted_scored = score_hourly_dataset(polluted_dataset)
    polluted_score = float(polluted_scored["hourly_score"].values.squeeze())

    assert polluted_score < clean_score
    assert float(polluted_scored["hourly_score_cap"].values.squeeze()) <= 65.0


def test_soft_reliability_aggregation_is_less_brittle_near_threshold_crossings() -> None:
    times = xr.date_range("2026-06-01T08:00", periods=13, freq="1h")
    lower_scores = np.full(13, 64.0)
    higher_scores = np.full(13, 65.0)

    def _scored_from_hourly(hourly_score_values: np.ndarray) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "hourly_score": (("time", "lat", "lon"), hourly_score_values[:, None, None]),
                "dewpoint_f": (("time", "lat", "lon"), np.full((13, 1, 1), 45.0)),
                "gust_mph": (("time", "lat", "lon"), np.full((13, 1, 1), 20.0)),
                "qpf_in": (("time", "lat", "lon"), np.zeros((13, 1, 1))),
                "pop_pct": (("time", "lat", "lon"), np.full((13, 1, 1), 35.0)),
                "thunder": (("time", "lat", "lon"), np.zeros((13, 1, 1), dtype=bool)),
            },
            coords={"time": times, "lat": [35.0], "lon": [-108.0]},
        )

    lower_scored = _scored_from_hourly(lower_scores)
    higher_scored = _scored_from_hourly(higher_scores)

    baseline_lower = aggregate_daily_scores(lower_scored, aggregation_mode="baseline")
    baseline_higher = aggregate_daily_scores(higher_scored, aggregation_mode="baseline")
    tuned_lower = aggregate_daily_scores(lower_scored, aggregation_mode="soft_reliability")
    tuned_higher = aggregate_daily_scores(higher_scored, aggregation_mode="soft_reliability")

    baseline_delta = float(baseline_higher["daily_score"].values.squeeze()) - float(baseline_lower["daily_score"].values.squeeze())
    tuned_delta = float(tuned_higher["daily_score"].values.squeeze()) - float(tuned_lower["daily_score"].values.squeeze())

    assert tuned_delta < baseline_delta
    assert "reliability_score" in tuned_lower
    assert "disruption_penalty" in tuned_lower
