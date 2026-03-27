from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr
from urllib.error import HTTPError

import comfortwx.data.openmeteo_verification as openmeteo_verification
from comfortwx.data.openmeteo_verification import (
    _fetch_forecast_payloads_for_batch,
    _subset_to_valid_local_day,
    _forecast_query_for_batch,
    _normalize_openmeteo_verification_payload,
    _normalize_previous_run_payload,
    resolve_openmeteo_verification_forecast_model,
)
from comfortwx.validation.verify_model import _verification_summary, build_verification_file_prefix


def test_normalize_openmeteo_verification_payload_derives_pop_proxy_and_visibility_units() -> None:
    payload = {
        "latitude": 35.0,
        "longitude": -78.0,
        "timezone": "America/New_York",
        "hourly_units": {
            "visibility": "ft",
        },
        "hourly": {
            "time": ["2026-03-20T00:00", "2026-03-20T01:00"],
            "temperature_2m": [60.0, 61.0],
            "dew_point_2m": [50.0, 51.0],
            "wind_speed_10m": [8.0, 9.0],
            "wind_gusts_10m": [12.0, 13.0],
            "cloud_cover": [20.0, 40.0],
            "precipitation_probability": [None, None],
            "precipitation": [0.0, 0.05],
            "weather_code": [1, 95],
            "visibility": [52800.0, 26400.0],
        },
    }

    dataset = _normalize_openmeteo_verification_payload(
        payload,
        requested_lat=35.0,
        requested_lon=-78.0,
        source_label="analysis",
        derive_pop_proxy=True,
    )

    assert float(dataset["pop_pct"].isel(time=0).values.squeeze()) == 0.0
    assert float(dataset["pop_pct"].isel(time=1).values.squeeze()) == 100.0
    assert float(dataset["visibility_mi"].isel(time=0).values.squeeze()) == 10.0
    assert bool(dataset["thunder"].isel(time=1).values.squeeze()) is True


def test_verification_summary_reports_score_and_category_agreement() -> None:
    forecast_daily = xr.Dataset(
        data_vars={
            "daily_score": (("lat", "lon"), np.array([[55.0, 80.0], [40.0, 92.0]], dtype=np.float32)),
            "category_index": (("lat", "lon"), np.array([[1, 3], [0, 4]], dtype=int)),
        },
        coords={"lat": [30.0, 31.0], "lon": [-90.0, -89.0]},
    )
    analysis_daily = xr.Dataset(
        data_vars={
            "daily_score": (("lat", "lon"), np.array([[50.0, 77.0], [44.0, 88.0]], dtype=np.float32)),
            "category_index": (("lat", "lon"), np.array([[1, 3], [0, 3]], dtype=int)),
        },
        coords={"lat": [30.0, 31.0], "lon": [-90.0, -89.0]},
    )

    summary = _verification_summary(
        forecast_daily=forecast_daily,
        analysis_daily=analysis_daily,
        metadata={
            "region_name": "southeast",
            "forecast_model": "gfs_seamless",
            "analysis_model": "best_match",
            "forecast_run_timestamp_utc": "2026-03-19T12:00",
            "forecast_lead_days": 2,
            "mesh_profile": "standard",
            "mesh_point_count": 4,
            "aggregation_policy": "experimental_regime_aware",
            "aggregation_mode": "long_lead_soft",
        },
        valid_date=date(2026, 3, 20),
    )

    assert summary["score_mae"] == 4.0
    assert summary["score_max_abs_diff"] == 5.0
    assert summary["exact_category_agreement_fraction"] == 0.75
    assert summary["near_category_agreement_fraction"] == 1.0
    assert summary["category_disagreement_fraction"] == 0.25
    assert summary["forecast_lead_days"] == 2
    assert summary["high_comfort_analysis_cell_count"] == 2
    assert summary["high_comfort_forecast_cell_count"] == 2
    assert summary["missed_high_comfort_cell_count"] == 0
    assert summary["false_high_comfort_cell_count"] == 0
    assert summary["high_comfort_precision"] == 1.0
    assert summary["high_comfort_recall"] == 1.0
    assert summary["verification_aggregation_policy"] == "experimental_regime_aware"
    assert summary["verification_aggregation_mode"] == "long_lead_soft"


def test_build_verification_file_prefix_adds_policy_suffix_for_non_baseline() -> None:
    assert (
        build_verification_file_prefix(
            region_name="southeast",
            resolved_forecast_model="gfs_seamless",
            forecast_lead_days=2,
            aggregation_policy="baseline",
        )
        == "comfortwx_verify_southeast_gfs_seamless_d2"
    )
    assert (
        build_verification_file_prefix(
            region_name="southeast",
            resolved_forecast_model="gfs_seamless",
            forecast_lead_days=2,
            aggregation_policy="experimental_regime_aware",
        )
        == "comfortwx_verify_southeast_gfs_seamless_d2_policy_experimental_regime_aware"
    )


def test_component_summary_fields_are_exposed() -> None:
    import pandas as pd

    from comfortwx.validation.verify_model import _component_summary_fields

    metrics = _component_summary_fields(
        pd.DataFrame(
            [
                {"component_name": "temp", "bias_mean": 1.2, "mae": 2.5},
                {"component_name": "reliability_score", "bias_mean": -0.7, "mae": 3.1},
            ]
        )
    )

    assert metrics["temp_bias_mean"] == 1.2
    assert metrics["temp_mae"] == 2.5
    assert metrics["reliability_score_bias_mean"] == -0.7
    assert metrics["reliability_score_mae"] == 3.1


def test_resolve_openmeteo_verification_forecast_model_prefers_hrrr_for_d1_default() -> None:
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="gfs_seamless",
            forecast_lead_days=1,
        )
        == "ncep_hrrr_conus"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="gfs_seamless",
            forecast_lead_days=2,
        )
        == "gfs_seamless"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="hrrr",
            forecast_lead_days=1,
        )
        == "ncep_hrrr_conus"
    )


def test_forecast_batch_does_not_fan_out_on_rate_limit(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_fetch_json(*_args, **_kwargs):
        calls["count"] += 1
        raise HTTPError("https://example.com", 429, "Too Many Requests", hdrs=None, fp=None)

    monkeypatch.setattr(openmeteo_verification, "_fetch_json", _fake_fetch_json)

    try:
        _fetch_forecast_payloads_for_batch(
            batch=[(35.0, -78.0), (36.0, -79.0)],
            valid_date=date(2026, 3, 20),
            forecast_lead_days=1,
            timezone_name="America/New_York",
            resolved_forecast_model="ncep_hrrr_conus",
        )
    except HTTPError as exc:
        assert exc.code == 429
    else:
        raise AssertionError("Expected HTTP 429 to be raised.")

    assert calls["count"] == 1


def test_subset_to_valid_local_day_raises_when_no_hours_match() -> None:
    dataset = xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), np.array([[[60.0]], [[61.0]]])),
        },
        coords={
            "time": xr.date_range("2026-03-19T00:00", periods=2, freq="1h"),
            "lat": [35.0],
            "lon": [-78.0],
        },
    )

    try:
        _subset_to_valid_local_day(dataset, date(2026, 3, 20))
    except ValueError as exc:
        assert "returned no hourly data" in str(exc)
    else:
        raise AssertionError("Expected empty valid-day subset to raise.")


def test_forecast_query_requests_previous_runs_variables_for_valid_day() -> None:
    query = _forecast_query_for_batch(
        batch=[(35.0, -78.0)],
        valid_date=date(2026, 3, 20),
        forecast_lead_days=2,
        timezone_name="America/New_York",
        model_name="gfs_seamless",
    )

    assert query["start_date"] == "2026-03-20"
    assert query["end_date"] == "2026-03-20"
    assert query["hourly"][0] == "temperature_2m_previous_day2"
    assert "run" not in query


def test_normalize_previous_run_payload_remaps_hourly_fields() -> None:
    payload = {
        "latitude": 35.0,
        "longitude": -78.0,
        "timezone": "America/New_York",
        "hourly_units": {
            "temperature_2m_previous_day3": "°F",
            "visibility_previous_day3": "m",
        },
        "hourly": {
            "time": ["2026-03-20T00:00", "2026-03-20T01:00"],
            "temperature_2m_previous_day3": [55.0, 56.0],
            "dew_point_2m_previous_day3": [45.0, 46.0],
            "precipitation_previous_day3": [0.0, 0.1],
            "precipitation_probability_previous_day3": [10.0, 40.0],
            "cloud_cover_previous_day3": [20.0, 40.0],
            "wind_speed_10m_previous_day3": [5.0, 7.0],
            "wind_gusts_10m_previous_day3": [12.0, 14.0],
            "weather_code_previous_day3": [1, 3],
            "visibility_previous_day3": [16093.44, 8046.72],
        },
    }

    normalized = _normalize_previous_run_payload(payload, forecast_lead_days=3)

    assert normalized["hourly"]["temperature_2m"] == [55.0, 56.0]
    assert normalized["hourly"]["visibility"] == [16093.44, 8046.72]
    assert normalized["hourly_units"]["temperature_2m"] == "°F"
    assert normalized["hourly_units"]["visibility"] == "m"
