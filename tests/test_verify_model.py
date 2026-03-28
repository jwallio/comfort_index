from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import xarray as xr
from urllib.error import HTTPError

import comfortwx.data.openmeteo_verification as openmeteo_verification
from comfortwx.data.openmeteo_verification import (
    _blend_forecast_grids,
    _fetch_forecast_payloads_for_batch,
    _subset_to_valid_local_day,
    _forecast_query_for_batch,
    _normalize_openmeteo_verification_payload,
    _normalize_previous_run_payload,
    resolve_openmeteo_verification_forecast_model,
)
from comfortwx.data.noaa_analysis import _precip_analysis_url, _surface_analysis_url, _utc_hour_schedule
from comfortwx.data.ndfd_forecast import _select_catalog_entry
from comfortwx.validation.verify_model import (
    _apply_truth_observability_overrides,
    _verification_summary,
    build_verification_file_prefix,
)
from comfortwx.data.noaa_analysis import _nearest_point_lookup, _observed_occurrence_pop_from_qpf


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


def test_noaa_observed_occurrence_pop_is_deterministic() -> None:
    values = _observed_occurrence_pop_from_qpf(np.array([0.0, 0.0005, 0.001, 0.02]))
    assert values.tolist() == [0.0, 0.0, 100.0, 100.0]


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
            analysis_model="noaa_urma_rtma",
            forecast_lead_days=2,
            aggregation_policy="baseline",
        )
        == "comfortwx_verify_southeast_gfs_seamless_noaa_urma_rtma_d2"
    )


def test_truth_observability_override_disables_thunder_on_both_sides() -> None:
    forecast_hourly = xr.Dataset(
        data_vars={
            "thunder": (("time", "lat", "lon"), np.array([[[True]], [[False]]], dtype=bool)),
        },
        coords={"time": xr.date_range("2026-03-20T00:00", periods=2, freq="1h"), "lat": [35.0], "lon": [-78.0]},
    )
    analysis_hourly = xr.Dataset(
        data_vars={
            "thunder": (("time", "lat", "lon"), np.array([[[False]], [[False]]], dtype=bool)),
        },
        coords={"time": xr.date_range("2026-03-20T00:00", periods=2, freq="1h"), "lat": [35.0], "lon": [-78.0]},
    )

    forecast_adjusted, analysis_adjusted = _apply_truth_observability_overrides(
        forecast_hourly=forecast_hourly,
        analysis_hourly=analysis_hourly,
        metadata={"analysis_has_thunder_truth": False},
    )

    assert not bool(forecast_adjusted["thunder"].any())
    assert not bool(analysis_adjusted["thunder"].any())


def test_noaa_lookup_normalizes_0_360_longitudes() -> None:
    latitude = np.array([[35.0, 35.0], [34.0, 34.0]])
    longitude = np.array([[270.0, 271.0], [270.0, 271.0]])
    lookup = _nearest_point_lookup(
        latitude=latitude,
        longitude=longitude,
        coordinate_pairs=[(35.0, -90.0), (34.0, -89.0)],
        bounds=(-91.0, -88.0, 33.0, 36.0),
    )
    assert lookup[(35.0, -90.0)] == (0, 0)
    assert lookup[(34.0, -89.0)] == (1, 1)
    assert (
        build_verification_file_prefix(
            region_name="southeast",
            resolved_forecast_model="gfs_seamless",
            analysis_model="noaa_urma_rtma",
            forecast_lead_days=2,
            aggregation_policy="experimental_regime_aware",
        )
        == "comfortwx_verify_southeast_gfs_seamless_noaa_urma_rtma_d2_policy_experimental_regime_aware"
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
            region_name="southeast",
        )
        == "ncep_hrrr_conus"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="gfs_seamless",
            forecast_lead_days=1,
            region_name="west_coast",
        )
        == "gfs_seamless"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="gfs_seamless",
            forecast_lead_days=2,
            region_name="west_coast",
        )
        == "gfs_seamless"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="hrrr",
            forecast_lead_days=1,
            region_name="west_coast",
        )
        == "ncep_hrrr_conus"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="gfs_seamless",
            forecast_lead_days=1,
            forecast_model_mode="exact",
        )
        == "gfs_seamless"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="ecmwf_ifs",
            forecast_lead_days=1,
            forecast_model_mode="exact",
        )
        == "ecmwf_ifs"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="nws_ndfd_hourly",
            forecast_lead_days=1,
            region_name="west_coast",
        )
        == "nws_ndfd_hourly"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="nws_ndfd_gfs_blend",
            forecast_lead_days=1,
            region_name="west_coast",
        )
        == "nws_ndfd_gfs_blend"
    )
    assert (
        resolve_openmeteo_verification_forecast_model(
            requested_model="nws_ndfd_cloud_blend",
            forecast_lead_days=1,
            region_name="west_coast",
        )
        == "nws_ndfd_cloud_blend"
    )


def test_blend_forecast_grids_subsets_fallback_to_primary_time_axis() -> None:
    primary = xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), np.array([[[70.0]], [[71.0]]], dtype=np.float32)),
            "pop_pct": (("time", "lat", "lon"), np.array([[[10.0]], [[20.0]]], dtype=np.float32)),
        },
        coords={"time": xr.date_range("2024-03-20T08:00", periods=2, freq="1h"), "lat": [40.0], "lon": [-123.0]},
    )
    fallback = xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), np.array([[[60.0]], [[61.0]], [[62.0]]], dtype=np.float32)),
            "pop_pct": (("time", "lat", "lon"), np.array([[[30.0]], [[40.0]], [[50.0]]], dtype=np.float32)),
            "qpf_in": (("time", "lat", "lon"), np.array([[[0.01]], [[0.02]], [[0.03]]], dtype=np.float32)),
        },
        coords={"time": xr.date_range("2024-03-20T07:00", periods=3, freq="1h"), "lat": [40.0], "lon": [-123.0]},
    )

    blended = _blend_forecast_grids(
        primary_grid=primary,
        fallback_grid=fallback,
        primary_vars=("temp_f", "pop_pct"),
        source_label="blend",
    )

    assert blended.sizes["time"] == 2
    assert blended["time"].equals(primary["time"])
    assert float(blended["temp_f"].isel(time=0).values.squeeze()) == 70.0
    assert np.isclose(float(blended["qpf_in"].isel(time=0).values.squeeze()), 0.02)


def test_select_catalog_entry_prefers_conus_tile_for_west_coast() -> None:
    entries = [
        {
            "dataset_id": "NDFD_kwbn/access/202403/20240319/YEAZ88_KWBN_202403191150",
            "label": "Surface Temperature",
            "run_timestamp_utc": datetime(2024, 3, 19, 11, 50, tzinfo=timezone.utc),
            "wmo_code": "YEAZ88",
            "center_code": "KWBN",
        },
        {
            "dataset_id": "NDFD_kwbn/access/202403/20240319/YEUZ98_KWBN_202403191150",
            "label": "Surface Temperature",
            "run_timestamp_utc": datetime(2024, 3, 19, 11, 50, tzinfo=timezone.utc),
            "wmo_code": "YEUZ98",
            "center_code": "KWBN",
        },
    ]
    selected = _select_catalog_entry(
        entries,
        label="Surface Temperature",
        target_run_timestamp_utc=datetime(2024, 3, 19, 12, 0, tzinfo=timezone.utc),
        region_name="west_coast",
    )
    assert selected["wmo_code"] == "YEUZ98"


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


def test_noaa_analysis_urls_and_local_hour_schedule() -> None:
    schedule = _utc_hour_schedule(date(2025, 3, 20), "America/New_York")

    assert schedule[0][0].hour == 8
    assert schedule[-1][0].hour == 20
    assert schedule[0][1].hour == 12
    assert "urma2p5.20250320/urma2p5.t12z.2dvaranl_ndfd.grb2_wexp" in _surface_analysis_url(
        source="urma",
        utc_dt=schedule[0][1],
    )
    assert "rtma2p5.2025032012.pcp.184.grb2" in _precip_analysis_url(
        source="rtma",
        utc_dt=schedule[0][1],
    )
