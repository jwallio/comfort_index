from __future__ import annotations

import json
from pathlib import Path
import pytest

import comfortwx.data.openmeteo as openmeteo
from comfortwx.data.openmeteo import (
    _fetch_json,
    assemble_point_datasets_to_grid,
    merge_openmeteo_air_quality,
    normalize_openmeteo_forecast_response,
)
from comfortwx.data.openmeteo_reliability import openmeteo_request_context, reset_openmeteo_request_records


def test_normalize_openmeteo_forecast_response_maps_fields() -> None:
    payload = {
        "latitude": 35.8,
        "longitude": -78.6,
        "timezone": "America/New_York",
        "hourly": {
            "time": ["2026-03-22T00:00", "2026-03-22T01:00"],
            "temperature_2m": [68.0, 66.0],
            "relative_humidity_2m": [70.0, 72.0],
            "dew_point_2m": [58.0, 57.0],
            "wind_speed_10m": [8.0, 7.0],
            "wind_gusts_10m": [12.0, 11.0],
            "cloud_cover": [25.0, 35.0],
            "precipitation_probability": [10.0, 15.0],
            "precipitation": [0.0, 0.02],
            "weather_code": [1, 95],
            "visibility": [16093.44, 8046.72],
            "cape": [100.0, 1500.0],
        },
    }

    dataset = normalize_openmeteo_forecast_response(payload, requested_lat=35.8, requested_lon=-78.6)

    assert set(["temp_f", "dewpoint_f", "wind_mph", "gust_mph", "cloud_pct", "pop_pct", "qpf_in"]).issubset(dataset.data_vars)
    assert float(dataset["visibility_mi"].isel(time=0).values.squeeze()) == 10.0
    assert bool(dataset["thunder"].isel(time=1).values.squeeze()) is True


def test_merge_openmeteo_air_quality_adds_optional_fields() -> None:
    payload = {
        "latitude": 35.8,
        "longitude": -78.6,
        "timezone": "America/New_York",
        "hourly": {
            "time": ["2026-03-22T00:00"],
            "temperature_2m": [68.0],
            "relative_humidity_2m": [70.0],
            "dew_point_2m": [58.0],
            "wind_speed_10m": [8.0],
            "wind_gusts_10m": [12.0],
            "cloud_cover": [25.0],
            "precipitation_probability": [10.0],
            "precipitation": [0.0],
            "weather_code": [1],
            "visibility": [16093.44],
            "cape": [100.0],
        },
    }
    air_payload = {
        "hourly": {
            "us_aqi": [45.0],
            "pm2_5": [8.0],
        }
    }

    dataset = normalize_openmeteo_forecast_response(payload, requested_lat=35.8, requested_lon=-78.6)
    merged = merge_openmeteo_air_quality(dataset, air_payload)

    assert "aqi" in merged
    assert "pm25" in merged


@pytest.mark.parametrize("region_name", ["west_coast", "southeast", "southwest", "rockies", "plains", "great_lakes", "northeast"])
def test_assemble_point_datasets_to_grid_builds_regional_mesh(region_name: str) -> None:
    payload_a = {
        "latitude": 30.0,
        "longitude": -90.0,
        "timezone": "America/New_York",
        "hourly": {
            "time": ["2026-03-22T00:00"],
            "temperature_2m": [70.0],
            "relative_humidity_2m": [70.0],
            "dew_point_2m": [59.0],
            "wind_speed_10m": [8.0],
            "wind_gusts_10m": [12.0],
            "cloud_cover": [25.0],
            "precipitation_probability": [10.0],
            "precipitation": [0.0],
            "weather_code": [1],
            "visibility": [16093.44],
            "cape": [100.0],
        },
    }
    payload_b = {
        "latitude": 33.5,
        "longitude": -86.5,
        "timezone": "America/New_York",
        "hourly": {
            "time": ["2026-03-22T00:00"],
            "temperature_2m": [75.0],
            "relative_humidity_2m": [60.0],
            "dew_point_2m": [60.0],
            "wind_speed_10m": [9.0],
            "wind_gusts_10m": [14.0],
            "cloud_cover": [40.0],
            "precipitation_probability": [20.0],
            "precipitation": [0.02],
            "weather_code": [2],
            "visibility": [12000.0],
            "cape": [200.0],
        },
    }
    datasets = {
        (30.0, -90.0): normalize_openmeteo_forecast_response(payload_a, requested_lat=30.0, requested_lon=-90.0),
        (33.5, -86.5): normalize_openmeteo_forecast_response(payload_b, requested_lat=33.5, requested_lon=-86.5),
    }
    assembled = assemble_point_datasets_to_grid(
        point_datasets=datasets,
        lat_values=[30.0, 33.5],
        lon_values=[-90.0, -86.5],
        region_name=region_name,
    )

    assert assembled.sizes["lat"] == 2
    assert assembled.sizes["lon"] == 2
    assert float(assembled["temp_f"].sel(lat=33.5, lon=-86.5).values.squeeze()) == 75.0


def test_fetch_json_uses_verification_request_cache(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}
    payload = {"hourly": {"time": ["2026-03-22T00:00"]}}

    def _fake_fetch_with_retries(**_kwargs):
        calls["count"] += 1
        return payload

    monkeypatch.setattr(openmeteo, "OPENMETEO_REQUEST_CACHE_DIR", tmp_path)
    monkeypatch.setattr(openmeteo, "fetch_with_retries", _fake_fetch_with_retries)

    reset_openmeteo_request_records()
    with openmeteo_request_context(workflow="verification_benchmark", label="cache-test", run_slug="cache_test"):
        first = _fetch_json("https://archive-api.open-meteo.com/v1/archive", {"latitude": 35.0, "longitude": -78.0})
        second = _fetch_json("https://archive-api.open-meteo.com/v1/archive", {"latitude": 35.0, "longitude": -78.0})

    assert first == payload
    assert second == payload
    assert calls["count"] == 1
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    assert json.loads(cache_files[0].read_text(encoding="utf-8")) == payload
