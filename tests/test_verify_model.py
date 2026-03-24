from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr

from comfortwx.data.openmeteo_verification import _normalize_openmeteo_verification_payload
from comfortwx.validation.verify_model import _verification_summary


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
            "mesh_profile": "standard",
            "mesh_point_count": 4,
        },
        valid_date=date(2026, 3, 20),
    )

    assert summary["score_mae"] == 4.0
    assert summary["score_max_abs_diff"] == 5.0
    assert summary["exact_category_agreement_fraction"] == 0.75
    assert summary["near_category_agreement_fraction"] == 1.0
