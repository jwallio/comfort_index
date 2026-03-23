"""Synthetic hourly gridded forecast data for V1 development."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from nicewx.config import DEFAULT_DOMAIN


def _gaussian(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    center_lat: float,
    center_lon: float,
    lat_scale: float,
    lon_scale: float,
) -> np.ndarray:
    """Return a 2-D Gaussian feature over the domain."""

    return np.exp(
        -(((lat_grid - center_lat) / lat_scale) ** 2)
        - (((lon_grid - center_lon) / lon_scale) ** 2)
    )


def generate_mock_hourly_grid(
    valid_date: date,
    lat_points: int = 65,
    lon_points: int = 115,
) -> xr.Dataset:
    """Build a deterministic synthetic CONUS-like hourly grid."""

    times = pd.date_range(pd.Timestamp(valid_date), periods=24, freq="1h")
    lat = np.linspace(DEFAULT_DOMAIN["lat_min"], DEFAULT_DOMAIN["lat_max"], lat_points)
    lon = np.linspace(DEFAULT_DOMAIN["lon_min"], DEFAULT_DOMAIN["lon_max"], lon_points)

    hour = np.arange(24, dtype=float)[:, None, None]
    lat_2d = lat[None, :, None]
    lon_2d = lon[None, None, :]
    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")

    gulf = _gaussian(lat_mesh, lon_mesh, 29.5, -89.0, 5.5, 14.0)
    southeast = _gaussian(lat_mesh, lon_mesh, 33.0, -84.0, 6.0, 8.0)
    plains = _gaussian(lat_mesh, lon_mesh, 38.5, -98.0, 7.0, 6.0)
    rockies = _gaussian(lat_mesh, lon_mesh, 39.5, -106.0, 5.0, 4.0)
    desert_sw = _gaussian(lat_mesh, lon_mesh, 34.0, -113.0, 5.5, 6.0)
    pacific_nw = _gaussian(lat_mesh, lon_mesh, 46.0, -123.0, 4.0, 4.0)
    great_lakes = _gaussian(lat_mesh, lon_mesh, 44.0, -86.0, 4.0, 5.0)
    california = _gaussian(lat_mesh, lon_mesh, 37.0, -121.0, 7.0, 3.0)

    diurnal_wave = np.sin((hour - 9.0) / 24.0 * 2.0 * np.pi)
    afternoon_boost = np.clip(np.sin((hour - 11.0) / 12.0 * np.pi), 0.0, None)
    marine_night = np.clip(np.cos((hour - 6.0) / 12.0 * np.pi), 0.0, None)

    base_temp = (
        84.0
        - 1.22 * (lat_2d - 25.0)
        + desert_sw[None, :, :] * 8.0
        + southeast[None, :, :] * 3.0
        - rockies[None, :, :] * 8.5
        - pacific_nw[None, :, :] * 5.0
        - california[None, :, :] * 2.5
    )
    diurnal_amp = (
        8.0
        + desert_sw[None, :, :] * 3.5
        + plains[None, :, :] * 1.5
        - pacific_nw[None, :, :] * 1.2
    )
    temp_f = np.clip(base_temp + diurnal_amp * diurnal_wave, 28.0, 104.0)

    dewpoint_f = np.clip(
        38.0
        + gulf[None, :, :] * 26.0
        + southeast[None, :, :] * 7.0
        + great_lakes[None, :, :] * 4.0
        - desert_sw[None, :, :] * 9.0
        - rockies[None, :, :] * 8.0
        - 1.2 * diurnal_wave,
        16.0,
        78.0,
    )

    wind_mph = np.clip(
        4.0
        + plains[None, :, :] * 7.5
        + rockies[None, :, :] * 4.0
        + california[None, :, :] * 2.0
        + afternoon_boost * (2.0 + plains[None, :, :] * 3.0),
        0.0,
        30.0,
    )

    cloud_pct = np.clip(
        18.0
        + pacific_nw[None, :, :] * (45.0 + 15.0 * marine_night)
        + southeast[None, :, :] * (18.0 + 42.0 * afternoon_boost)
        + plains[None, :, :] * (10.0 + 50.0 * afternoon_boost)
        + great_lakes[None, :, :] * 18.0
        - desert_sw[None, :, :] * 18.0
        - rockies[None, :, :] * 8.0,
        0.0,
        100.0,
    )

    pop_pct = np.clip(
        5.0
        + pacific_nw[None, :, :] * (28.0 + 18.0 * marine_night)
        + southeast[None, :, :] * (10.0 + 58.0 * afternoon_boost)
        + plains[None, :, :] * (8.0 + 62.0 * afternoon_boost)
        + great_lakes[None, :, :] * 18.0
        - desert_sw[None, :, :] * 7.0,
        0.0,
        100.0,
    )

    qpf_in = np.clip(
        np.where(pop_pct >= 55.0, 0.01 + (pop_pct - 55.0) / 650.0, 0.0)
        + pacific_nw[None, :, :] * 0.018 * marine_night,
        0.0,
        0.22,
    )

    thunder = (
        (afternoon_boost >= 0.45)
        & (pop_pct >= 58.0)
        & ((southeast[None, :, :] >= 0.35) | (plains[None, :, :] >= 0.45))
    )

    gust_mph = np.clip(
        wind_mph
        + 3.0
        + plains[None, :, :] * 3.0
        + afternoon_boost * 4.0
        + thunder.astype(float) * 12.0,
        2.0,
        48.0,
    )

    return xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), temp_f.astype(np.float32)),
            "dewpoint_f": (("time", "lat", "lon"), dewpoint_f.astype(np.float32)),
            "wind_mph": (("time", "lat", "lon"), wind_mph.astype(np.float32)),
            "gust_mph": (("time", "lat", "lon"), gust_mph.astype(np.float32)),
            "cloud_pct": (("time", "lat", "lon"), cloud_pct.astype(np.float32)),
            "pop_pct": (("time", "lat", "lon"), pop_pct.astype(np.float32)),
            "qpf_in": (("time", "lat", "lon"), qpf_in.astype(np.float32)),
            "thunder": (("time", "lat", "lon"), thunder.astype(bool)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={
            "source": "synthetic_mock",
            "description": "Synthetic local-day forecast grid for nice weather scoring.",
            "time_basis": "naive local proxy hours",
        },
    )

