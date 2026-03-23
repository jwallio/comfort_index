"""Interaction adjustments for combined weather effects."""

from __future__ import annotations

import xarray as xr

from nicewx.config import (
    TEMP_CLOUD_ADJUSTMENTS,
    TEMP_DEWPOINT_PENALTIES,
    TEMP_WIND_ADJUSTMENTS,
)
from nicewx.scoring._helpers import zeros_like


def temperature_dewpoint_adjustment(
    temp_f: xr.DataArray,
    dewpoint_f: xr.DataArray,
) -> xr.DataArray:
    """Apply warm-season humidity discomfort penalties."""

    adjustment = zeros_like(temp_f)
    for temp_min, dewpoint_min, penalty in TEMP_DEWPOINT_PENALTIES:
        adjustment = xr.where((temp_f >= temp_min) & (dewpoint_f >= dewpoint_min), penalty, adjustment)
    return adjustment


def temperature_wind_adjustment(
    temp_f: xr.DataArray,
    dewpoint_f: xr.DataArray,
    wind_mph: xr.DataArray,
) -> xr.DataArray:
    """Reward helpful warm breezes and penalize raw windy chill."""

    adjustment = zeros_like(temp_f)
    for temp_min, temp_max, dewpoint_max, wind_min, wind_max, delta in TEMP_WIND_ADJUSTMENTS:
        mask = xr.ones_like(temp_f, dtype=bool)
        if temp_min is not None:
            mask = mask & (temp_f >= temp_min)
        if temp_max is not None:
            mask = mask & (temp_f <= temp_max)
        if dewpoint_max is not None:
            mask = mask & (dewpoint_f <= dewpoint_max)
        if wind_min is not None:
            mask = mask & (wind_mph >= wind_min)
        if wind_max is not None:
            mask = mask & (wind_mph <= wind_max)
        adjustment = xr.where(mask, delta, adjustment)
    return adjustment


def temperature_cloud_adjustment(
    temp_f: xr.DataArray,
    cloud_pct: xr.DataArray,
) -> xr.DataArray:
    """Adjust cloud preference based on thermal regime."""

    adjustment = zeros_like(temp_f)
    for temp_min, temp_max, cloud_min, cloud_max, delta in TEMP_CLOUD_ADJUSTMENTS:
        mask = xr.ones_like(temp_f, dtype=bool)
        if temp_min is not None:
            mask = mask & (temp_f >= temp_min)
        if temp_max is not None:
            mask = mask & (temp_f <= temp_max)
        if cloud_min is not None:
            mask = mask & (cloud_pct >= cloud_min)
        if cloud_max is not None:
            mask = mask & (cloud_pct <= cloud_max)
        adjustment = xr.where(mask, delta, adjustment)
    return adjustment


def total_interaction_adjustment(
    temp_f: xr.DataArray,
    dewpoint_f: xr.DataArray,
    wind_mph: xr.DataArray,
    cloud_pct: xr.DataArray,
) -> xr.DataArray:
    """Combine independent interaction families into one field."""

    return (
        temperature_dewpoint_adjustment(temp_f, dewpoint_f)
        + temperature_wind_adjustment(temp_f, dewpoint_f, wind_mph)
        + temperature_cloud_adjustment(temp_f, cloud_pct)
    )
