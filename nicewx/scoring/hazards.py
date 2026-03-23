"""Hazard penalties and score caps."""

from __future__ import annotations

import xarray as xr

from nicewx.config import THUNDER_HOURLY_CAP
from nicewx.scoring._helpers import zeros_like


def hazard_penalty(gust_mph: xr.DataArray, thunder: xr.DataArray | None = None) -> xr.DataArray:
    """Return hazard penalties as positive points to subtract."""

    penalty = zeros_like(gust_mph)
    penalty = xr.where(gust_mph >= 25.0, 5.0, penalty)
    penalty = xr.where(gust_mph >= 35.0, 12.0, penalty)
    penalty = xr.where(gust_mph >= 45.0, 20.0, penalty)
    if thunder is not None:
        penalty = penalty + xr.where(thunder.astype(bool), 20.0, 0.0)
    return penalty


def hazard_cap(reference: xr.DataArray, thunder: xr.DataArray | None = None) -> xr.DataArray:
    """Return hourly caps imposed by hazardous weather."""

    cap = xr.full_like(reference, fill_value=100.0, dtype=float)
    if thunder is not None:
        cap = xr.where(thunder.astype(bool), THUNDER_HOURLY_CAP, cap)
    return cap

