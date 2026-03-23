"""Internal scoring helpers."""

from __future__ import annotations

import xarray as xr


def apply_interval_scores(
    values: xr.DataArray,
    bins: tuple[tuple[float | None, float | None, float], ...],
    default: float = 0.0,
) -> xr.DataArray:
    """Apply inclusive interval scoring to an xarray field."""

    scored = xr.full_like(values, fill_value=float(default), dtype=float)
    for lower, upper, score in bins:
        mask = xr.ones_like(values, dtype=bool)
        if lower is not None:
            mask = mask & (values >= lower)
        if upper is not None:
            mask = mask & (values <= upper)
        scored = xr.where(mask, float(score), scored)
    return scored


def zeros_like(values: xr.DataArray) -> xr.DataArray:
    """Return a floating zero field matching the input shape."""

    return xr.zeros_like(values, dtype=float)

