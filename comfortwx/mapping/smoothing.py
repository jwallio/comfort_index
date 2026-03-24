"""Optional light smoothing helpers for score maps."""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None


def smooth_field(field: xr.DataArray, sigma: float = 0.75) -> xr.DataArray:
    """Apply light Gaussian smoothing if SciPy is available."""

    if gaussian_filter is None:
        return field
    values = field.values.astype(float)
    if not np.isnan(values).any():
        smoothed = gaussian_filter(values, sigma=sigma, mode="nearest")
        return xr.DataArray(smoothed, coords=field.coords, dims=field.dims, attrs=field.attrs)

    valid_mask = np.isfinite(values).astype(float)
    smoothed_values = gaussian_filter(np.nan_to_num(values, nan=0.0), sigma=sigma, mode="nearest")
    smoothed_weights = gaussian_filter(valid_mask, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(smoothed_weights > 0, smoothed_values / smoothed_weights, np.nan)
    return xr.DataArray(smoothed, coords=field.coords, dims=field.dims, attrs=field.attrs)
