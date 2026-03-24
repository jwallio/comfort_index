"""Optional air-quality, smoke, and visibility hooks."""

from __future__ import annotations

import xarray as xr

from comfortwx.config import (
    AQI_CAP_BINS,
    AQI_PENALTY_BINS,
    PM25_CAP_BINS,
    PM25_PENALTY_BINS,
    SMOKE_CAP_BINS,
    SMOKE_PENALTY_BINS,
    VISIBILITY_CAP_BINS,
    VISIBILITY_PENALTY_BINS,
)
from comfortwx.scoring._helpers import apply_interval_scores, zeros_like


def _cap_from_bins(values: xr.DataArray, bins: tuple[tuple[float | None, float | None, float], ...]) -> xr.DataArray:
    return apply_interval_scores(values, bins, default=100.0)


def score_optional_air_quality(dataset: xr.Dataset) -> tuple[dict[str, xr.DataArray], xr.DataArray]:
    """Return optional AQI/smoke/visibility adjustments and a combined cap.

    Clean air stays close to neutral. Poor air quality and low visibility suppress
    the top end through both penalties and category-limiting caps.
    """

    reference = dataset["temp_f"]
    zero_field = zeros_like(reference)
    cap = xr.full_like(reference, fill_value=100.0, dtype=float)

    components = {
        "air_quality_penalty": zero_field.copy(),
        "visibility_penalty": zero_field.copy(),
        "air_quality_cap": cap.copy(),
    }

    if "aqi" in dataset:
        aqi_penalty = apply_interval_scores(dataset["aqi"], AQI_PENALTY_BINS)
        aqi_cap = _cap_from_bins(dataset["aqi"], AQI_CAP_BINS)
        components["air_quality_penalty"] = components["air_quality_penalty"] + aqi_penalty
        cap = xr.where(aqi_cap < cap, aqi_cap, cap)

    if "pm25" in dataset:
        pm25_penalty = apply_interval_scores(dataset["pm25"], PM25_PENALTY_BINS)
        pm25_cap = _cap_from_bins(dataset["pm25"], PM25_CAP_BINS)
        components["air_quality_penalty"] = components["air_quality_penalty"] + pm25_penalty
        cap = xr.where(pm25_cap < cap, pm25_cap, cap)

    if "smoke" in dataset:
        smoke_penalty = apply_interval_scores(dataset["smoke"], SMOKE_PENALTY_BINS)
        smoke_cap = _cap_from_bins(dataset["smoke"], SMOKE_CAP_BINS)
        components["air_quality_penalty"] = components["air_quality_penalty"] + smoke_penalty
        cap = xr.where(smoke_cap < cap, smoke_cap, cap)

    if "visibility_mi" in dataset:
        visibility_penalty = apply_interval_scores(dataset["visibility_mi"], VISIBILITY_PENALTY_BINS)
        visibility_cap = _cap_from_bins(dataset["visibility_mi"], VISIBILITY_CAP_BINS)
        components["visibility_penalty"] = visibility_penalty
        cap = xr.where(visibility_cap < cap, visibility_cap, cap)

    components["air_quality_cap"] = cap
    return components, cap
