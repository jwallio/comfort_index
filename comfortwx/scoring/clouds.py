"""Cloud cover comfort scoring."""

from __future__ import annotations

import xarray as xr

from comfortwx.config import CLOUD_REGIMES


def _score_cloud_regime(temp_f: xr.DataArray, cloud_pct: xr.DataArray, regime) -> xr.DataArray:
    """Score a single cloud regime using ideal ranges and distance penalties."""

    low_gap = (regime.ideal_min - cloud_pct).clip(min=0.0)
    high_gap = (cloud_pct - regime.ideal_max).clip(min=0.0)

    score = (
        regime.max_score
        - (low_gap / 10.0) * regime.low_penalty_per_10pct
        - (high_gap / 10.0) * regime.high_penalty_per_10pct
    )
    score = xr.where(cloud_pct >= regime.overcast_threshold, score - regime.overcast_penalty, score)
    if regime.hot_sun_threshold is not None:
        score = xr.where(
            (temp_f >= 86.0) & (cloud_pct <= regime.hot_sun_threshold),
            score - regime.hot_sun_penalty,
            score,
        )
    score = xr.where(cloud_pct >= regime.elite_cap_threshold, score.clip(max=regime.elite_cap_score), score)
    return score.clip(min=0.0, max=regime.max_score)


def score_clouds(temp_f: xr.DataArray, cloud_pct: xr.DataArray) -> xr.DataArray:
    """Score cloud cover with temperature-aware, non-static regime logic."""

    score = xr.zeros_like(temp_f, dtype=float)
    for regime in CLOUD_REGIMES:
        mask = xr.ones_like(temp_f, dtype=bool)
        if regime.temp_min is not None:
            mask = mask & (temp_f >= regime.temp_min)
        if regime.temp_max is not None:
            mask = mask & (temp_f <= regime.temp_max)
        regime_score = _score_cloud_regime(temp_f=temp_f, cloud_pct=cloud_pct, regime=regime)
        score = xr.where(mask, regime_score, score)
    return score
