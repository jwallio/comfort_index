"""Precipitation reliability scoring."""

from __future__ import annotations

import xarray as xr

from comfortwx.config import POP_SCORE_BINS
from comfortwx.scoring._helpers import apply_interval_scores


def score_precipitation(
    pop_pct: xr.DataArray,
    qpf_in: xr.DataArray,
    thunder: xr.DataArray | None = None,
) -> xr.DataArray:
    """Score precip reliability on a 0-15 scale."""

    pop_score = apply_interval_scores(pop_pct, POP_SCORE_BINS)
    precip_score = xr.where(pop_pct > 70.0, 0.0, pop_score)
    precip_score = xr.where(qpf_in >= 0.10, 0.0, precip_score)
    precip_score = xr.where((qpf_in >= 0.05) & (qpf_in < 0.10), 4.0, precip_score)
    precip_score = xr.where((qpf_in >= 0.01) & (qpf_in < 0.05), 8.0, precip_score)
    if thunder is not None:
        precip_score = xr.where(thunder.astype(bool), 0.0, precip_score)
    return precip_score

