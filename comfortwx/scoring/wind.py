"""Wind comfort scoring."""

from __future__ import annotations

import xarray as xr

from comfortwx.config import WIND_SCORE_BINS
from comfortwx.scoring._helpers import apply_interval_scores


def score_wind(wind_mph: xr.DataArray) -> xr.DataArray:
    """Score wind comfort on a 0-10 scale."""

    return apply_interval_scores(wind_mph, WIND_SCORE_BINS)
