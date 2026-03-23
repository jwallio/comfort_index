"""Temperature comfort scoring."""

from __future__ import annotations

import xarray as xr

from nicewx.config import TEMP_SCORE_BINS
from nicewx.scoring._helpers import apply_interval_scores


def score_temperature(temp_f: xr.DataArray) -> xr.DataArray:
    """Score hourly temperature comfort on a 0-35 scale."""

    return apply_interval_scores(temp_f, TEMP_SCORE_BINS)

