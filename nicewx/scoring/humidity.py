"""Humidity and dew point comfort scoring."""

from __future__ import annotations

import xarray as xr

from nicewx.config import DEWPOINT_SCORE_BINS
from nicewx.scoring._helpers import apply_interval_scores


def score_dewpoint(dewpoint_f: xr.DataArray) -> xr.DataArray:
    """Score dew point comfort on a 0-20 scale."""

    return apply_interval_scores(dewpoint_f, DEWPOINT_SCORE_BINS)

