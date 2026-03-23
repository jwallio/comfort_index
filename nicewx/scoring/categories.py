"""Daily score categories."""

from __future__ import annotations

import numpy as np
import xarray as xr

from nicewx.config import CATEGORIES


def categorize_scores(
    score: xr.DataArray,
    pristine_allowed: xr.DataArray | None = None,
) -> xr.DataArray:
    """Map raw scores to category indices, with optional pristine gating."""

    category_index = xr.full_like(score, fill_value=0, dtype=int)
    for index, category in enumerate(CATEGORIES):
        mask = (score >= category.lower) & (score <= category.upper)
        category_index = xr.where(mask, index, category_index)
    if pristine_allowed is not None:
        perfect_index = len(CATEGORIES) - 2
        pristine_index = len(CATEGORIES) - 1
        category_index = xr.where(
            (category_index == pristine_index) & (~pristine_allowed.astype(bool)),
            perfect_index,
            category_index,
        )
    return category_index


def category_name_from_value(score: float, pristine_allowed: bool = True) -> str:
    """Return the string category for a scalar score."""

    bounded_score = float(np.clip(score, 0.0, 100.0))
    if bounded_score >= CATEGORIES[-1].lower and not pristine_allowed:
        return CATEGORIES[-2].name
    for category in CATEGORIES:
        if category.lower <= bounded_score <= category.upper:
            return category.name
    return CATEGORIES[0].name


def category_name_from_index(index: int) -> str:
    """Return the category name for an integer category index."""

    bounded_index = int(np.clip(index, 0, len(CATEGORIES) - 1))
    return CATEGORIES[bounded_index].name


def category_labels() -> list[str]:
    """Return ordered category names."""

    return [category.name for category in CATEGORIES]


def category_colors() -> list[str]:
    """Return ordered category colors."""

    return [category.color for category in CATEGORIES]
