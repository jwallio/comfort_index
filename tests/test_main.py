from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from comfortwx.main import _build_city_rankings_frame, _iter_pilot_valid_dates, _mosaic_file_prefix, _pilot_day_row, _region_file_prefix


def test_output_file_prefixes_preserve_existing_standard_names() -> None:
    assert _region_file_prefix("openmeteo", "southwest", "standard", "baseline") == "comfortwx_region_southwest_openmeteo"
    assert (
        _mosaic_file_prefix(
            ["southwest", "rockies"],
            "openmeteo",
            "standard",
            "baseline",
            "taper",
            "adaptive",
        )
        == "comfortwx_mosaic_southwest_rockies_openmeteo"
    )


def test_pilot_day_row_captures_status_metadata() -> None:
    row = _pilot_day_row(
        product_type="mosaic",
        product_name="southwest+rockies",
        valid_date=date(2026, 3, 24),
        result={"mosaic_daily_fields": Path("output/example.nc")},
        status="skipped",
        build_source="missing_regional_cache",
        notes="Missing plains",
    )

    assert row["status"] == "skipped"
    assert row["build_source"] == "missing_regional_cache"
    assert row["notes"] == "Missing plains"
    assert row["daily_fields_path"].endswith("example.nc")


def test_iter_pilot_valid_dates_expands_consecutive_days() -> None:
    valid_dates = _iter_pilot_valid_dates(date(2026, 3, 24), 3)

    assert valid_dates == [date(2026, 3, 24), date(2026, 3, 25), date(2026, 3, 26)]


def test_iter_pilot_valid_dates_rejects_nonpositive_span() -> None:
    with pytest.raises(ValueError):
        _iter_pilot_valid_dates(date(2026, 3, 24), 0)


def test_build_city_rankings_frame_returns_best_and_worst_groups() -> None:
    daily = xr.Dataset(
        {
            "daily_score": (("lat", "lon"), np.array([[20.0, 40.0], [70.0, 90.0]], dtype=np.float32)),
            "category_index": (("lat", "lon"), np.array([[0, 1], [3, 4]], dtype=np.int32)),
        },
        coords={
            "lat": np.array([30.0, 40.0], dtype=np.float32),
            "lon": np.array([-100.0, -80.0], dtype=np.float32),
        },
    )

    ranking_frame = _build_city_rankings_frame(daily)

    assert not ranking_frame.empty
    assert set(ranking_frame["ranking_group"]) == {"best", "worst"}
    assert int((ranking_frame["ranking_group"] == "best").sum()) == 10
    assert int((ranking_frame["ranking_group"] == "worst").sum()) == 10
