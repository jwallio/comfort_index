from __future__ import annotations

from datetime import date
from pathlib import Path

from comfortwx.main import _mosaic_file_prefix, _pilot_day_row, _region_file_prefix


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
