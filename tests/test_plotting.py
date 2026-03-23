from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr

from nicewx.mapping.plotting import render_daily_maps


def test_render_daily_maps_writes_debug_and_presentation_outputs(tmp_path) -> None:
    daily = xr.Dataset(
        {
            "daily_score": (("lat", "lon"), np.array([[43.5, 47.2], [81.0, 91.0]], dtype=float)),
            "category_index": (("lat", "lon"), np.array([[0, 1], [3, 4]], dtype=float)),
        },
        coords={"lat": [30.0, 31.0], "lon": [-90.0, -89.0]},
    )

    outputs = render_daily_maps(
        daily=daily,
        valid_date=date(2026, 3, 24),
        output_dir=tmp_path,
        file_prefix="test_render",
        extent=(-91.0, -88.0, 29.0, 32.0),
        map_label="Test Region",
        include_presentation=True,
        presentation_canvas="stitched_conus",
        product_metadata={
            "product_title": "Nice Weather Outlook — Stitched CONUS Pilot",
            "subtitle_source_line": "Open-Meteo stitched regional pilot",
        },
    )

    assert outputs["raw_map"].exists()
    assert outputs["category_map"].exists()
    assert outputs["presentation_raw_map"].exists()
    assert outputs["presentation_category_map"].exists()
