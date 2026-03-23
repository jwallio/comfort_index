from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr

from nicewx.validation.score_palette_compare import render_score_palette_variants


def test_render_score_palette_variants_writes_all_variant_pngs(tmp_path) -> None:
    valid_date = date(2026, 3, 24)
    daily_fields_path = tmp_path / "nicewx_mosaic_test_daily_fields_20260324.nc"
    daily = xr.Dataset(
        {
            "daily_score": (("lat", "lon"), np.array([[15.0, 35.0], [65.0, 92.0]], dtype=float)),
            "category_index": (("lat", "lon"), np.array([[0.0, 1.0], [3.0, 4.0]], dtype=float)),
        },
        coords={"lat": [30.0, 31.0], "lon": [-90.0, -89.0]},
    )
    daily.to_netcdf(daily_fields_path)

    written = render_score_palette_variants(
        daily_fields_path=daily_fields_path,
        valid_date=valid_date,
        output_dir=tmp_path,
    )

    assert len(written) == 3
    assert all(path.exists() for path in written)
    assert any("premium_muted" in path.name for path in written)
    assert any("bold_social" in path.name for path in written)
    assert any("blue_green_yellow_magenta" in path.name for path in written)
