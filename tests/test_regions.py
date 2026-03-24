from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr

from comfortwx.data.mock_data import generate_mock_hourly_grid
from comfortwx.mapping.mosaic import RegionalDailyRaster, weighted_overlap_merge
from comfortwx.mapping.regions import get_region_definition, region_blend_weights, subset_dataset_to_region


def test_region_subset_uses_expanded_bounds() -> None:
    dataset = generate_mock_hourly_grid(date(2026, 5, 18), lat_points=20, lon_points=30)
    region = get_region_definition("southeast")
    subset = subset_dataset_to_region(dataset, region, include_overlap=True)

    assert float(subset["lat"].min()) >= region.expanded_bounds[2] - 1e-6
    assert float(subset["lat"].max()) <= region.expanded_bounds[3] + 1e-6
    assert float(subset["lon"].min()) >= region.expanded_bounds[0] - 1e-6
    assert float(subset["lon"].max()) <= region.expanded_bounds[1] + 1e-6


def test_region_blend_weights_taper_to_edges() -> None:
    region = get_region_definition("southeast")
    lat = xr.DataArray(
        np.array([region.lat_min, region.lat_min - region.overlap_buffer]),
        dims=("lat",),
        coords={"lat": [region.lat_min, region.lat_min - region.overlap_buffer]},
    )
    lon = xr.DataArray(np.array([region.lon_min]), dims=("lon",), coords={"lon": [region.lon_min]})
    weights = region_blend_weights(lat, lon, region)

    assert float(weights.sel(lat=region.lat_min, lon=region.lon_min).values.squeeze()) == 1.0
    assert float(weights.sel(lat=region.lat_min - region.overlap_buffer, lon=region.lon_min).values.squeeze()) == 0.0


def test_weighted_overlap_merge_merges_two_mock_rasters() -> None:
    lat = xr.DataArray([30.0, 31.0], dims=("lat",))
    lon = xr.DataArray([-90.0, -89.0], dims=("lon",))
    region = get_region_definition("southeast")
    daily_a = xr.Dataset({"daily_score": (("lat", "lon"), np.full((2, 2), 60.0))}, coords={"lat": lat, "lon": lon})
    daily_b = xr.Dataset({"daily_score": (("lat", "lon"), np.full((2, 2), 80.0))}, coords={"lat": lat, "lon": lon})
    merged = weighted_overlap_merge(
        [
            RegionalDailyRaster(region=region, daily=daily_a),
            RegionalDailyRaster(region=region, daily=daily_b),
        ]
    )

    assert "daily_score" in merged
    assert float(merged["daily_score"].mean().values) == 70.0
