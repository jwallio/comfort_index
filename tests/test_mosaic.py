from __future__ import annotations

import numpy as np
import xarray as xr

from comfortwx.config import WESTERN_MOSAIC_FIXED_TARGET_GRID
from comfortwx.mapping.mosaic import (
    RegionalDailyRaster,
    build_common_target_grid,
    build_fixed_target_grid,
    mosaic_regional_rasters,
)
from comfortwx.mapping.regions import RegionDefinition


def _daily_dataset(lat_values: list[float], lon_values: list[float], fill_value: float) -> xr.Dataset:
    return xr.Dataset(
        {"daily_score": (("lat", "lon"), np.full((len(lat_values), len(lon_values)), fill_value))},
        coords={"lat": lat_values, "lon": lon_values},
    )


def test_build_common_target_grid_uses_union_extent_and_finest_spacing() -> None:
    raster_a = RegionalDailyRaster(
        region=RegionDefinition("alpha", 30.0, 32.0, -115.0, -113.0, 0.0),
        daily=_daily_dataset([30.0, 32.0], [-115.0, -113.0], 60.0),
    )
    raster_b = RegionalDailyRaster(
        region=RegionDefinition("beta", 34.0, 36.0, -85.0, -83.0, 0.0),
        daily=_daily_dataset([34.0, 35.0, 36.0], [-85.0, -84.0, -83.0], 80.0),
    )

    target_lat, target_lon = build_common_target_grid([raster_a, raster_b])

    assert float(target_lat.min().values) == 30.0
    assert float(target_lat.max().values) == 36.0
    assert float(target_lon.min().values) == -115.0
    assert float(target_lon.max().values) == -83.0
    assert float(np.diff(target_lat.values).min()) == 1.0
    assert float(np.diff(target_lon.values).min()) == 1.0


def test_build_fixed_target_grid_uses_configured_spacing() -> None:
    target_lat, target_lon = build_fixed_target_grid(WESTERN_MOSAIC_FIXED_TARGET_GRID)

    assert float(np.diff(target_lat.values).min()) == WESTERN_MOSAIC_FIXED_TARGET_GRID["lat_step"]
    assert float(np.diff(target_lon.values).min()) == WESTERN_MOSAIC_FIXED_TARGET_GRID["lon_step"]


def test_mosaic_regional_rasters_blends_scores_and_recomputes_categories() -> None:
    region_a = RegionDefinition("alpha", 0.0, 1.0, 0.0, 1.0, 0.0)
    region_b = RegionDefinition("beta", 0.0, 1.0, 0.0, 1.0, 0.0)
    lat_values = [0.0, 1.0]
    lon_values = [0.0, 1.0]
    raster_a = RegionalDailyRaster(region=region_a, daily=_daily_dataset(lat_values, lon_values, 60.0))
    raster_b = RegionalDailyRaster(region=region_b, daily=_daily_dataset(lat_values, lon_values, 80.0))

    merged, summary = mosaic_regional_rasters([raster_a, raster_b])

    assert float(merged["daily_score"].mean().values) == 70.0
    assert float(merged["category_index"].mean().values) == 2.0
    assert summary["pair_overlap_cell_count"] == 4
    assert summary["pair_mean_abs_score_diff"] == 20.0
    assert summary["pair_overlap_category_agreement_fraction"] == 0.0
    assert summary["pair_overlap_category_near_agreement_fraction"] == 1.0


def test_mosaic_regional_rasters_supports_equal_overlap_and_winner_take_all() -> None:
    region_a = RegionDefinition("alpha", 0.0, 1.0, 0.0, 1.0, 0.0)
    region_b = RegionDefinition("beta", 0.0, 1.0, 0.0, 1.0, 0.0)
    lat_values = [0.0, 1.0]
    lon_values = [0.0, 1.0]
    raster_a = RegionalDailyRaster(region=region_a, daily=_daily_dataset(lat_values, lon_values, 60.0))
    raster_b = RegionalDailyRaster(region=region_b, daily=_daily_dataset(lat_values, lon_values, 80.0))

    equal_overlap, _ = mosaic_regional_rasters([raster_a, raster_b], blend_method="equal_overlap")
    winner_take_all, summary = mosaic_regional_rasters([raster_a, raster_b], blend_method="winner_take_all")

    assert float(equal_overlap["daily_score"].mean().values) == 70.0
    assert float(winner_take_all["daily_score"].mean().values) == 60.0
    assert summary["blend_method"] == "winner_take_all"


def test_mosaic_regional_rasters_keeps_uncovered_union_cells_nan() -> None:
    raster_a = RegionalDailyRaster(
        region=RegionDefinition("alpha", 0.0, 1.0, 0.0, 1.0, 0.0),
        daily=_daily_dataset([0.0, 1.0], [0.0, 1.0], 65.0),
    )
    raster_b = RegionalDailyRaster(
        region=RegionDefinition("beta", 0.0, 1.0, 3.0, 4.0, 0.0),
        daily=_daily_dataset([0.0, 1.0], [3.0, 4.0], 75.0),
    )

    merged, summary = mosaic_regional_rasters([raster_a, raster_b])

    assert np.isnan(float(merged["daily_score"].sel(lat=0.0, lon=2.0).values))
    assert np.isnan(float(merged["category_index"].sel(lat=0.0, lon=2.0).values))
    assert summary["overlap_cell_count"] == 0


def test_mosaic_regional_rasters_reports_partial_overlap_diagnostics() -> None:
    region_a = RegionDefinition("southeast", 0.0, 2.0, 0.0, 2.0, 1.0)
    region_b = RegionDefinition("northeast", 1.0, 3.0, 1.0, 3.0, 1.0)
    lat_values = [0.0, 1.0, 2.0, 3.0]
    lon_values = [0.0, 1.0, 2.0, 3.0]
    daily_a = xr.Dataset(
        {"daily_score": (("lat", "lon"), np.array([
            [60.0, 62.0, 64.0, np.nan],
            [61.0, 63.0, 65.0, np.nan],
            [62.0, 64.0, 66.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]))},
        coords={"lat": lat_values, "lon": lon_values},
    )
    daily_b = xr.Dataset(
        {"daily_score": (("lat", "lon"), np.array([
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, 70.0, 72.0, 74.0],
            [np.nan, 71.0, 73.0, 75.0],
            [np.nan, 72.0, 74.0, 76.0],
        ]))},
        coords={"lat": lat_values, "lon": lon_values},
    )

    merged, summary = mosaic_regional_rasters(
        [
            RegionalDailyRaster(region=region_a, daily=daily_a),
            RegionalDailyRaster(region=region_b, daily=daily_b),
        ]
    )

    assert summary["pair_overlap_cell_count"] > 0
    assert summary["pair_mean_abs_score_diff"] > 0.0
    assert 0.0 <= summary["pair_overlap_category_agreement_fraction"] <= 1.0
    assert 0.0 <= summary["pair_overlap_category_near_agreement_fraction"] <= 1.0
    assert float(merged["blend_weight_sum"].max().values) > 1.0
    assert summary["pair_overlap_category_near_agreement_fraction"] >= summary["pair_overlap_category_agreement_fraction"]


def test_mosaic_regional_rasters_reports_pairwise_metrics_for_multi_region_stitched_run() -> None:
    lat_values = [0.0, 1.0, 2.0, 3.0]
    lon_values = [0.0, 1.0, 2.0, 3.0]
    rasters = [
        RegionalDailyRaster(
            region=RegionDefinition("southwest", 0.0, 1.5, 0.0, 1.5, 1.0),
            daily=xr.Dataset(
                {"daily_score": (("lat", "lon"), np.array([
                    [55.0, 57.0, 59.0, np.nan],
                    [58.0, 60.0, 62.0, np.nan],
                    [59.0, 61.0, 63.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ]))},
                coords={"lat": lat_values, "lon": lon_values},
            ),
        ),
        RegionalDailyRaster(
            region=RegionDefinition("rockies", 0.5, 2.5, 1.5, 3.0, 1.0),
            daily=xr.Dataset(
                {"daily_score": (("lat", "lon"), np.array([
                    [np.nan, 63.0, 65.0, 66.0],
                    [np.nan, 64.0, 66.0, 67.0],
                    [np.nan, 65.0, 67.0, 68.0],
                    [np.nan, np.nan, np.nan, np.nan],
                ]))},
                coords={"lat": lat_values, "lon": lon_values},
            ),
        ),
        RegionalDailyRaster(
            region=RegionDefinition("southeast", 1.5, 3.0, 0.0, 1.5, 1.0),
            daily=xr.Dataset(
                {"daily_score": (("lat", "lon"), np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [62.0, 64.0, np.nan, np.nan],
                    [63.0, 65.0, np.nan, np.nan],
                    [64.0, 66.0, np.nan, np.nan],
                ]))},
                coords={"lat": lat_values, "lon": lon_values},
            ),
        ),
    ]

    merged, summary = mosaic_regional_rasters(rasters)

    assert summary["pairwise_pair_count"] == 3
    assert summary["pairwise_pairs_with_overlap_count"] >= 1
    assert "pair_southwest_rockies_overlap_cell_count" in summary
    assert "pair_southwest_southeast_overlap_cell_count" in summary
    assert "pair_rockies_southeast_overlap_cell_count" in summary
    assert "pairwise_metrics_json" in summary
    assert float(merged["daily_score"].mean(skipna=True).values) > 0.0
