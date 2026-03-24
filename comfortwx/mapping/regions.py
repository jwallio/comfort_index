"""Regional domain definitions and overlap helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class RegionDefinition:
    """A core regional domain plus overlap metadata."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    overlap_buffer: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def core_bounds(self) -> tuple[float, float, float, float]:
        return (self.lon_min, self.lon_max, self.lat_min, self.lat_max)

    @property
    def expanded_bounds(self) -> tuple[float, float, float, float]:
        return (
            self.lon_min - self.overlap_buffer,
            self.lon_max + self.overlap_buffer,
            self.lat_min - self.overlap_buffer,
            self.lat_max + self.overlap_buffer,
        )


REGIONS: tuple[RegionDefinition, ...] = (
    RegionDefinition("west_coast", 31.0, 49.0, -125.0, -116.0, 2.0, {"group": "west"}),
    RegionDefinition("southwest", 31.0, 41.5, -118.0, -107.0, 2.5, {"group": "west"}),
    RegionDefinition("rockies", 37.0, 49.0, -113.5, -102.0, 2.0, {"group": "interior"}),
    RegionDefinition("plains", 25.0, 49.0, -104.5, -92.0, 2.0, {"group": "central"}),
    RegionDefinition("great_lakes", 40.0, 49.5, -94.0, -80.0, 2.0, {"group": "north"}),
    RegionDefinition("southeast", 24.0, 37.5, -92.0, -75.0, 2.0, {"group": "south"}),
    RegionDefinition("northeast", 37.0, 47.5, -82.0, -66.5, 2.0, {"group": "east"}),
)


def list_region_names() -> list[str]:
    """Return supported region names."""

    return [region.name for region in REGIONS]


def get_region_definition(name: str) -> RegionDefinition:
    """Return a named region definition."""

    normalized = name.strip().lower()
    for region in REGIONS:
        if region.name == normalized:
            return region
    raise ValueError(f"Unknown region '{name}'. Available regions: {', '.join(list_region_names())}")


def subset_dataset_to_region(dataset: xr.Dataset, region: RegionDefinition, include_overlap: bool = True) -> xr.Dataset:
    """Subset a gridded dataset to a region's core or expanded extent."""

    lon_min, lon_max, lat_min, lat_max = region.expanded_bounds if include_overlap else region.core_bounds
    lat_mask = (dataset["lat"] >= lat_min) & (dataset["lat"] <= lat_max)
    lon_mask = (dataset["lon"] >= lon_min) & (dataset["lon"] <= lon_max)
    subset = dataset.where(lat_mask & lon_mask, drop=True)
    subset.attrs.update(
        {
            "region_name": region.name,
            "region_core_bounds": region.core_bounds,
            "region_expanded_bounds": region.expanded_bounds,
        }
    )
    return subset


def region_overlap_mask(lat: xr.DataArray, lon: xr.DataArray, region: RegionDefinition) -> xr.DataArray:
    """Return a mask for the overlap ring around the core region."""

    lat2d, lon2d = xr.broadcast(lat, lon)
    in_expanded = (
        (lat2d >= region.expanded_bounds[2])
        & (lat2d <= region.expanded_bounds[3])
        & (lon2d >= region.expanded_bounds[0])
        & (lon2d <= region.expanded_bounds[1])
    )
    in_core = (
        (lat2d >= region.lat_min)
        & (lat2d <= region.lat_max)
        & (lon2d >= region.lon_min)
        & (lon2d <= region.lon_max)
    )
    return in_expanded & (~in_core)


def region_blend_weights(lat: xr.DataArray, lon: xr.DataArray, region: RegionDefinition) -> xr.DataArray:
    """Return tapering blend weights from region core to expanded edge."""

    lat2d, lon2d = xr.broadcast(lat, lon)
    core_distance = xr.full_like(lat2d, fill_value=1.0, dtype=float)

    if region.overlap_buffer <= 0:
        return xr.where(
            (lat2d >= region.lat_min) & (lat2d <= region.lat_max) & (lon2d >= region.lon_min) & (lon2d <= region.lon_max),
            1.0,
            0.0,
        )

    left = ((lon2d - (region.lon_min - region.overlap_buffer)) / region.overlap_buffer).clip(0.0, 1.0)
    right = (((region.lon_max + region.overlap_buffer) - lon2d) / region.overlap_buffer).clip(0.0, 1.0)
    bottom = ((lat2d - (region.lat_min - region.overlap_buffer)) / region.overlap_buffer).clip(0.0, 1.0)
    top = (((region.lat_max + region.overlap_buffer) - lat2d) / region.overlap_buffer).clip(0.0, 1.0)

    in_expanded = (
        (lat2d >= region.expanded_bounds[2])
        & (lat2d <= region.expanded_bounds[3])
        & (lon2d >= region.expanded_bounds[0])
        & (lon2d <= region.expanded_bounds[1])
    )
    edge_stack = xr.concat([left, right, bottom, top], dim=pd.Index(["left", "right", "bottom", "top"], name="edge"))
    weight = xr.where(in_expanded, edge_stack.min("edge"), 0.0)
    in_core = (
        (lat2d >= region.lat_min)
        & (lat2d <= region.lat_max)
        & (lon2d >= region.lon_min)
        & (lon2d <= region.lon_max)
    )
    weight = xr.where(in_core, core_distance, weight)
    return weight.clip(0.0, 1.0)


def target_grid_alignment_reference(dataset: xr.Dataset) -> dict[str, object]:
    """Return lightweight metadata for future common-grid alignment."""

    lat = dataset["lat"].values
    lon = dataset["lon"].values
    return {
        "lat_size": int(len(lat)),
        "lon_size": int(len(lon)),
        "lat_step": float(np.round(np.diff(lat).mean(), 6)) if len(lat) > 1 else None,
        "lon_step": float(np.round(np.diff(lon).mean(), 6)) if len(lon) > 1 else None,
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
    }


def regional_summary_record(region_daily: xr.Dataset, region: RegionDefinition) -> dict[str, object]:
    """Return a compact summary record for a processed region."""

    category_counts = {
        "poor_count": int((region_daily["category_index"] == 0).sum().values),
        "fair_count": int((region_daily["category_index"] == 1).sum().values),
        "pleasant_count": int((region_daily["category_index"] == 2).sum().values),
        "ideal_count": int((region_daily["category_index"] == 3).sum().values),
        "exceptional_count": int((region_daily["category_index"] == 4).sum().values),
    }
    overlap_fraction = (
        float(region_daily["overlap_mask"].mean().values) if "overlap_mask" in region_daily else float(region_overlap_mask(region_daily["lat"], region_daily["lon"], region).mean().values)
    )
    return {
        "region_name": region.name,
        "core_bounds": region.core_bounds,
        "expanded_bounds": region.expanded_bounds,
        "mean_daily_score": round(float(region_daily["daily_score"].mean().values), 2),
        "min_daily_score": round(float(region_daily["daily_score"].min().values), 2),
        "max_daily_score": round(float(region_daily["daily_score"].max().values), 2),
        "grid_points": int(region_daily["daily_score"].size),
        "overlap_fraction": round(overlap_fraction, 4),
        **category_counts,
        **target_grid_alignment_reference(region_daily),
    }
