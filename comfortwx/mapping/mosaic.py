"""Regional mosaic helpers for future stitched products."""

from __future__ import annotations

from dataclasses import dataclass
import json
from itertools import combinations

import numpy as np
import xarray as xr

from comfortwx.config import (
    CATEGORIES,
    MOSAIC_CATEGORY_THRESHOLD_BUFFER,
    MOSAIC_DEFAULT_BLEND_METHOD,
    MOSAIC_DEFAULT_TARGET_GRID_POLICY,
    WESTERN_MOSAIC_FIXED_TARGET_GRID,
)
from comfortwx.mapping.regions import RegionDefinition, region_blend_weights, target_grid_alignment_reference
from comfortwx.scoring.categories import categorize_scores


@dataclass(frozen=True)
class RegionalDailyRaster:
    """A regional daily field plus its region definition."""

    region: RegionDefinition
    daily: xr.Dataset


def build_regional_weight_field(raster: RegionalDailyRaster) -> xr.DataArray:
    """Return per-gridcell blend weights for a regional daily field."""

    return region_blend_weights(raster.daily["lat"], raster.daily["lon"], raster.region)


def _infer_axis_step(values: np.ndarray) -> float:
    unique_values = np.unique(np.asarray(values, dtype=float))
    if unique_values.size <= 1:
        return 1.0
    diffs = np.diff(np.sort(unique_values))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return 1.0
    return float(np.round(positive_diffs.min(), 6))


def _coordinate_axis(start: float, stop: float, step: float, name: str) -> xr.DataArray:
    values = np.arange(start, stop + (step * 0.5), step, dtype=float)
    return xr.DataArray(np.round(values, 6), dims=(name,), coords={name: np.round(values, 6)})


def build_fixed_target_grid(grid_spec: dict[str, float]) -> tuple[xr.DataArray, xr.DataArray]:
    """Build a target grid from an explicit grid specification."""

    return (
        _coordinate_axis(grid_spec["lat_min"], grid_spec["lat_max"], grid_spec["lat_step"], "lat"),
        _coordinate_axis(grid_spec["lon_min"], grid_spec["lon_max"], grid_spec["lon_step"], "lon"),
    )


def build_common_target_grid(rasters: list[RegionalDailyRaster]) -> tuple[xr.DataArray, xr.DataArray]:
    """Build a shared target grid spanning the union of all regional daily rasters."""

    if not rasters:
        raise ValueError("At least one regional raster is required to define a target grid.")

    lat_min = min(float(raster.daily["lat"].min().values) for raster in rasters)
    lat_max = max(float(raster.daily["lat"].max().values) for raster in rasters)
    lon_min = min(float(raster.daily["lon"].min().values) for raster in rasters)
    lon_max = max(float(raster.daily["lon"].max().values) for raster in rasters)
    lat_step = min(_infer_axis_step(raster.daily["lat"].values) for raster in rasters)
    lon_step = min(_infer_axis_step(raster.daily["lon"].values) for raster in rasters)
    return (
        _coordinate_axis(lat_min, lat_max, lat_step, "lat"),
        _coordinate_axis(lon_min, lon_max, lon_step, "lon"),
    )


def _regrid_regional_raster(
    raster: RegionalDailyRaster,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Interpolate a regional daily field and its blend weights onto the target grid."""

    daily_score = raster.daily["daily_score"].interp(lat=target_lat, lon=target_lon, method="linear")
    weights = build_regional_weight_field(raster).interp(lat=target_lat, lon=target_lon, method="linear")
    effective_weights = weights.fillna(0.0).clip(0.0, 1.0) * daily_score.notnull()
    return daily_score, effective_weights


def _blend_weights_for_method(
    aligned_weights: dict[str, xr.DataArray],
    blend_method: str,
) -> dict[str, xr.DataArray]:
    """Return method-specific blend weights for already aligned regional weights."""

    normalized_method = blend_method.strip().lower()
    if normalized_method == "taper":
        return aligned_weights

    contributor_count: xr.DataArray | None = None
    for weight in aligned_weights.values():
        contributor_mask = (weight > 0).astype(int)
        contributor_count = contributor_mask if contributor_count is None else contributor_count + contributor_mask
    assert contributor_count is not None

    if normalized_method == "equal_overlap":
        return {
            region_name: xr.where(weight > 0, 1.0 / contributor_count.where(contributor_count > 0, other=1), 0.0)
            for region_name, weight in aligned_weights.items()
        }

    if normalized_method == "winner_take_all":
        stacked = xr.concat(list(aligned_weights.values()), dim="region")
        max_weight = stacked.max("region")
        region_names = list(aligned_weights.keys())
        method_weights: dict[str, xr.DataArray] = {}
        assigned_mask = xr.zeros_like(max_weight, dtype=bool)
        for region_name in region_names:
            weight = aligned_weights[region_name]
            is_winner = (weight == max_weight) & (weight > 0) & (~assigned_mask)
            method_weights[region_name] = is_winner.astype(float)
            assigned_mask = assigned_mask | is_winner
        return method_weights

    raise ValueError(f"Unsupported mosaic blend method '{blend_method}'.")


def _distance_to_category_thresholds(score: xr.DataArray) -> xr.DataArray:
    """Return minimum distance to a category threshold for each score."""

    thresholds = [category.lower for category in CATEGORIES[1:]]
    distance: xr.DataArray | None = None
    for threshold in thresholds:
        current = abs(score - threshold)
        distance = current if distance is None else xr.where(current < distance, current, distance)
    assert distance is not None
    return distance


def _pairwise_overlap_metrics(
    *,
    region_a: str,
    region_b: str,
    score_a: xr.DataArray,
    score_b: xr.DataArray,
    weight_a: xr.DataArray,
    weight_b: xr.DataArray,
    blended_score: xr.DataArray,
) -> dict[str, object]:
    """Return overlap diagnostics for a regional pair on the shared target grid."""

    pair_overlap = (weight_a > 0) & (weight_b > 0)
    overlap_cell_count = int(pair_overlap.sum().values)
    if overlap_cell_count == 0:
        return {
            "region_a": region_a,
            "region_b": region_b,
            "overlap_cell_count": 0,
            "mean_abs_score_diff": 0.0,
            "max_abs_score_diff": 0.0,
            "score_diff_p90": 0.0,
            "overlap_mean_blended_score": 0.0,
            "overlap_blended_score_p10": 0.0,
            "overlap_blended_score_p90": 0.0,
            "category_agreement_fraction": 0.0,
            "category_near_agreement_fraction": 0.0,
            "near_threshold_cell_count": 0,
        }

    pair_diff = np.abs(score_a - score_b).where(pair_overlap)
    category_a = categorize_scores(score_a.fillna(0.0)).where(pair_overlap)
    category_b = categorize_scores(score_b.fillna(0.0)).where(pair_overlap)
    category_delta = np.abs(category_a - category_b).where(pair_overlap)
    blended_overlap = blended_score.where(pair_overlap)
    overlap_threshold_distance = _distance_to_category_thresholds(blended_overlap).where(pair_overlap)
    exact_match_count = int(((category_delta == 0) & pair_overlap).sum().values)
    near_match_count = int(((category_delta <= 1) & pair_overlap).sum().values)
    return {
        "region_a": region_a,
        "region_b": region_b,
        "overlap_cell_count": overlap_cell_count,
        "mean_abs_score_diff": round(float(pair_diff.mean(skipna=True).fillna(0.0).values), 2),
        "max_abs_score_diff": round(float(pair_diff.max(skipna=True).fillna(0.0).values), 2),
        "score_diff_p90": round(float(pair_diff.quantile(0.9, skipna=True).fillna(0.0).values), 2),
        "overlap_mean_blended_score": round(float(blended_overlap.mean(skipna=True).fillna(0.0).values), 2),
        "overlap_blended_score_p10": round(float(blended_overlap.quantile(0.1, skipna=True).fillna(0.0).values), 2),
        "overlap_blended_score_p90": round(float(blended_overlap.quantile(0.9, skipna=True).fillna(0.0).values), 2),
        "category_agreement_fraction": round(float(exact_match_count / overlap_cell_count), 4),
        "category_near_agreement_fraction": round(float(near_match_count / overlap_cell_count), 4),
        "near_threshold_cell_count": int(((overlap_threshold_distance <= MOSAIC_CATEGORY_THRESHOLD_BUFFER) & pair_overlap).sum().values),
    }


def mosaic_regional_rasters(
    rasters: list[RegionalDailyRaster],
    target_lat: xr.DataArray | None = None,
    target_lon: xr.DataArray | None = None,
    blend_method: str = MOSAIC_DEFAULT_BLEND_METHOD,
    target_grid_policy: str = MOSAIC_DEFAULT_TARGET_GRID_POLICY,
) -> tuple[xr.Dataset, dict[str, object]]:
    """Blend regional daily rasters onto a shared target grid."""

    if not rasters:
        raise ValueError("At least one regional raster is required for mosaicking.")

    if target_lat is None or target_lon is None:
        if target_grid_policy == "adaptive":
            target_lat, target_lon = build_common_target_grid(rasters)
        elif target_grid_policy == "fixed_western":
            target_lat, target_lon = build_fixed_target_grid(WESTERN_MOSAIC_FIXED_TARGET_GRID)
        else:
            raise ValueError(f"Unsupported target grid policy '{target_grid_policy}'.")

    aligned_scores: dict[str, xr.DataArray] = {}
    aligned_weights: dict[str, xr.DataArray] = {}
    score_sum = xr.DataArray(
        np.zeros((target_lat.size, target_lon.size), dtype=float),
        dims=("lat", "lon"),
        coords={"lat": target_lat, "lon": target_lon},
    )
    weight_sum = xr.zeros_like(score_sum)
    contributor_count = xr.zeros_like(score_sum, dtype=int)

    for raster in rasters:
        aligned_score, aligned_weight = _regrid_regional_raster(raster, target_lat, target_lon)
        aligned_scores[raster.region.name] = aligned_score
        aligned_weights[raster.region.name] = aligned_weight
    method_weights = _blend_weights_for_method(aligned_weights, blend_method)
    for region_name, aligned_score in aligned_scores.items():
        aligned_weight = method_weights[region_name]
        score_sum = score_sum + aligned_score.fillna(0.0) * aligned_weight
        weight_sum = weight_sum + aligned_weight
        contributor_count = contributor_count + (aligned_weight > 0).astype(int)

    blended_score = xr.where(weight_sum > 0, score_sum / weight_sum, np.nan).clip(0.0, 100.0)
    overlap_mask = contributor_count > 1
    category_index = categorize_scores(blended_score.fillna(0.0)).where(weight_sum > 0)

    merged = xr.Dataset(
        {
            "daily_score": blended_score,
            "category_index": category_index,
            "blend_weight_sum": weight_sum,
            "contributor_count": contributor_count,
            "overlap_mask": overlap_mask.astype(int),
        }
    )
    for region_name, score in aligned_scores.items():
        merged[f"source_score_{region_name}"] = score
    for region_name, weight in method_weights.items():
        merged[f"source_weight_{region_name}"] = weight

    merged.attrs["mosaic_method"] = blend_method
    merged.attrs["target_grid_policy"] = target_grid_policy
    merged.attrs["participating_regions"] = ",".join(raster.region.name for raster in rasters)
    merged.attrs["todo"] = "Future production mosaics should add explicit native-grid regridding and broader multi-region seam diagnostics."

    category_counts = {
        "poor_count": int((category_index == 0).sum().values),
        "fair_count": int((category_index == 1).sum().values),
        "pleasant_count": int((category_index == 2).sum().values),
        "ideal_count": int((category_index == 3).sum().values),
        "exceptional_count": int((category_index == 4).sum().values),
    }

    overlap_cell_count = int(overlap_mask.sum().values)
    covered_cell_count = int((weight_sum > 0).sum().values)
    summary: dict[str, object] = {
        "participating_regions": ",".join(raster.region.name for raster in rasters),
        "target_grid_shape": f"{target_lat.size}x{target_lon.size}",
        "mean_daily_score": round(float(blended_score.mean(skipna=True).values), 2),
        "min_daily_score": round(float(blended_score.min(skipna=True).values), 2),
        "max_daily_score": round(float(blended_score.max(skipna=True).values), 2),
        "blend_method": blend_method,
        "target_grid_policy": target_grid_policy,
        "covered_cell_count": covered_cell_count,
        "overlap_cell_count": overlap_cell_count,
        "overlap_fraction_of_covered": round(float(overlap_cell_count / covered_cell_count), 4) if covered_cell_count else 0.0,
        **category_counts,
        **target_grid_alignment_reference(merged),
    }

    pairwise_metrics = [
        _pairwise_overlap_metrics(
            region_a=region_a,
            region_b=region_b,
            score_a=aligned_scores[region_a],
            score_b=aligned_scores[region_b],
            weight_a=aligned_weights[region_a],
            weight_b=aligned_weights[region_b],
            blended_score=blended_score,
        )
        for region_a, region_b in combinations(aligned_scores.keys(), 2)
    ]
    summary["pairwise_pair_count"] = len(pairwise_metrics)
    summary["pairwise_pairs_with_overlap_count"] = sum(1 for metrics in pairwise_metrics if int(metrics["overlap_cell_count"]) > 0)
    summary["pairwise_metrics_json"] = json.dumps(pairwise_metrics)
    for metrics in pairwise_metrics:
        pair_prefix = f"pair_{metrics['region_a']}_{metrics['region_b']}"
        summary[f"{pair_prefix}_overlap_cell_count"] = int(metrics["overlap_cell_count"])
        summary[f"{pair_prefix}_mean_abs_score_diff"] = float(metrics["mean_abs_score_diff"])
        summary[f"{pair_prefix}_max_abs_score_diff"] = float(metrics["max_abs_score_diff"])
        summary[f"{pair_prefix}_category_agreement_fraction"] = float(metrics["category_agreement_fraction"])
        summary[f"{pair_prefix}_category_near_agreement_fraction"] = float(metrics["category_near_agreement_fraction"])

    if len(pairwise_metrics) == 1:
        pair_metrics = pairwise_metrics[0]
        summary["pair_overlap_cell_count"] = int(pair_metrics["overlap_cell_count"])
        summary["pair_mean_abs_score_diff"] = float(pair_metrics["mean_abs_score_diff"])
        summary["pair_max_abs_score_diff"] = float(pair_metrics["max_abs_score_diff"])
        summary["pair_score_diff_p90"] = float(pair_metrics["score_diff_p90"])
        summary["pair_overlap_mean_blended_score"] = float(pair_metrics["overlap_mean_blended_score"])
        summary["pair_overlap_blended_score_p10"] = float(pair_metrics["overlap_blended_score_p10"])
        summary["pair_overlap_blended_score_p90"] = float(pair_metrics["overlap_blended_score_p90"])
        summary["pair_overlap_category_agreement_fraction"] = float(pair_metrics["category_agreement_fraction"])
        summary["pair_overlap_category_near_agreement_fraction"] = float(pair_metrics["category_near_agreement_fraction"])
        summary["pair_overlap_near_threshold_cell_count"] = int(pair_metrics["near_threshold_cell_count"])

    return merged, summary


def weighted_overlap_merge(rasters: list[RegionalDailyRaster]) -> xr.Dataset:
    """Basic weighted merge for overlapping regional daily rasters.

    This assumes the regions already share a compatible target grid, which is true
    for the current mock-grid regional processing path. Future real-data mosaics
    can reuse this interface after an explicit regridding step.
    """

    if not rasters:
        raise ValueError("At least one regional raster is required for mosaicking.")

    score_sum: xr.DataArray | None = None
    weight_sum: xr.DataArray | None = None
    for raster in rasters:
        daily_score = raster.daily["daily_score"]
        weights = build_regional_weight_field(raster)
        weighted_score = daily_score.fillna(0.0) * weights
        if score_sum is None or weight_sum is None:
            score_sum = weighted_score
            weight_sum = weights
            continue
        score_sum, weighted_score = xr.align(score_sum, weighted_score, join="outer", fill_value=0.0)
        weight_sum, weights = xr.align(weight_sum, weights, join="outer", fill_value=0.0)
        score_sum = score_sum + weighted_score
        weight_sum = weight_sum + weights

    assert score_sum is not None and weight_sum is not None
    merged_score = xr.where(weight_sum > 0, score_sum / weight_sum, np.nan).fillna(0.0).clip(0.0, 100.0)
    merged = xr.Dataset({"daily_score": merged_score, "category_index": categorize_scores(merged_score)})
    merged.attrs["mosaic_method"] = "weighted_overlap_merge"
    return merged
