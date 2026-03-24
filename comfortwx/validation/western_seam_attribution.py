"""Diagnose which score components drive the southwest + rockies seam roughness."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from comfortwx.config import (
    DAYTIME_HOUR_WEIGHTS,
    LOCAL_DAY_HOURS,
    MOSAIC_CATEGORY_THRESHOLD_BUFFER,
    OUTPUT_DIR,
)
from comfortwx.main import _build_regional_daily
from comfortwx.mapping.mosaic import RegionalDailyRaster, build_common_target_grid, build_regional_weight_field, mosaic_regional_rasters
from comfortwx.scoring.categories import category_name_from_index

WESTERN_SEAM_PAIR: tuple[str, str] = ("southwest", "rockies")
ATTRIBUTION_COMPONENTS: tuple[str, ...] = (
    "temp_score",
    "dewpoint_score",
    "wind_score",
    "cloud_score",
    "precip_score",
    "hazard_penalty",
    "interaction_adjustment",
    "reliability_score",
    "disruption_penalty",
    "daily_score",
    "category_index",
)
DOMINANT_DRIVER_COMPONENTS: tuple[str, ...] = (
    "temp_score",
    "dewpoint_score",
    "wind_score",
    "cloud_score",
    "precip_score",
    "hazard_penalty",
    "interaction_adjustment",
    "reliability_score",
    "disruption_penalty",
)
COMPONENT_GROUPS: dict[str, str] = {
    "temp_score": "thermodynamic",
    "dewpoint_score": "thermodynamic",
    "wind_score": "thermodynamic",
    "interaction_adjustment": "thermodynamic",
    "cloud_score": "cloud/precip",
    "precip_score": "cloud/precip",
    "hazard_penalty": "cloud/precip",
    "reliability_score": "aggregation/reliability",
    "disruption_penalty": "aggregation/reliability",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run overlap-zone seam attribution for southwest + rockies.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for attribution outputs.")
    parser.add_argument("--mesh-profile", default="standard", help="Open-Meteo mesh profile. Default: standard.")
    parser.add_argument("--aggregation-mode", default="baseline", help="Daily aggregation mode. Default: baseline.")
    return parser.parse_args()


def _daytime_subset(scored_hourly: xr.Dataset) -> tuple[xr.Dataset, xr.DataArray]:
    start_hour, end_hour = LOCAL_DAY_HOURS
    hour_values = scored_hourly["time"].dt.hour
    subset = scored_hourly.where((hour_values >= start_hour) & (hour_values <= end_hour), drop=True)
    weights = xr.DataArray(
        np.array([DAYTIME_HOUR_WEIGHTS[int(hour)] for hour in subset["time"].dt.hour.values], dtype=float),
        dims=("time",),
        coords={"time": subset["time"]},
    )
    return subset, weights


def _aggregate_component_daily(scored_hourly: xr.Dataset, daily: xr.Dataset) -> xr.Dataset:
    subset, weights = _daytime_subset(scored_hourly)
    aggregated = xr.Dataset(coords={"lat": scored_hourly["lat"], "lon": scored_hourly["lon"]})
    for component in (
        "temp_score",
        "dewpoint_score",
        "wind_score",
        "cloud_score",
        "precip_score",
        "hazard_penalty",
        "interaction_adjustment",
    ):
        aggregated[component] = subset[component].weighted(weights).mean("time")
    aggregated["daily_score"] = daily["daily_score"]
    aggregated["category_index"] = daily["category_index"]
    aggregated["reliability_score"] = daily["reliability_score"]
    aggregated["disruption_penalty"] = daily["disruption_penalty"]
    return aggregated


def _distance_to_thresholds(score: xr.DataArray) -> xr.DataArray:
    thresholds = [45.0, 60.0, 75.0, 90.0]
    distance: xr.DataArray | None = None
    for threshold in thresholds:
        current = abs(score - threshold)
        distance = current if distance is None else xr.where(current < distance, current, distance)
    assert distance is not None
    return distance


def _build_overlap_detail_frame(
    southwest_daily_components: xr.Dataset,
    rockies_daily_components: xr.Dataset,
    southwest_region,
    rockies_region,
) -> pd.DataFrame:
    target_lat, target_lon = build_common_target_grid(
        [
            RegionalDailyRaster(region=southwest_region, daily=southwest_daily_components),
            RegionalDailyRaster(region=rockies_region, daily=rockies_daily_components),
        ]
    )
    southwest_raster = RegionalDailyRaster(region=southwest_region, daily=southwest_daily_components)
    rockies_raster = RegionalDailyRaster(region=rockies_region, daily=rockies_daily_components)
    southwest_weights = build_regional_weight_field(southwest_raster).interp(lat=target_lat, lon=target_lon, method="linear").fillna(0.0)
    rockies_weights = build_regional_weight_field(rockies_raster).interp(lat=target_lat, lon=target_lon, method="linear").fillna(0.0)
    overlap_mask = (southwest_weights > 0) & (rockies_weights > 0)

    records: list[dict[str, object]] = []
    for lat in target_lat.values:
        for lon in target_lon.values:
            if not bool(overlap_mask.sel(lat=lat, lon=lon).values):
                continue
            row: dict[str, object] = {
                "lat": float(lat),
                "lon": float(lon),
                "southwest_weight": round(float(southwest_weights.sel(lat=lat, lon=lon).values), 4),
                "rockies_weight": round(float(rockies_weights.sel(lat=lat, lon=lon).values), 4),
            }
            dominant_components: dict[str, float] = {}
            for component in ATTRIBUTION_COMPONENTS:
                southwest_value = float(southwest_daily_components[component].interp(lat=[lat], lon=[lon], method="linear" if component != "category_index" else "nearest").values.squeeze())
                rockies_value = float(rockies_daily_components[component].interp(lat=[lat], lon=[lon], method="linear" if component != "category_index" else "nearest").values.squeeze())
                row[f"southwest_{component}"] = round(southwest_value, 3)
                row[f"rockies_{component}"] = round(rockies_value, 3)
                abs_diff = abs(rockies_value - southwest_value)
                row[f"{component}_abs_diff"] = round(abs_diff, 3)
                if component in DOMINANT_DRIVER_COMPONENTS:
                    dominant_components[component] = abs_diff
            dominant_component = max(dominant_components, key=dominant_components.get)
            row["dominant_component"] = dominant_component
            row["dominant_component_group"] = COMPONENT_GROUPS[dominant_component]
            records.append(row)

    detail = pd.DataFrame(records)
    return detail


def summarize_overlap_attribution(detail_frame: pd.DataFrame) -> dict[str, object]:
    """Return a compact overlap-zone attribution summary."""

    dominant_counts = Counter(detail_frame["dominant_component"])
    dominant_driver, dominant_count = dominant_counts.most_common(1)[0]
    secondary_driver, secondary_count = dominant_counts.most_common(2)[1] if len(dominant_counts) > 1 else (dominant_driver, dominant_count)
    driver_group = COMPONENT_GROUPS[dominant_driver]

    summary: dict[str, object] = {
        "overlap_cell_count": int(len(detail_frame)),
        "dominant_driver": dominant_driver,
        "dominant_driver_fraction": round(float(dominant_count / len(detail_frame)), 4),
        "secondary_driver": secondary_driver,
        "secondary_driver_fraction": round(float(secondary_count / len(detail_frame)), 4),
        "driver_group": driver_group,
    }
    for component in ATTRIBUTION_COMPONENTS:
        summary[f"southwest_{component}_mean_overlap"] = round(float(detail_frame[f"southwest_{component}"].mean()), 3)
        summary[f"rockies_{component}_mean_overlap"] = round(float(detail_frame[f"rockies_{component}"].mean()), 3)
        summary[f"{component}_mean_abs_diff"] = round(float(detail_frame[f"{component}_abs_diff"].mean()), 3)
        summary[f"{component}_max_abs_diff"] = round(float(detail_frame[f"{component}_abs_diff"].max()), 3)

    return summary


def run_western_seam_attribution(
    valid_date: date,
    output_dir: Path,
    mesh_profile: str = "standard",
    aggregation_mode: str = "baseline",
) -> tuple[Path, Path]:
    """Run overlap-zone attribution for southwest + rockies and write summary/detail CSVs."""

    southwest_scored, southwest_daily, southwest_region = _build_regional_daily(
        valid_date=valid_date,
        loader_name="openmeteo",
        lat_points=65,
        lon_points=115,
        region_name="southwest",
        mesh_profile=mesh_profile,
        aggregation_mode=aggregation_mode,
    )
    rockies_scored, rockies_daily, rockies_region = _build_regional_daily(
        valid_date=valid_date,
        loader_name="openmeteo",
        lat_points=65,
        lon_points=115,
        region_name="rockies",
        mesh_profile=mesh_profile,
        aggregation_mode=aggregation_mode,
    )
    southwest_components = _aggregate_component_daily(southwest_scored, southwest_daily)
    rockies_components = _aggregate_component_daily(rockies_scored, rockies_daily)

    detail_frame = _build_overlap_detail_frame(
        southwest_daily_components=southwest_components,
        rockies_daily_components=rockies_components,
        southwest_region=southwest_region,
        rockies_region=rockies_region,
    )

    mosaic_daily, _ = mosaic_regional_rasters(
        [
            RegionalDailyRaster(region=southwest_region, daily=southwest_daily),
            RegionalDailyRaster(region=rockies_region, daily=rockies_daily),
        ]
    )
    overlap_daily = detail_frame[["lat", "lon"]].copy()
    overlap_daily["blended_daily_score"] = [
        float(mosaic_daily["daily_score"].sel(lat=row.lat, lon=row.lon, method="nearest").values)
        for row in overlap_daily.itertuples(index=False)
    ]
    overlap_daily["blended_category_index"] = [
        int(float(mosaic_daily["category_index"].sel(lat=row.lat, lon=row.lon, method="nearest").values))
        for row in overlap_daily.itertuples(index=False)
    ]
    overlap_daily["near_category_threshold"] = [
        bool(
            float(
                _distance_to_thresholds(mosaic_daily["daily_score"]).sel(lat=row.lat, lon=row.lon, method="nearest").values
            )
            <= MOSAIC_CATEGORY_THRESHOLD_BUFFER
        )
        for row in overlap_daily.itertuples(index=False)
    ]
    detail_frame = detail_frame.merge(overlap_daily, on=["lat", "lon"], how="left")
    detail_frame["blended_category"] = detail_frame["blended_category_index"].apply(category_name_from_index)

    summary = summarize_overlap_attribution(detail_frame)
    near_threshold_fraction = float(detail_frame["near_category_threshold"].mean()) if not detail_frame.empty else 0.0
    summary["mesh_profile"] = mesh_profile
    summary["aggregation_mode"] = aggregation_mode
    summary["seam_pair"] = "+".join(WESTERN_SEAM_PAIR)
    summary["near_threshold_cell_count"] = int(detail_frame["near_category_threshold"].sum())
    summary["near_threshold_fraction"] = round(near_threshold_fraction, 4)
    summary["pair_overlap_category_agreement_fraction"] = round(float((detail_frame["category_index_abs_diff"] == 0).mean()), 4)
    summary["pair_overlap_category_near_agreement_fraction"] = round(float((detail_frame["category_index_abs_diff"] <= 1).mean()), 4)

    aggregation_suffix = "" if aggregation_mode == "baseline" else f"_{aggregation_mode}"
    summary_path = output_dir / f"comfortwx_western_seam_attribution_{mesh_profile}{aggregation_suffix}_{valid_date:%Y%m%d}.csv"
    detail_path = output_dir / f"comfortwx_western_seam_attribution_overlap_cells_{mesh_profile}{aggregation_suffix}_{valid_date:%Y%m%d}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    detail_frame.to_csv(detail_path, index=False)
    return summary_path, detail_path


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    summary_path, detail_path = run_western_seam_attribution(
        valid_date=valid_date,
        output_dir=Path(args.output_dir),
        mesh_profile=args.mesh_profile,
        aggregation_mode=args.aggregation_mode,
    )
    summary = pd.read_csv(summary_path).iloc[0]
    print(f"Valid date: {valid_date:%Y-%m-%d}")
    print(f"Dominant seam driver: {summary['dominant_driver']}")
    print(f"Secondary seam driver: {summary['secondary_driver']}")
    print(f"Near-threshold fraction: {summary['near_threshold_fraction']:.4f}")
    print(f"Driver group: {summary['driver_group']}")
    print(f"Saved seam attribution summary: {summary_path}")
    print(f"Saved seam attribution overlap detail: {detail_path}")


if __name__ == "__main__":
    main()
