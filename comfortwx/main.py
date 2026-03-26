"""CLI entry point for the Comfort Index Map Generator."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from comfortwx.config import (
    ARCHIVE_SETTINGS,
    OUTPUT_DIR,
    PILOT_DAY_CACHE_DEFAULT_MODE,
    PILOT_DAY_CACHE_MODES,
    PILOT_DAY_MOSAICS,
    PILOT_DAY_REGIONS,
    PUBLIC_CITY_RANKING_LOCATIONS,
    STITCHED_CONUS_PRESENTATION,
)
from comfortwx.data.loaders import get_loader
from comfortwx.data.openmeteo_reliability import (
    openmeteo_request_context,
    reset_openmeteo_request_records,
    write_openmeteo_request_report,
)
from comfortwx.mapping.mosaic import RegionalDailyRaster, mosaic_regional_rasters
from comfortwx.mapping.regions import (
    RegionDefinition,
    get_region_definition,
    list_region_names,
    region_blend_weights,
    region_overlap_mask,
    regional_summary_record,
    subset_dataset_to_region,
)
from comfortwx.mapping.plotting import render_daily_maps
from comfortwx.publishing import (
    build_archive_run_directory,
    resolve_publish_preset,
    write_archive_index,
    write_pilot_day_index,
    write_pilot_day_status_summary,
    write_publish_bundle,
)
from comfortwx.scoring.categories import category_name_from_index, category_name_from_value
from comfortwx.scoring.daily import aggregate_daily_scores
from comfortwx.scoring.hourly import score_hourly_dataset
from comfortwx.validation.demo_cases import build_demo_case_hourly_breakdown, run_demo_case_validation
from comfortwx.validation.inspection import export_point_inspection, inspect_point


def _run_slug(*parts: object) -> str:
    return "_".join(
        str(part).strip().lower().replace(" ", "_").replace("+", "_").replace("/", "_")
        for part in parts
        if str(part).strip()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a daily comfort map.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--source", default="mock", help="Data source. Supports 'mock' and 'openmeteo'.")
    parser.add_argument("--loader", default="mock", help="Loader name. V1 supports 'mock'.")
    parser.add_argument("--lat", type=float, default=None, help="Point latitude for real-data point mode.")
    parser.add_argument("--lon", type=float, default=None, help="Point longitude for real-data point mode.")
    parser.add_argument("--region", default=None, help=f"Optional regional mode. Available: {', '.join(list_region_names())}")
    parser.add_argument("--mosaic", nargs="+", default=None, help="Optional regional mosaic test. Example: --mosaic southeast southwest")
    parser.add_argument("--mesh-profile", default="standard", help="Optional Open-Meteo regional mesh profile. Default: standard.")
    parser.add_argument("--mosaic-blend-method", default="taper", help="Optional mosaic blend method. Default: taper.")
    parser.add_argument("--mosaic-target-grid", default="adaptive", help="Optional mosaic target-grid policy. Default: adaptive.")
    parser.add_argument("--aggregation-mode", default="baseline", help="Optional daily aggregation mode. Default: baseline.")
    parser.add_argument("--publish-preset", default=None, help="Optional publish bundle preset. Example: --publish-preset standard")
    parser.add_argument("--presentation-theme", default="default", help="Presentation theme for polished maps. Default: default.")
    parser.add_argument("--pilot-day", action="store_true", help="Run all currently supported real pilot regions and seam mosaics for one date.")
    parser.add_argument("--pilot-day-archive", action="store_true", help="Run pilot-day mode into a dated archive folder and refresh the archive landing page.")
    parser.add_argument(
        "--pilot-span-days",
        type=int,
        default=1,
        help="Number of consecutive valid dates for pilot-day or pilot-day-archive workflows. Default: 1.",
    )
    parser.add_argument(
        "--pilot-cache-mode",
        default=PILOT_DAY_CACHE_DEFAULT_MODE,
        choices=PILOT_DAY_CACHE_MODES,
        help="Pilot-day cache behavior. 'reuse' reuses existing regional daily fields when present; 'refresh' refetches regions. Default: reuse.",
    )
    parser.add_argument("--archive-root", default=None, help="Optional archive root. Default: <output-dir>/archive")
    parser.add_argument("--archive-layout", default=ARCHIVE_SETTINGS["layout"], help="Archive layout. Default: year/month/day")
    parser.add_argument("--lat-points", type=int, default=65, help="Latitude grid size for the mock loader.")
    parser.add_argument("--lon-points", type=int, default=115, help="Longitude grid size for the mock loader.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for maps and daily field output.")
    parser.add_argument("--inspect-lat", type=float, default=None, help="Optional latitude for nearest-grid inspection.")
    parser.add_argument("--inspect-lon", type=float, default=None, help="Optional longitude for nearest-grid inspection.")
    return parser.parse_args()


def _save_daily_fields(daily: xr.Dataset, valid_date: date, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    netcdf_path = output_dir / f"comfortwx_daily_fields_{valid_date:%Y%m%d}.nc"
    daily.to_netcdf(netcdf_path)
    return netcdf_path


def _grid_summary(daily: xr.Dataset) -> str:
    mean_score = float(daily["daily_score"].mean().values)
    min_score = float(daily["daily_score"].min().values)
    max_score = float(daily["daily_score"].max().values)
    stacked = daily["daily_score"].stack(points=("lat", "lon"))
    best_point = stacked.isel(points=int(stacked.argmax("points").values))
    best_lat = float(best_point["lat"].values)
    best_lon = float(best_point["lon"].values)
    category_stack = daily["category_index"].stack(points=("lat", "lon"))
    best_category = category_name_from_index(int(category_stack.isel(points=int(stacked.argmax("points").values)).values))
    return (
        f"Grid summary: mean={mean_score:.1f}, min={min_score:.1f}, max={max_score:.1f} "
        f"at ({best_lat:.2f}, {best_lon:.2f}) [{best_category}]"
    )


def _point_summary(daily: xr.Dataset) -> str:
    daily_point = daily.isel(lat=0, lon=0)
    category = category_name_from_index(int(daily_point["category_index"].values))
    return (
        f"Point summary: daily_score={float(daily_point['daily_score'].values):.1f}, "
        f"category={category}, best_6hr={float(daily_point['best_6hr'].values):.1f}, "
        f"reliability={float(daily_point['reliability_score'].values):.1f}, "
        f"disruption={float(daily_point['disruption_penalty'].values):.1f}"
    )


def _build_regional_daily(
    valid_date: date,
    loader_name: str,
    lat_points: int,
    lon_points: int,
    region_name: str,
    mesh_profile: str = "standard",
    aggregation_mode: str = "baseline",
) -> tuple[xr.Dataset, xr.Dataset, RegionDefinition]:
    region = get_region_definition(region_name)
    loader = get_loader(
        loader_name=loader_name,
        lat_points=lat_points,
        lon_points=lon_points,
        region_name=region_name if loader_name == "openmeteo" else None,
        mesh_profile=mesh_profile,
    )
    hourly = loader.load_hourly_grid(valid_date=valid_date)
    if loader_name == "mock":
        hourly = subset_dataset_to_region(hourly, region, include_overlap=True)
    elif loader_name == "openmeteo" and region.name not in {"west_coast", "southeast", "southwest", "rockies", "plains", "great_lakes", "northeast"}:
        raise ValueError("Pilot real regional mesh ingestion currently supports only the west_coast, southwest, rockies, plains, great_lakes, southeast, and northeast regions.")

    scored_hourly = score_hourly_dataset(hourly)
    regional_daily = aggregate_daily_scores(scored_hourly, aggregation_mode=aggregation_mode).copy()
    regional_daily["blend_weight"] = region_blend_weights(regional_daily["lat"], regional_daily["lon"], region)
    regional_daily["overlap_mask"] = region_overlap_mask(regional_daily["lat"], regional_daily["lon"], region).astype(int)
    regional_daily.attrs["region_name"] = region.name
    regional_daily.attrs["region_core_bounds"] = region.core_bounds
    regional_daily.attrs["region_expanded_bounds"] = region.expanded_bounds
    regional_daily.attrs["mesh_profile"] = mesh_profile
    regional_daily.attrs["aggregation_mode"] = aggregation_mode
    return scored_hourly, regional_daily, region


def _region_file_prefix(loader_name: str, region_name: str, mesh_profile: str, aggregation_mode: str) -> str:
    profile_suffix = "" if mesh_profile == "standard" else f"_{mesh_profile}"
    aggregation_suffix = "" if aggregation_mode == "baseline" else f"_{aggregation_mode}"
    return f"comfortwx_region_{region_name}" if loader_name == "mock" else f"comfortwx_region_{region_name}_openmeteo{profile_suffix}{aggregation_suffix}"


def _mosaic_file_prefix(
    region_names: list[str],
    loader_name: str,
    mesh_profile: str,
    aggregation_mode: str,
    mosaic_blend_method: str,
    mosaic_target_grid: str,
) -> str:
    ordered_regions = "_".join(region_names)
    profile_suffix = "" if mesh_profile == "standard" else f"_{mesh_profile}"
    method_suffix = "" if mosaic_blend_method == "taper" and mosaic_target_grid == "adaptive" else f"_{mosaic_blend_method}_{mosaic_target_grid}"
    aggregation_suffix = "" if aggregation_mode == "baseline" else f"_{aggregation_mode}"
    if loader_name == "mock":
        return f"comfortwx_mosaic_{ordered_regions}{aggregation_suffix}{method_suffix}"
    return f"comfortwx_mosaic_{ordered_regions}_openmeteo{profile_suffix}{aggregation_suffix}{method_suffix}"


def _load_daily_dataset(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as dataset:
        return dataset.load()


def _cached_region_daily_matches_request(
    daily: xr.Dataset,
    *,
    region: RegionDefinition,
    mesh_profile: str,
    aggregation_mode: str,
) -> bool:
    if str(daily.attrs.get("region_name", "")).strip().lower() != region.name:
        return False
    if str(daily.attrs.get("mesh_profile", "")).strip().lower() != mesh_profile.strip().lower():
        return False
    if str(daily.attrs.get("aggregation_mode", "")).strip().lower() != aggregation_mode.strip().lower():
        return False

    def _matches_bounds(attr_name: str, expected: tuple[float, float, float, float]) -> bool:
        if attr_name not in daily.attrs:
            return False
        actual = np.asarray(daily.attrs[attr_name], dtype=float).reshape(-1)
        return actual.size == 4 and np.allclose(actual, np.asarray(expected, dtype=float), atol=1e-6)

    return _matches_bounds("region_core_bounds", region.core_bounds) and _matches_bounds(
        "region_expanded_bounds", region.expanded_bounds
    )


def _write_region_outputs_from_daily(
    *,
    daily: xr.Dataset,
    valid_date: date,
    output_dir: Path,
    loader_name: str,
    region: RegionDefinition,
    mesh_profile: str,
    aggregation_mode: str,
    publish_preset_name: str | None,
    presentation_theme: str,
    existing_samples_path: Path | None = None,
) -> dict[str, Path]:
    publish_preset = resolve_publish_preset(publish_preset_name) if publish_preset_name else None
    file_prefix = _region_file_prefix(loader_name, region.name, mesh_profile, aggregation_mode)
    field_path = output_dir / f"{file_prefix}_daily_fields_{valid_date:%Y%m%d}.nc"
    map_paths = render_daily_maps(
        daily=daily,
        valid_date=valid_date,
        output_dir=output_dir,
        file_prefix=file_prefix,
        extent=region.expanded_bounds,
        map_label=f"Region {region.name} ({loader_name})",
        include_presentation=True,
        presentation_theme=presentation_theme,
    )
    regional_summary_path = output_dir / f"{file_prefix}_summary_{valid_date:%Y%m%d}.csv"
    summary_record = regional_summary_record(daily, region)
    summary_record["source"] = loader_name
    summary_record["mesh_profile"] = mesh_profile
    summary_record["aggregation_mode"] = aggregation_mode
    stacked = daily["daily_score"].stack(points=("lat", "lon"))
    best_point = stacked.isel(points=int(stacked.argmax("points").values))
    summary_record["sample_best_lat"] = round(float(best_point["lat"].values), 2)
    summary_record["sample_best_lon"] = round(float(best_point["lon"].values), 2)
    center_lat = round((region.lat_min + region.lat_max) / 2.0, 2)
    center_lon = round((region.lon_min + region.lon_max) / 2.0, 2)
    summary_record["sample_center_lat"] = center_lat
    summary_record["sample_center_lon"] = center_lon
    pd.DataFrame([summary_record]).to_csv(regional_summary_path, index=False)
    regional_samples_path = existing_samples_path if existing_samples_path and existing_samples_path.exists() else None
    bundle_paths: dict[str, Path | None] = {
        "daily_fields_netcdf": field_path,
        "debug_score_map": map_paths["raw_map"],
        "debug_category_map": map_paths["category_map"],
        "presentation_score_map": map_paths.get("presentation_raw_map"),
        "presentation_category_map": map_paths.get("presentation_category_map"),
        "summary_csv": regional_summary_path,
        "samples_csv": regional_samples_path,
    }
    bundle_csv_path: Path | None = None
    bundle_json_path: Path | None = None
    if publish_preset and bool(publish_preset["write_bundle_manifest"]):
        bundle_csv_path, bundle_json_path = write_publish_bundle(
            output_dir=output_dir,
            valid_date=valid_date,
            preset_name=str(publish_preset["name"]),
            product_kind="region",
            product_slug=file_prefix.replace("comfortwx_", ""),
            theme_name=presentation_theme,
            bundle_files=bundle_paths,
        )

    return {
        "regional_daily_fields": field_path,
        "regional_raw_map": map_paths["raw_map"],
        "regional_category_map": map_paths["category_map"],
        "regional_presentation_raw_map": map_paths.get("presentation_raw_map"),
        "regional_presentation_category_map": map_paths.get("presentation_category_map"),
        "regional_summary_csv": regional_summary_path,
        "regional_samples_csv": regional_samples_path,
        "regional_publish_bundle_csv": bundle_csv_path,
        "regional_publish_bundle_json": bundle_json_path,
    }


def _write_mosaic_outputs_from_rasters(
    *,
    rasters: list[RegionalDailyRaster],
    normalized_regions: list[str],
    valid_date: date,
    output_dir: Path,
    loader_name: str,
    mesh_profile: str,
    mosaic_blend_method: str,
    mosaic_target_grid: str,
    aggregation_mode: str,
    publish_preset_name: str | None,
    presentation_theme: str,
) -> dict[str, Path]:
    publish_preset = resolve_publish_preset(publish_preset_name) if publish_preset_name else None
    mosaic_daily, mosaic_summary = mosaic_regional_rasters(
        rasters,
        blend_method=mosaic_blend_method,
        target_grid_policy=mosaic_target_grid,
    )
    mosaic_daily.attrs["mesh_profile"] = mesh_profile
    mosaic_daily.attrs["aggregation_mode"] = aggregation_mode
    mosaic_summary["mesh_profile"] = mesh_profile
    mosaic_summary["aggregation_mode"] = aggregation_mode
    stitched_conus = len(normalized_regions) >= 4
    file_prefix = _mosaic_file_prefix(
        normalized_regions,
        loader_name,
        mesh_profile,
        aggregation_mode,
        mosaic_blend_method,
        mosaic_target_grid,
    )
    field_path = output_dir / f"{file_prefix}_daily_fields_{valid_date:%Y%m%d}.nc"
    mosaic_daily.to_netcdf(field_path)
    extent = (
        float(mosaic_daily["lon"].min().values),
        float(mosaic_daily["lon"].max().values),
        float(mosaic_daily["lat"].min().values),
        float(mosaic_daily["lat"].max().values),
    )
    map_paths = render_daily_maps(
        daily=mosaic_daily,
        valid_date=valid_date,
        output_dir=output_dir,
        file_prefix=file_prefix,
        extent=extent,
        map_label=f"Stitched Mosaic {' + '.join(normalized_regions)} ({loader_name})",
        include_presentation=True,
        presentation_theme=presentation_theme,
        product_metadata=(
            {
                "product_title": str(STITCHED_CONUS_PRESENTATION["title"]),
                "subtitle_source_line": str(STITCHED_CONUS_PRESENTATION["subtitle_source_line"]),
            }
            if stitched_conus
            else None
        ),
        presentation_canvas="stitched_conus" if stitched_conus else None,
    )
    summary_path = output_dir / f"{file_prefix}_summary_{valid_date:%Y%m%d}.csv"
    pd.DataFrame([mosaic_summary]).to_csv(summary_path, index=False)
    city_rankings_csv_path: Path | None = None
    city_rankings_json_path: Path | None = None
    if stitched_conus:
        city_rankings_csv_path, city_rankings_json_path = _write_city_rankings(
            daily=mosaic_daily,
            output_dir=output_dir,
            file_prefix=file_prefix,
            valid_date=valid_date,
        )
    bundle_paths: dict[str, Path | None] = {
        "daily_fields_netcdf": field_path,
        "debug_score_map": map_paths["raw_map"],
        "debug_category_map": map_paths["category_map"],
        "presentation_score_map": map_paths.get("presentation_raw_map"),
        "presentation_category_map": map_paths.get("presentation_category_map"),
        "summary_csv": summary_path,
        "seam_diagnostics_csv": summary_path,
        "city_rankings_csv": city_rankings_csv_path,
        "city_rankings_json": city_rankings_json_path,
    }
    bundle_csv_path: Path | None = None
    bundle_json_path: Path | None = None
    if publish_preset and bool(publish_preset["write_bundle_manifest"]):
        bundle_csv_path, bundle_json_path = write_publish_bundle(
            output_dir=output_dir,
            valid_date=valid_date,
            preset_name=str(publish_preset["name"]),
            product_kind="mosaic",
            product_slug=file_prefix.replace("comfortwx_", ""),
            theme_name=presentation_theme,
            bundle_files=bundle_paths,
        )

    return {
        "mosaic_daily_fields": field_path,
        "mosaic_raw_map": map_paths["raw_map"],
        "mosaic_category_map": map_paths["category_map"],
        "mosaic_presentation_raw_map": map_paths.get("presentation_raw_map"),
        "mosaic_presentation_category_map": map_paths.get("presentation_category_map"),
        "mosaic_city_rankings_csv": city_rankings_csv_path,
        "mosaic_city_rankings_json": city_rankings_json_path,
        "mosaic_summary_csv": summary_path,
        "mosaic_publish_bundle_csv": bundle_csv_path,
        "mosaic_publish_bundle_json": bundle_json_path,
    }


def run_pipeline(
    valid_date: date,
    loader_name: str,
    lat_points: int,
    lon_points: int,
    output_dir: Path,
    inspect_lat: float | None = None,
    inspect_lon: float | None = None,
    point_lat: float | None = None,
    point_lon: float | None = None,
    region_name: str | None = None,
    mosaic_regions: list[str] | None = None,
    mesh_profile: str = "standard",
    mosaic_blend_method: str = "taper",
    mosaic_target_grid: str = "adaptive",
    aggregation_mode: str = "baseline",
    publish_preset_name: str | None = None,
    presentation_theme: str = "default",
) -> dict[str, Path]:
    """Run the synthetic V1 pipeline end to end."""

    output_dir.mkdir(parents=True, exist_ok=True)
    region = get_region_definition(region_name) if region_name else None
    publish_preset = resolve_publish_preset(publish_preset_name) if publish_preset_name else None

    if mosaic_regions:
        normalized_regions = [name.strip().lower() for name in mosaic_regions]
        if loader_name != "openmeteo":
            raise ValueError("The pilot mosaic test currently requires --source openmeteo.")
        supported_mosaic_regions = {"west_coast", "southeast", "southwest", "rockies", "plains", "great_lakes", "northeast"}
        if len(normalized_regions) < 2 or len(set(normalized_regions)) != len(normalized_regions):
            raise ValueError("The pilot real mosaic currently requires at least two distinct regions.")
        if not set(normalized_regions).issubset(supported_mosaic_regions):
            raise ValueError("The pilot real mosaic currently supports only west_coast, southwest, rockies, plains, great_lakes, southeast, and northeast.")

        rasters: list[RegionalDailyRaster] = []
        for mosaic_region_name in normalized_regions:
            _, regional_daily, region = _build_regional_daily(
                valid_date=valid_date,
                loader_name=loader_name,
                lat_points=lat_points,
                lon_points=lon_points,
                region_name=mosaic_region_name,
                mesh_profile=mesh_profile,
                aggregation_mode=aggregation_mode,
            )
            rasters.append(RegionalDailyRaster(region=region, daily=regional_daily))

        mosaic_daily, mosaic_summary = mosaic_regional_rasters(
            rasters,
            blend_method=mosaic_blend_method,
            target_grid_policy=mosaic_target_grid,
        )
        mosaic_daily.attrs["mesh_profile"] = mesh_profile
        mosaic_daily.attrs["aggregation_mode"] = aggregation_mode
        mosaic_summary["mesh_profile"] = mesh_profile
        mosaic_summary["aggregation_mode"] = aggregation_mode
        stitched_conus = len(normalized_regions) >= 4
        ordered_regions = "_".join(normalized_regions)
        profile_suffix = "" if mesh_profile == "standard" else f"_{mesh_profile}"
        method_suffix = "" if mosaic_blend_method == "taper" and mosaic_target_grid == "adaptive" else f"_{mosaic_blend_method}_{mosaic_target_grid}"
        aggregation_suffix = "" if aggregation_mode == "baseline" else f"_{aggregation_mode}"
        file_prefix = f"comfortwx_mosaic_{ordered_regions}_openmeteo{profile_suffix}{aggregation_suffix}{method_suffix}"
        field_path = output_dir / f"{file_prefix}_daily_fields_{valid_date:%Y%m%d}.nc"
        mosaic_daily.to_netcdf(field_path)
        extent = (
            float(mosaic_daily["lon"].min().values),
            float(mosaic_daily["lon"].max().values),
            float(mosaic_daily["lat"].min().values),
            float(mosaic_daily["lat"].max().values),
        )
        map_paths = render_daily_maps(
            daily=mosaic_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=file_prefix,
            extent=extent,
            map_label=f"Stitched Mosaic {' + '.join(normalized_regions)} ({loader_name})",
            include_presentation=True,
            presentation_theme=presentation_theme,
            product_metadata=(
                {
                    "product_title": str(STITCHED_CONUS_PRESENTATION["title"]),
                    "subtitle_source_line": str(STITCHED_CONUS_PRESENTATION["subtitle_source_line"]),
                }
                if stitched_conus
                else None
            ),
            presentation_canvas="stitched_conus" if stitched_conus else None,
        )
        summary_path = output_dir / f"{file_prefix}_summary_{valid_date:%Y%m%d}.csv"
        pd.DataFrame([mosaic_summary]).to_csv(summary_path, index=False)
        city_rankings_csv_path: Path | None = None
        city_rankings_json_path: Path | None = None
        if stitched_conus:
            city_rankings_csv_path, city_rankings_json_path = _write_city_rankings(
                daily=mosaic_daily,
                output_dir=output_dir,
                file_prefix=file_prefix,
                valid_date=valid_date,
            )
        bundle_paths: dict[str, Path | None] = {
            "daily_fields_netcdf": field_path,
            "debug_score_map": map_paths["raw_map"],
            "debug_category_map": map_paths["category_map"],
            "presentation_score_map": map_paths.get("presentation_raw_map"),
            "presentation_category_map": map_paths.get("presentation_category_map"),
            "summary_csv": summary_path,
            "seam_diagnostics_csv": summary_path,
            "city_rankings_csv": city_rankings_csv_path,
            "city_rankings_json": city_rankings_json_path,
        }
        if publish_preset and bool(publish_preset["write_bundle_manifest"]):
            bundle_csv_path, bundle_json_path = write_publish_bundle(
                output_dir=output_dir,
                valid_date=valid_date,
                preset_name=str(publish_preset["name"]),
                product_kind="mosaic",
                product_slug=file_prefix.replace("comfortwx_", ""),
                theme_name=presentation_theme,
                bundle_files=bundle_paths,
            )
            bundle_paths["publish_bundle_csv"] = bundle_csv_path
            bundle_paths["publish_bundle_json"] = bundle_json_path

        print(f"Valid date: {valid_date:%Y-%m-%d}")
        print(
            f"Mosaic regions: {', '.join(normalized_regions)} source={loader_name} mesh_profile={mesh_profile} "
            f"aggregation_mode={aggregation_mode} blend_method={mosaic_blend_method} target_grid={mosaic_target_grid}"
        )
        print(f"Saved mosaic daily fields: {field_path}")
        print(f"Saved mosaic raw map: {map_paths['raw_map']}")
        print(f"Saved mosaic category map: {map_paths['category_map']}")
        if "presentation_raw_map" in map_paths and "presentation_category_map" in map_paths:
            print(f"Saved mosaic presentation raw map: {map_paths['presentation_raw_map']}")
            print(f"Saved mosaic presentation category map: {map_paths['presentation_category_map']}")
        print(f"Saved mosaic summary: {summary_path}")
        if city_rankings_csv_path and city_rankings_json_path:
            print(f"Saved city rankings CSV: {city_rankings_csv_path}")
            print(f"Saved city rankings JSON: {city_rankings_json_path}")
        if publish_preset:
            print(f"Saved publish bundle: {bundle_paths['publish_bundle_csv']}")
        return {
            "mosaic_daily_fields": field_path,
            "mosaic_raw_map": map_paths["raw_map"],
            "mosaic_category_map": map_paths["category_map"],
            "mosaic_presentation_raw_map": map_paths.get("presentation_raw_map"),
            "mosaic_presentation_category_map": map_paths.get("presentation_category_map"),
            "mosaic_city_rankings_csv": city_rankings_csv_path,
            "mosaic_city_rankings_json": city_rankings_json_path,
            "mosaic_summary_csv": summary_path,
            "mosaic_publish_bundle_csv": bundle_paths.get("publish_bundle_csv"),
            "mosaic_publish_bundle_json": bundle_paths.get("publish_bundle_json"),
        }

    if region is not None:
        scored_hourly, regional_daily, region = _build_regional_daily(
            valid_date=valid_date,
            loader_name=loader_name,
            lat_points=lat_points,
            lon_points=lon_points,
            region_name=region.name,
            mesh_profile=mesh_profile,
            aggregation_mode=aggregation_mode,
        )

        profile_suffix = "" if mesh_profile == "standard" else f"_{mesh_profile}"
        aggregation_suffix = "" if aggregation_mode == "baseline" else f"_{aggregation_mode}"
        file_prefix = f"comfortwx_region_{region.name}" if loader_name == "mock" else f"comfortwx_region_{region.name}_openmeteo{profile_suffix}{aggregation_suffix}"
        field_path = output_dir / f"{file_prefix}_daily_fields_{valid_date:%Y%m%d}.nc"
        regional_daily.to_netcdf(field_path)
        map_paths = render_daily_maps(
            daily=regional_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=file_prefix,
            extent=region.expanded_bounds,
            map_label=f"Region {region.name} ({loader_name})",
            include_presentation=True,
            presentation_theme=presentation_theme,
        )
        regional_summary_path = output_dir / f"{file_prefix}_summary_{valid_date:%Y%m%d}.csv"
        summary_record = regional_summary_record(regional_daily, region)
        summary_record["source"] = loader_name
        summary_record["mesh_profile"] = mesh_profile
        summary_record["aggregation_mode"] = aggregation_mode
        stacked = regional_daily["daily_score"].stack(points=("lat", "lon"))
        best_point = stacked.isel(points=int(stacked.argmax("points").values))
        summary_record["sample_best_lat"] = round(float(best_point["lat"].values), 2)
        summary_record["sample_best_lon"] = round(float(best_point["lon"].values), 2)
        center_lat = round((region.lat_min + region.lat_max) / 2.0, 2)
        center_lon = round((region.lon_min + region.lon_max) / 2.0, 2)
        summary_record["sample_center_lat"] = center_lat
        summary_record["sample_center_lon"] = center_lon
        pd.DataFrame([summary_record]).to_csv(regional_summary_path, index=False)
        sample_rows: list[dict[str, object]] = []
        for sample_name, sample_lat, sample_lon in [
            ("best_point", float(best_point["lat"].values), float(best_point["lon"].values)),
            ("region_center", center_lat, center_lon),
        ]:
            _, sample_summary_frame, sample_explanation, (resolved_lat, resolved_lon) = inspect_point(
                scored_hourly=scored_hourly,
                daily=regional_daily,
                lat=sample_lat,
                lon=sample_lon,
            )
            sample_record = sample_summary_frame.iloc[0].to_dict()
            sample_record["sample_name"] = sample_name
            sample_record["requested_lat"] = sample_lat
            sample_record["requested_lon"] = sample_lon
            sample_record["resolved_lat"] = round(resolved_lat, 2)
            sample_record["resolved_lon"] = round(resolved_lon, 2)
            sample_record["explanation"] = sample_explanation
            sample_rows.append(sample_record)
        regional_samples_path = output_dir / f"{file_prefix}_samples_{valid_date:%Y%m%d}.csv"
        pd.DataFrame(sample_rows).to_csv(regional_samples_path, index=False)
        bundle_paths: dict[str, Path | None] = {
            "daily_fields_netcdf": field_path,
            "debug_score_map": map_paths["raw_map"],
            "debug_category_map": map_paths["category_map"],
            "presentation_score_map": map_paths.get("presentation_raw_map"),
            "presentation_category_map": map_paths.get("presentation_category_map"),
            "summary_csv": regional_summary_path,
            "samples_csv": regional_samples_path,
        }
        if publish_preset and bool(publish_preset["write_bundle_manifest"]):
            bundle_csv_path, bundle_json_path = write_publish_bundle(
                output_dir=output_dir,
                valid_date=valid_date,
                preset_name=str(publish_preset["name"]),
                product_kind="region",
                product_slug=file_prefix.replace("comfortwx_", ""),
                theme_name=presentation_theme,
                bundle_files=bundle_paths,
            )
            bundle_paths["publish_bundle_csv"] = bundle_csv_path
            bundle_paths["publish_bundle_json"] = bundle_json_path

        print(f"Valid date: {valid_date:%Y-%m-%d}")
        print(
            f"Region: {region.name} source={loader_name} mesh_profile={mesh_profile} "
            f"aggregation_mode={aggregation_mode} core={region.core_bounds} expanded={region.expanded_bounds}"
        )
        print(f"Saved regional daily fields: {field_path}")
        print(f"Saved regional raw map: {map_paths['raw_map']}")
        print(f"Saved regional category map: {map_paths['category_map']}")
        if "presentation_raw_map" in map_paths and "presentation_category_map" in map_paths:
            print(f"Saved regional presentation raw map: {map_paths['presentation_raw_map']}")
            print(f"Saved regional presentation category map: {map_paths['presentation_category_map']}")
        print(f"Saved regional summary: {regional_summary_path}")
        print(f"Saved regional samples: {regional_samples_path}")
        if publish_preset:
            print(f"Saved publish bundle: {bundle_paths['publish_bundle_csv']}")
        return {
            "regional_daily_fields": field_path,
            "regional_raw_map": map_paths["raw_map"],
            "regional_category_map": map_paths["category_map"],
            "regional_presentation_raw_map": map_paths.get("presentation_raw_map"),
            "regional_presentation_category_map": map_paths.get("presentation_category_map"),
            "regional_summary_csv": regional_summary_path,
            "regional_samples_csv": regional_samples_path,
            "regional_publish_bundle_csv": bundle_paths.get("publish_bundle_csv"),
            "regional_publish_bundle_json": bundle_paths.get("publish_bundle_json"),
        }

    loader = get_loader(
        loader_name=loader_name,
        lat_points=lat_points,
        lon_points=lon_points,
        lat=point_lat,
        lon=point_lon,
        region_name=region_name if loader_name == "openmeteo" else None,
        mesh_profile=mesh_profile,
    )
    hourly = loader.load_hourly_grid(valid_date=valid_date)

    if loader_name == "openmeteo" and region is None:
        if point_lat is None or point_lon is None:
            raise ValueError("Open-Meteo point mode requires --lat and --lon.")
        scored_hourly = score_hourly_dataset(hourly)
        daily = aggregate_daily_scores(scored_hourly)
        inspection_outputs = export_point_inspection(
            scored_hourly=scored_hourly,
            daily=daily,
            valid_date=valid_date,
            output_dir=output_dir,
            lat=point_lat,
            lon=point_lon,
        )
        print(f"Valid date: {valid_date:%Y-%m-%d}")
        print(_point_summary(daily))
        print(
            "Resolved point: "
            f"({inspection_outputs['resolved_lat']:.2f}, {inspection_outputs['resolved_lon']:.2f})"
        )
        print(f"Saved hourly diagnostics: {inspection_outputs['hourly_csv']}")
        print(f"Saved daily summary: {inspection_outputs['summary_csv']}")
        print(f"Explanation: {inspection_outputs['explanation']}")
        return {
            "inspection_hourly_csv": inspection_outputs["hourly_csv"],
            "inspection_summary_csv": inspection_outputs["summary_csv"],
        }

    scored_hourly = score_hourly_dataset(hourly)
    daily = aggregate_daily_scores(scored_hourly, aggregation_mode=aggregation_mode)
    field_path = _save_daily_fields(daily=daily, valid_date=valid_date, output_dir=output_dir)
    include_presentation = bool(publish_preset["include_presentation"]) if publish_preset else False
    map_paths = render_daily_maps(
        daily=daily,
        valid_date=valid_date,
        output_dir=output_dir,
        include_presentation=include_presentation,
        presentation_theme=presentation_theme,
    )

    demo_summary, demo_report = run_demo_case_validation(valid_date)
    demo_csv_path = output_dir / f"comfortwx_demo_cases_{valid_date:%Y%m%d}.csv"
    demo_hourly_csv_path = output_dir / f"comfortwx_demo_case_hourly_{valid_date:%Y%m%d}.csv"
    demo_summary.to_csv(demo_csv_path, index=False)
    build_demo_case_hourly_breakdown(valid_date).to_csv(demo_hourly_csv_path, index=False)

    inspection_outputs: dict[str, object] = {}
    if inspect_lat is not None and inspect_lon is not None:
        inspection_outputs = export_point_inspection(
            scored_hourly=scored_hourly,
            daily=daily,
            valid_date=valid_date,
            output_dir=output_dir,
            lat=inspect_lat,
            lon=inspect_lon,
        )

    print(f"Valid date: {valid_date:%Y-%m-%d}")
    print(_grid_summary(daily))
    print(f"Saved daily fields: {field_path}")
    print(f"Saved raw map: {map_paths['raw_map']}")
    print(f"Saved category map: {map_paths['category_map']}")
    print(f"Saved demo case summary: {demo_csv_path}")
    print(f"Saved demo case hourly diagnostics: {demo_hourly_csv_path}")
    if inspection_outputs:
        print(
            "Saved point inspection: "
            f"{inspection_outputs['hourly_csv']} and {inspection_outputs['summary_csv']}"
        )
        print(
            "Inspection point: "
            f"({inspection_outputs['resolved_lat']:.2f}, {inspection_outputs['resolved_lon']:.2f})"
        )
        print(f"Inspection explanation: {inspection_outputs['explanation']}")
    print(demo_report)

    result: dict[str, Path] = {
        "daily_fields": field_path,
        "raw_map": map_paths["raw_map"],
        "category_map": map_paths["category_map"],
        "demo_case_csv": demo_csv_path,
        "demo_case_hourly_csv": demo_hourly_csv_path,
    }
    if publish_preset and bool(publish_preset["write_bundle_manifest"]):
        bundle_csv_path, bundle_json_path = write_publish_bundle(
            output_dir=output_dir,
            valid_date=valid_date,
            preset_name=str(publish_preset["name"]),
            product_kind="grid",
            product_slug="daily",
            theme_name=presentation_theme,
            bundle_files={
                "daily_fields_netcdf": field_path,
                "debug_score_map": map_paths["raw_map"],
                "debug_category_map": map_paths["category_map"],
                "presentation_score_map": map_paths.get("presentation_raw_map"),
                "presentation_category_map": map_paths.get("presentation_category_map"),
                "demo_case_csv": demo_csv_path,
                "demo_case_hourly_csv": demo_hourly_csv_path,
            },
        )
        result["publish_bundle_csv"] = bundle_csv_path
        result["publish_bundle_json"] = bundle_json_path
    if inspection_outputs:
        result["inspection_hourly_csv"] = inspection_outputs["hourly_csv"]
        result["inspection_summary_csv"] = inspection_outputs["summary_csv"]
    return result


def _pilot_day_row(
    *,
    product_type: str,
    product_name: str,
    valid_date: date,
    result: dict[str, Path | None],
    status: str = "completed",
    build_source: str = "",
    notes: str = "",
) -> dict[str, object]:
    prefix = "regional" if product_type == "region" else "mosaic"
    return {
        "product_type": product_type,
        "product_name": product_name,
        "valid_date": valid_date.isoformat(),
        "status": status,
        "build_source": build_source,
        "notes": notes,
        "daily_fields_path": str(result.get(f"{prefix}_daily_fields", "")),
        "debug_score_map_path": str(result.get(f"{prefix}_raw_map", "")),
        "debug_category_map_path": str(result.get(f"{prefix}_category_map", "")),
        "presentation_score_map_path": str(result.get(f"{prefix}_presentation_raw_map", "")),
        "presentation_category_map_path": str(result.get(f"{prefix}_presentation_category_map", "")),
        "city_rankings_csv_path": str(result.get(f"{prefix}_city_rankings_csv", "")),
        "city_rankings_json_path": str(result.get(f"{prefix}_city_rankings_json", "")),
        "summary_csv_path": str(result.get(f"{prefix}_summary_csv", "")),
        "bundle_csv_path": str(result.get(f"{prefix}_publish_bundle_csv", "")),
        "bundle_json_path": str(result.get(f"{prefix}_publish_bundle_json", "")),
        "samples_or_seam_path": str(
            result.get("regional_samples_csv", "") if product_type == "region" else result.get("mosaic_summary_csv", "")
        ),
    }


def _build_city_rankings_frame(daily: xr.Dataset) -> pd.DataFrame:
    score_values = np.asarray(daily["daily_score"].values, dtype=float)
    lat_values = np.asarray(daily["lat"].values, dtype=float)
    lon_values = np.asarray(daily["lon"].values, dtype=float)

    valid_points = np.argwhere(np.isfinite(score_values))
    if valid_points.size == 0:
        return pd.DataFrame(
            columns=[
                "city",
                "score",
                "category",
                "sample_lat",
                "sample_lon",
                "distance_degrees",
                "priority",
                "ranking_group",
                "ranking_position",
            ]
        )

    valid_latitudes = lat_values[valid_points[:, 0]]
    valid_longitudes = lon_values[valid_points[:, 1]]
    rows: list[dict[str, object]] = []
    for city in PUBLIC_CITY_RANKING_LOCATIONS:
        city_lat = float(city["lat"])
        city_lon = float(city["lon"])
        lon_scale = max(np.cos(np.deg2rad(city_lat)), 0.35)
        distance = (valid_latitudes - city_lat) ** 2 + ((valid_longitudes - city_lon) * lon_scale) ** 2
        best_index = int(np.argmin(distance))
        lat_index = int(valid_points[best_index, 0])
        lon_index = int(valid_points[best_index, 1])
        score = float(score_values[lat_index, lon_index])
        rows.append(
            {
                "city": str(city["name"]),
                "score": round(score, 1),
                "category": category_name_from_value(score),
                "sample_lat": round(float(lat_values[lat_index]), 3),
                "sample_lon": round(float(lon_values[lon_index]), 3),
                "distance_degrees": round(float(np.sqrt(distance[best_index])), 3),
                "priority": int(city["priority"]),
            }
        )

    ranking_frame = pd.DataFrame(rows).sort_values(["score", "priority", "city"], ascending=[False, True, True]).reset_index(drop=True)
    best_frame = ranking_frame.head(10).copy()
    best_frame["ranking_group"] = "best"
    best_frame["ranking_position"] = np.arange(1, len(best_frame) + 1)

    worst_frame = ranking_frame.sort_values(["score", "priority", "city"], ascending=[True, True, True]).head(10).copy()
    worst_frame["ranking_group"] = "worst"
    worst_frame["ranking_position"] = np.arange(1, len(worst_frame) + 1)

    return pd.concat([best_frame, worst_frame], ignore_index=True)


def _write_city_rankings(*, daily: xr.Dataset, output_dir: Path, file_prefix: str, valid_date: date) -> tuple[Path, Path]:
    ranking_frame = _build_city_rankings_frame(daily)
    csv_path = output_dir / f"{file_prefix}_city_rankings_{valid_date:%Y%m%d}.csv"
    json_path = output_dir / f"{file_prefix}_city_rankings_{valid_date:%Y%m%d}.json"
    ranking_frame.to_csv(csv_path, index=False)
    payload = {
        "valid_date": valid_date.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cities": ranking_frame.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def _iter_pilot_valid_dates(start_valid_date: date, span_days: int) -> list[date]:
    if span_days < 1:
        raise ValueError("Pilot span days must be at least 1.")
    return [start_valid_date + timedelta(days=offset) for offset in range(span_days)]


def run_pilot_day(
    *,
    valid_date: date,
    loader_name: str,
    output_dir: Path,
    publish_preset_name: str,
    presentation_theme: str,
    lat_points: int,
    lon_points: int,
    mesh_profile: str,
    mosaic_blend_method: str,
    mosaic_target_grid: str,
    aggregation_mode: str,
    pilot_cache_mode: str,
) -> dict[str, Path]:
    """Run all currently supported real pilot regions and seam mosaics for one date."""

    if loader_name != "openmeteo":
        raise ValueError("Pilot-day mode currently requires --source openmeteo.")

    normalized_cache_mode = pilot_cache_mode.strip().lower()
    if normalized_cache_mode not in PILOT_DAY_CACHE_MODES:
        raise ValueError(f"Unsupported pilot-day cache mode '{pilot_cache_mode}'.")

    reset_openmeteo_request_records()
    run_timestamp = datetime.now()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_slug = _run_slug("comfortwx", "pilot_day", loader_name, valid_date.strftime("%Y%m%d"))
    product_rows: list[dict[str, object]] = []
    regional_rasters: dict[str, RegionalDailyRaster] = {}
    region_build_sources: dict[str, str] = {}
    fetched_regions: list[str] = []
    reused_regions: list[str] = []
    refreshed_regions: list[str] = []
    failed_regions: list[str] = []
    built_mosaics: list[str] = []
    skipped_mosaics: list[str] = []
    failed_mosaics: list[str] = []

    with openmeteo_request_context(workflow="pilot_day", label=f"pilot_day:{valid_date.isoformat()}", run_slug=run_slug):
        for region_name in PILOT_DAY_REGIONS:
            region = get_region_definition(region_name)
            region_prefix = _region_file_prefix(loader_name, region_name, mesh_profile, aggregation_mode)
            existing_field_path = output_dir / f"{region_prefix}_daily_fields_{valid_date:%Y%m%d}.nc"
            existing_samples_path = output_dir / f"{region_prefix}_samples_{valid_date:%Y%m%d}.csv"
            try:
                if normalized_cache_mode == "reuse" and existing_field_path.exists():
                    regional_daily = _load_daily_dataset(existing_field_path)
                    if _cached_region_daily_matches_request(
                        regional_daily,
                        region=region,
                        mesh_profile=mesh_profile,
                        aggregation_mode=aggregation_mode,
                    ):
                        result = _write_region_outputs_from_daily(
                            daily=regional_daily,
                            valid_date=valid_date,
                            output_dir=output_dir,
                            loader_name=loader_name,
                            region=region,
                            mesh_profile=mesh_profile,
                            aggregation_mode=aggregation_mode,
                            publish_preset_name=publish_preset_name,
                            presentation_theme=presentation_theme,
                            existing_samples_path=existing_samples_path,
                        )
                        build_source = "reused_region_daily_cache"
                        reused_regions.append(region_name)
                    else:
                        with openmeteo_request_context(
                            workflow="pilot_day",
                            label=f"region={region_name};date={valid_date.isoformat()}",
                            run_slug=run_slug,
                        ):
                            result = run_pipeline(
                                valid_date=valid_date,
                                loader_name=loader_name,
                                lat_points=lat_points,
                                lon_points=lon_points,
                                output_dir=output_dir,
                                region_name=region_name,
                                mesh_profile=mesh_profile,
                                aggregation_mode=aggregation_mode,
                                publish_preset_name=publish_preset_name,
                                presentation_theme=presentation_theme,
                            )
                        regional_daily = _load_daily_dataset(Path(result["regional_daily_fields"]))
                        build_source = "stale_cache_refreshed"
                        refreshed_regions.append(region_name)
                else:
                    with openmeteo_request_context(
                        workflow="pilot_day",
                        label=f"region={region_name};date={valid_date.isoformat()}",
                        run_slug=run_slug,
                    ):
                        result = run_pipeline(
                            valid_date=valid_date,
                            loader_name=loader_name,
                            lat_points=lat_points,
                            lon_points=lon_points,
                            output_dir=output_dir,
                            region_name=region_name,
                            mesh_profile=mesh_profile,
                            aggregation_mode=aggregation_mode,
                            publish_preset_name=publish_preset_name,
                            presentation_theme=presentation_theme,
                        )
                    regional_daily = _load_daily_dataset(Path(result["regional_daily_fields"]))
                    build_source = "fetched_and_persisted"
                    fetched_regions.append(region_name)

                regional_rasters[region_name] = RegionalDailyRaster(region=region, daily=regional_daily)
                region_build_sources[region_name] = build_source
                product_rows.append(
                    _pilot_day_row(
                        product_type="region",
                        product_name=region_name,
                        valid_date=valid_date,
                        result=result,
                        status="completed",
                        build_source=build_source,
                    )
                )
            except Exception as exc:
                failed_regions.append(region_name)
                product_rows.append(
                    _pilot_day_row(
                        product_type="region",
                        product_name=region_name,
                        valid_date=valid_date,
                        result={},
                        status="failed",
                        build_source="fetch_failed" if not existing_field_path.exists() else "cache_reuse_failed",
                        notes=str(exc),
                    )
                )

    for mosaic_region_names in PILOT_DAY_MOSAICS:
        mosaic_name = "+".join(mosaic_region_names)
        missing_regions = [region_name for region_name in mosaic_region_names if region_name not in regional_rasters]
        if missing_regions:
            skipped_mosaics.append(mosaic_name)
            product_rows.append(
                _pilot_day_row(
                    product_type="mosaic",
                    product_name=mosaic_name,
                    valid_date=valid_date,
                    result={},
                    status="skipped",
                    build_source="missing_regional_cache",
                    notes=f"Missing regional daily fields for: {', '.join(missing_regions)}",
                )
            )
            continue

        try:
            result = _write_mosaic_outputs_from_rasters(
                rasters=[regional_rasters[region_name] for region_name in mosaic_region_names],
                normalized_regions=list(mosaic_region_names),
                valid_date=valid_date,
                output_dir=output_dir,
                loader_name=loader_name,
                mesh_profile=mesh_profile,
                mosaic_blend_method=mosaic_blend_method,
                mosaic_target_grid=mosaic_target_grid,
                aggregation_mode=aggregation_mode,
                publish_preset_name=publish_preset_name,
                presentation_theme=presentation_theme,
            )
            source_states = {region_build_sources[region_name] for region_name in mosaic_region_names}
            if source_states == {"reused_region_daily_cache"}:
                build_source = "built_from_cached_regions"
            elif source_states == {"fetched_and_persisted"}:
                build_source = "built_from_fresh_regions"
            else:
                build_source = "built_from_mixed_region_cache"
            built_mosaics.append(mosaic_name)
            product_rows.append(
                _pilot_day_row(
                    product_type="mosaic",
                    product_name=mosaic_name,
                    valid_date=valid_date,
                    result=result,
                    status="completed",
                    build_source=build_source,
                )
            )
        except Exception as exc:
            failed_mosaics.append(mosaic_name)
            product_rows.append(
                _pilot_day_row(
                    product_type="mosaic",
                    product_name=mosaic_name,
                    valid_date=valid_date,
                    result={},
                    status="failed",
                    build_source="regional_cache_build_failed",
                    notes=str(exc),
                )
            )

    request_summary_path, request_detail_path = write_openmeteo_request_report(output_dir=output_dir, run_slug=run_slug)
    request_summary_record = pd.read_csv(request_summary_path).iloc[0].to_dict()
    completed_count = sum(1 for row in product_rows if row["status"] == "completed")
    attempted_count = len(product_rows)
    if failed_regions or failed_mosaics:
        overall_run_status = "partial_failure"
    elif skipped_mosaics:
        overall_run_status = "completed_with_skips"
    else:
        overall_run_status = "completed"
    status_record = {
        "valid_date": valid_date.isoformat(),
        "run_timestamp": run_timestamp.isoformat(timespec="seconds"),
        "source": loader_name,
        "product_count_attempted": attempted_count,
        "product_count_completed": completed_count,
        "regions_fetched": ",".join(fetched_regions),
        "regions_refreshed_from_stale_cache": ",".join(refreshed_regions),
        "regions_reused_from_cache": ",".join(reused_regions),
        "regions_failed": ",".join(failed_regions),
        "mosaics_built_from_cached_regional_fields": ",".join(
            [
                row["product_name"]
                for row in product_rows
                if row["product_type"] == "mosaic" and row["build_source"] == "built_from_cached_regions"
            ]
        ),
        "mosaics_built": ",".join(built_mosaics),
        "mosaics_skipped": ",".join(skipped_mosaics),
        "mosaics_failed": ",".join(failed_mosaics),
        "openmeteo_total_requests": int(request_summary_record.get("total_requests", 0)),
        "openmeteo_successful_requests": int(request_summary_record.get("successful_requests", 0)),
        "openmeteo_retry_events": int(request_summary_record.get("retry_events", 0)),
        "openmeteo_timeouts": int(request_summary_record.get("timeouts", 0)),
        "openmeteo_http_429s": int(request_summary_record.get("http_429s", 0)),
        "openmeteo_errors": int(request_summary_record.get("errors", 0)),
        "openmeteo_average_elapsed_seconds": float(request_summary_record.get("average_elapsed_seconds", 0.0)),
        "overall_run_status": overall_run_status,
    }
    status_csv_path, status_json_path = write_pilot_day_status_summary(
        output_dir=output_dir,
        valid_date=valid_date,
        source_name=loader_name,
        status_record=status_record,
    )
    index_csv_path, index_json_path, index_html_path = write_pilot_day_index(
        output_dir=output_dir,
        valid_date=valid_date,
        source_name=loader_name,
        presentation_theme=presentation_theme,
        publish_preset_name=publish_preset_name,
        product_rows=product_rows,
        run_timestamp=run_timestamp,
        status_summary_csv_path=status_csv_path,
        status_summary_json_path=status_json_path,
    )

    print(f"Pilot day complete for {valid_date:%Y-%m-%d}")
    print(f"Cache mode: {normalized_cache_mode}")
    print(f"Regions fetched: {', '.join(fetched_regions) if fetched_regions else 'none'}")
    print(f"Regions refreshed from stale cache: {', '.join(refreshed_regions) if refreshed_regions else 'none'}")
    print(f"Regions reused from cache: {', '.join(reused_regions) if reused_regions else 'none'}")
    print(f"Regions failed: {', '.join(failed_regions) if failed_regions else 'none'}")
    print(f"Mosaics built from regional daily fields: {', '.join(built_mosaics) if built_mosaics else 'none'}")
    print(f"Mosaics skipped: {', '.join(skipped_mosaics) if skipped_mosaics else 'none'}")
    print(f"Mosaics failed: {', '.join(failed_mosaics) if failed_mosaics else 'none'}")
    print(f"Saved pilot-day status CSV: {status_csv_path}")
    print(f"Saved pilot-day status JSON: {status_json_path}")
    print(f"Saved pilot-day index CSV: {index_csv_path}")
    print(f"Saved pilot-day index JSON: {index_json_path}")
    print(f"Saved pilot-day index HTML: {index_html_path}")
    print(f"Saved Open-Meteo request summary: {request_summary_path}")
    print(f"Saved Open-Meteo request detail: {request_detail_path}")
    return {
        "pilot_day_status_csv": status_csv_path,
        "pilot_day_status_json": status_json_path,
        "pilot_day_index_csv": index_csv_path,
        "pilot_day_index_json": index_json_path,
        "pilot_day_index_html": index_html_path,
        "pilot_day_openmeteo_request_summary_csv": request_summary_path,
        "pilot_day_openmeteo_request_detail_csv": request_detail_path,
    }


def run_pilot_day_archive(
    *,
    valid_date: date,
    loader_name: str,
    output_dir: Path,
    archive_root: Path | None,
    archive_layout: str,
    publish_preset_name: str,
    presentation_theme: str,
    lat_points: int,
    lon_points: int,
    mesh_profile: str,
    mosaic_blend_method: str,
    mosaic_target_grid: str,
    aggregation_mode: str,
    pilot_cache_mode: str,
) -> dict[str, Path]:
    """Run a pilot-day build into a dated archive folder and refresh the archive landing page."""

    resolved_archive_root = (archive_root or (output_dir / ARCHIVE_SETTINGS["root_name"])).resolve()
    run_dir = build_archive_run_directory(
        archive_root=resolved_archive_root,
        valid_date=valid_date,
        layout=archive_layout,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    pilot_outputs = run_pilot_day(
        valid_date=valid_date,
        loader_name=loader_name,
        output_dir=run_dir,
        publish_preset_name=publish_preset_name,
        presentation_theme=presentation_theme,
        lat_points=lat_points,
        lon_points=lon_points,
        mesh_profile=mesh_profile,
        mosaic_blend_method=mosaic_blend_method,
        mosaic_target_grid=mosaic_target_grid,
        aggregation_mode=aggregation_mode,
        pilot_cache_mode=pilot_cache_mode,
    )
    archive_csv_path, archive_json_path, archive_html_path = write_archive_index(archive_root=resolved_archive_root)
    print(f"Archive root: {resolved_archive_root}")
    print(f"Archived run directory: {run_dir}")
    print(f"Saved archive landing page: {archive_html_path}")
    return {
        **pilot_outputs,
        "archive_index_csv": archive_csv_path,
        "archive_index_json": archive_json_path,
        "archive_index_html": archive_html_path,
    }


def run_pilot_day_series(
    *,
    start_valid_date: date,
    span_days: int,
    archive_mode: bool,
    loader_name: str,
    output_dir: Path,
    archive_root: Path | None,
    archive_layout: str,
    publish_preset_name: str,
    presentation_theme: str,
    lat_points: int,
    lon_points: int,
    mesh_profile: str,
    mosaic_blend_method: str,
    mosaic_target_grid: str,
    aggregation_mode: str,
    pilot_cache_mode: str,
) -> dict[str, Path]:
    valid_dates = _iter_pilot_valid_dates(start_valid_date, span_days)
    last_outputs: dict[str, Path] = {}
    for index, current_valid_date in enumerate(valid_dates, start=1):
        print(f"Running pilot-day product build {index}/{len(valid_dates)} for {current_valid_date:%Y-%m-%d}")
        if archive_mode:
            last_outputs = run_pilot_day_archive(
                valid_date=current_valid_date,
                loader_name=loader_name,
                output_dir=output_dir,
                archive_root=archive_root,
                archive_layout=archive_layout,
                publish_preset_name=publish_preset_name,
                presentation_theme=presentation_theme,
                lat_points=lat_points,
                lon_points=lon_points,
                mesh_profile=mesh_profile,
                mosaic_blend_method=mosaic_blend_method,
                mosaic_target_grid=mosaic_target_grid,
                aggregation_mode=aggregation_mode,
                pilot_cache_mode=pilot_cache_mode,
            )
        else:
            last_outputs = run_pilot_day(
                valid_date=current_valid_date,
                loader_name=loader_name,
                output_dir=output_dir,
                publish_preset_name=publish_preset_name,
                presentation_theme=presentation_theme,
                lat_points=lat_points,
                lon_points=lon_points,
                mesh_profile=mesh_profile,
                mosaic_blend_method=mosaic_blend_method,
                mosaic_target_grid=mosaic_target_grid,
                aggregation_mode=aggregation_mode,
                pilot_cache_mode=pilot_cache_mode,
            )
    if len(valid_dates) > 1:
        print(
            f"Pilot-day span complete: {len(valid_dates)} dates from "
            f"{valid_dates[0]:%Y-%m-%d} through {valid_dates[-1]:%Y-%m-%d}"
        )
    return last_outputs


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    loader_name = args.source if args.source != "mock" else args.loader
    if args.source == "mock":
        loader_name = args.loader
    else:
        loader_name = args.source
    if args.pilot_day_archive:
        run_pilot_day_series(
            start_valid_date=valid_date,
            span_days=args.pilot_span_days,
            archive_mode=True,
            loader_name=loader_name,
            output_dir=Path(args.output_dir),
            archive_root=Path(args.archive_root) if args.archive_root else None,
            archive_layout=args.archive_layout,
            publish_preset_name=args.publish_preset or "standard",
            presentation_theme=args.presentation_theme,
            lat_points=args.lat_points,
            lon_points=args.lon_points,
            mesh_profile=args.mesh_profile,
            mosaic_blend_method=args.mosaic_blend_method,
            mosaic_target_grid=args.mosaic_target_grid,
            aggregation_mode=args.aggregation_mode,
            pilot_cache_mode=args.pilot_cache_mode,
        )
        return
    if args.pilot_day:
        run_pilot_day_series(
            start_valid_date=valid_date,
            span_days=args.pilot_span_days,
            archive_mode=False,
            loader_name=loader_name,
            output_dir=Path(args.output_dir),
            archive_root=None,
            archive_layout=args.archive_layout,
            publish_preset_name=args.publish_preset or "standard",
            presentation_theme=args.presentation_theme,
            lat_points=args.lat_points,
            lon_points=args.lon_points,
            mesh_profile=args.mesh_profile,
            mosaic_blend_method=args.mosaic_blend_method,
            mosaic_target_grid=args.mosaic_target_grid,
            aggregation_mode=args.aggregation_mode,
            pilot_cache_mode=args.pilot_cache_mode,
        )
        return
    output_dir = Path(args.output_dir)
    if loader_name == "openmeteo":
        if args.mosaic:
            workflow_name = "mosaic_build"
            label = f"mosaic={'+'.join(args.mosaic)};date={valid_date.isoformat()}"
            run_slug = _run_slug("comfortwx", "mosaic", *args.mosaic, loader_name, valid_date.strftime("%Y%m%d"))
        elif args.region:
            workflow_name = "region_build"
            label = f"region={args.region};date={valid_date.isoformat()}"
            run_slug = _run_slug("comfortwx", "region", args.region, loader_name, valid_date.strftime("%Y%m%d"))
        else:
            workflow_name = "point_build"
            label = f"point={args.lat},{args.lon};date={valid_date.isoformat()}"
            run_slug = _run_slug("comfortwx", "point", loader_name, valid_date.strftime("%Y%m%d"))
        reset_openmeteo_request_records()
        try:
            with openmeteo_request_context(workflow=workflow_name, label=label, run_slug=run_slug):
                run_pipeline(
                    valid_date=valid_date,
                    loader_name=loader_name,
                    lat_points=args.lat_points,
                    lon_points=args.lon_points,
                    output_dir=output_dir,
                    inspect_lat=args.inspect_lat,
                    inspect_lon=args.inspect_lon,
                    point_lat=args.lat,
                    point_lon=args.lon,
                    region_name=args.region,
                    mosaic_regions=args.mosaic,
                    mesh_profile=args.mesh_profile,
                    mosaic_blend_method=args.mosaic_blend_method,
                    mosaic_target_grid=args.mosaic_target_grid,
                    aggregation_mode=args.aggregation_mode,
                    publish_preset_name=args.publish_preset,
                    presentation_theme=args.presentation_theme,
                )
        finally:
            request_summary_path, request_detail_path = write_openmeteo_request_report(output_dir=output_dir, run_slug=run_slug)
            print(f"Saved Open-Meteo request summary: {request_summary_path}")
            print(f"Saved Open-Meteo request detail: {request_detail_path}")
        return

    run_pipeline(
        valid_date=valid_date,
        loader_name=loader_name,
        lat_points=args.lat_points,
        lon_points=args.lon_points,
        output_dir=output_dir,
        inspect_lat=args.inspect_lat,
        inspect_lon=args.inspect_lon,
        point_lat=args.lat,
        point_lon=args.lon,
        region_name=args.region,
        mosaic_regions=args.mosaic,
        mesh_profile=args.mesh_profile,
        mosaic_blend_method=args.mosaic_blend_method,
        mosaic_target_grid=args.mosaic_target_grid,
        aggregation_mode=args.aggregation_mode,
        publish_preset_name=args.publish_preset,
        presentation_theme=args.presentation_theme,
    )


if __name__ == "__main__":
    main()
