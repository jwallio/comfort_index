"""Forecast-vs-analysis Comfort Index verification runner."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from comfortwx.config import (
    DAYTIME_HOUR_WEIGHTS,
    LOCAL_DAY_HOURS,
    OPENMETEO_VERIFICATION_DEFAULT_REGION,
    OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS,
    OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC,
    OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_SAMPLE_POINT_NAMES,
    OUTPUT_DIR,
    VERIFICATION_HIGH_COMFORT_CATEGORY_MIN_INDEX,
    list_verification_aggregation_policies,
    resolve_verification_aggregation_mode,
)
from comfortwx.data.openmeteo_reliability import (
    openmeteo_request_context,
    reset_openmeteo_request_records,
    write_openmeteo_request_report,
)
from comfortwx.data.openmeteo_verification import (
    OpenMeteoVerificationRegionalLoader,
    resolve_openmeteo_verification_forecast_model,
)
from comfortwx.mapping.plotting import render_daily_maps
from comfortwx.mapping.regions import get_region_definition, list_region_names
from comfortwx.scoring.categories import category_name_from_index
from comfortwx.scoring.daily import aggregate_daily_scores
from comfortwx.scoring.hourly import score_hourly_dataset


HOURLY_COMPONENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("temp_score", "temp"),
    ("dewpoint_score", "dewpoint"),
    ("wind_score", "wind"),
    ("cloud_score", "cloud"),
    ("precip_score", "precip"),
    ("interaction_adjustment", "interaction"),
    ("hazard_penalty", "hazard_penalty"),
)

DAILY_COMPONENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("best_3hr", "best_3hr"),
    ("best_6hr", "best_6hr"),
    ("daytime_weighted_mean", "daytime_weighted_mean"),
    ("reliability_score", "reliability_score"),
    ("disruption_penalty", "disruption_penalty"),
)


def _category_counts(prefix: str, daily: xr.Dataset) -> dict[str, object]:
    values = daily["category_index"].values.astype(int).ravel()
    records: dict[str, object] = {}
    for index in sorted(np.unique(values)):
        label = category_name_from_index(int(index)).lower()
        records[f"{prefix}_{label}_count"] = int(np.sum(values == index))
    return records


def _slugify_policy_name(policy_name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in policy_name.strip().lower()).strip("_")


def build_verification_file_prefix(
    *,
    region_name: str,
    resolved_forecast_model: str,
    analysis_model: str,
    forecast_lead_days: int,
    aggregation_policy: str,
) -> str:
    normalized_analysis_model = "".join(character if character.isalnum() else "_" for character in analysis_model.strip().lower()).strip("_")
    prefix = f"comfortwx_verify_{region_name}_{resolved_forecast_model}_{normalized_analysis_model}_d{forecast_lead_days}"
    normalized_policy = aggregation_policy.strip().lower()
    if normalized_policy == "baseline":
        return prefix
    return f"{prefix}_policy_{_slugify_policy_name(aggregation_policy)}"


def _verification_summary(
    *,
    forecast_daily: xr.Dataset,
    analysis_daily: xr.Dataset,
    metadata: dict[str, object],
    valid_date: date,
) -> dict[str, object]:
    diff = forecast_daily["daily_score"] - analysis_daily["daily_score"]
    abs_diff = np.abs(diff.values)
    category_diff = np.abs(forecast_daily["category_index"].values - analysis_daily["category_index"].values)
    valid_mask = np.isfinite(diff.values)
    high_comfort_mask = analysis_daily["category_index"].values >= VERIFICATION_HIGH_COMFORT_CATEGORY_MIN_INDEX
    forecast_high_comfort_mask = forecast_daily["category_index"].values >= VERIFICATION_HIGH_COMFORT_CATEGORY_MIN_INDEX
    missed_high_comfort = high_comfort_mask & ~forecast_high_comfort_mask & valid_mask
    false_high_comfort = forecast_high_comfort_mask & ~high_comfort_mask & valid_mask
    high_comfort_hits = high_comfort_mask & forecast_high_comfort_mask & valid_mask
    high_comfort_precision = float(np.sum(high_comfort_hits)) / float(np.sum(forecast_high_comfort_mask & valid_mask)) if np.sum(forecast_high_comfort_mask & valid_mask) else np.nan
    high_comfort_recall = float(np.sum(high_comfort_hits)) / float(np.sum(high_comfort_mask & valid_mask)) if np.sum(high_comfort_mask & valid_mask) else np.nan

    summary = {
        "valid_date": valid_date.isoformat(),
        "region_name": metadata["region_name"],
        "forecast_lead_days": metadata["forecast_lead_days"],
        "forecast_source": f"Open-Meteo Previous Runs ({metadata['forecast_model']})",
        "analysis_source": str(metadata["analysis_model"]).replace("_", " "),
        "forecast_run_timestamp_utc": metadata["forecast_run_timestamp_utc"],
        "mesh_profile": metadata["mesh_profile"],
        "mesh_point_count": metadata["mesh_point_count"],
        "verification_aggregation_policy": metadata["aggregation_policy"],
        "verification_aggregation_mode": metadata["aggregation_mode"],
        "grid_lat_count": int(forecast_daily.sizes["lat"]),
        "grid_lon_count": int(forecast_daily.sizes["lon"]),
        "forecast_mean_score": round(float(forecast_daily["daily_score"].mean().values), 2),
        "analysis_mean_score": round(float(analysis_daily["daily_score"].mean().values), 2),
        "score_bias_mean": round(float(diff.mean().values), 2),
        "score_mae": round(float(np.nanmean(abs_diff)), 2),
        "score_rmse": round(float(np.sqrt(np.nanmean(diff.values**2))), 2),
        "score_max_abs_diff": round(float(np.nanmax(abs_diff)), 2),
        "exact_category_agreement_fraction": round(float(np.mean((category_diff == 0)[valid_mask])), 4),
        "near_category_agreement_fraction": round(float(np.mean((category_diff <= 1)[valid_mask])), 4),
        "category_disagreement_fraction": round(float(np.mean((category_diff > 0)[valid_mask])), 4),
        "high_comfort_analysis_cell_count": int(np.sum(high_comfort_mask & valid_mask)),
        "high_comfort_forecast_cell_count": int(np.sum(forecast_high_comfort_mask & valid_mask)),
        "missed_high_comfort_cell_count": int(np.sum(missed_high_comfort)),
        "false_high_comfort_cell_count": int(np.sum(false_high_comfort)),
        "high_comfort_precision": round(high_comfort_precision, 4) if np.isfinite(high_comfort_precision) else "",
        "high_comfort_recall": round(high_comfort_recall, 4) if np.isfinite(high_comfort_recall) else "",
    }
    summary.update(_category_counts("forecast", forecast_daily))
    summary.update(_category_counts("analysis", analysis_daily))
    return summary


def _sample_point_targets(region_name: str) -> list[tuple[str, float, float]]:
    region = get_region_definition(region_name)
    center_lat = (region.lat_min + region.lat_max) / 2.0
    center_lon = (region.lon_min + region.lon_max) / 2.0
    targets = {
        "center": (center_lat, center_lon),
        "northwest": (region.lat_max, region.lon_min),
        "northeast": (region.lat_max, region.lon_max),
        "southwest": (region.lat_min, region.lon_min),
        "southeast": (region.lat_min, region.lon_max),
    }
    return [(name, *targets[name]) for name in OPENMETEO_VERIFICATION_SAMPLE_POINT_NAMES]


def _select_daytime_hours(dataset: xr.Dataset) -> xr.Dataset:
    start_hour, end_hour = LOCAL_DAY_HOURS
    hour_values = dataset["time"].dt.hour
    return dataset.where((hour_values >= start_hour) & (hour_values <= end_hour), drop=True)


def _daytime_weights(dataset: xr.Dataset) -> xr.DataArray:
    hours = dataset["time"].dt.hour.values
    weights = np.array([DAYTIME_HOUR_WEIGHTS[int(hour)] for hour in hours], dtype=float)
    return xr.DataArray(weights, dims=("time",), coords={"time": dataset["time"]})


def _point_metrics(
    *,
    forecast_daily: xr.Dataset,
    analysis_daily: xr.Dataset,
    region_name: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sample_name, target_lat, target_lon in _sample_point_targets(region_name):
        forecast_point = forecast_daily.sel(lat=target_lat, lon=target_lon, method="nearest")
        actual_lat = float(forecast_point["lat"].values)
        actual_lon = float(forecast_point["lon"].values)
        analysis_point = analysis_daily.sel(lat=actual_lat, lon=actual_lon)
        forecast_score = float(forecast_point["daily_score"].values)
        analysis_score = float(analysis_point["daily_score"].values)
        forecast_category_index = int(forecast_point["category_index"].values)
        analysis_category_index = int(analysis_point["category_index"].values)
        records.append(
            {
                "sample_name": sample_name,
                "lat": round(actual_lat, 2),
                "lon": round(actual_lon, 2),
                "forecast_score": round(forecast_score, 2),
                "analysis_score": round(analysis_score, 2),
                "score_diff": round(forecast_score - analysis_score, 2),
                "forecast_category": category_name_from_index(forecast_category_index),
                "analysis_category": category_name_from_index(analysis_category_index),
                "exact_category_match": forecast_category_index == analysis_category_index,
                "near_category_match": abs(forecast_category_index - analysis_category_index) <= 1,
            }
        )
    return pd.DataFrame.from_records(records)


def _component_metrics(
    *,
    forecast_scored_hourly: xr.Dataset,
    analysis_scored_hourly: xr.Dataset,
    forecast_daily: xr.Dataset,
    analysis_daily: xr.Dataset,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    forecast_daytime = _select_daytime_hours(forecast_scored_hourly)
    analysis_daytime = _select_daytime_hours(analysis_scored_hourly)
    weights = _daytime_weights(forecast_daytime)

    for variable_name, label in HOURLY_COMPONENT_FIELDS:
        forecast_component = forecast_daytime[variable_name].weighted(weights).mean("time")
        analysis_component = analysis_daytime[variable_name].weighted(weights).mean("time")
        diff = forecast_component - analysis_component
        records.append(
            {
                "component_group": "hourly_component",
                "component_name": label,
                "forecast_mean": round(float(forecast_component.mean().values), 3),
                "analysis_mean": round(float(analysis_component.mean().values), 3),
                "bias_mean": round(float(diff.mean().values), 3),
                "mae": round(float(np.abs(diff.values).mean()), 3),
            }
        )

    for variable_name, label in DAILY_COMPONENT_FIELDS:
        forecast_component = forecast_daily[variable_name]
        analysis_component = analysis_daily[variable_name]
        diff = forecast_component - analysis_component
        records.append(
            {
                "component_group": "daily_aggregate",
                "component_name": label,
                "forecast_mean": round(float(forecast_component.mean().values), 3),
                "analysis_mean": round(float(analysis_component.mean().values), 3),
                "bias_mean": round(float(diff.mean().values), 3),
                "mae": round(float(np.abs(diff.values).mean()), 3),
            }
        )

    return pd.DataFrame.from_records(records)


def _component_summary_fields(component_metrics: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for record in component_metrics.to_dict(orient="records"):
        component_name = str(record["component_name"])
        summary[f"{component_name}_bias_mean"] = record["bias_mean"]
        summary[f"{component_name}_mae"] = record["mae"]
    return summary


def _apply_truth_observability_overrides(
    *,
    forecast_hourly: xr.Dataset,
    analysis_hourly: xr.Dataset,
    metadata: dict[str, object],
) -> tuple[xr.Dataset, xr.Dataset]:
    forecast_adjusted = forecast_hourly.copy()
    analysis_adjusted = analysis_hourly.copy()
    if not bool(metadata.get("analysis_has_thunder_truth", True)):
        if "thunder" in forecast_adjusted:
            forecast_adjusted["thunder"] = xr.zeros_like(forecast_adjusted["thunder"]).astype(bool)
        if "thunder" in analysis_adjusted:
            analysis_adjusted["thunder"] = xr.zeros_like(analysis_adjusted["thunder"]).astype(bool)
    return forecast_adjusted, analysis_adjusted


def _write_difference_map(
    *,
    forecast_daily: xr.Dataset,
    analysis_daily: xr.Dataset,
    valid_date: date,
    output_dir: Path,
    file_prefix: str,
) -> Path:
    diff = (forecast_daily["daily_score"] - analysis_daily["daily_score"]).transpose("lat", "lon")
    max_abs = max(5.0, float(np.nanmax(np.abs(diff.values))))
    lon_min = float(diff["lon"].min().values)
    lon_max = float(diff["lon"].max().values)
    lat_min = float(diff["lat"].min().values)
    lat_max = float(diff["lat"].max().values)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        diff["lon"].values,
        diff["lat"].values,
        diff.values,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        shading="auto",
    )
    ax.set_title(f"Comfort Index Forecast - Analysis Difference ({valid_date:%Y-%m-%d})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(alpha=0.2, linewidth=0.4)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Score difference (forecast - analysis)")
    path = output_dir / f"{file_prefix}_score_diff_{valid_date:%Y%m%d}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_absolute_error_map(
    *,
    forecast_daily: xr.Dataset,
    analysis_daily: xr.Dataset,
    valid_date: date,
    output_dir: Path,
    file_prefix: str,
) -> Path:
    abs_error = np.abs((forecast_daily["daily_score"] - analysis_daily["daily_score"]).transpose("lat", "lon"))
    error_ceiling = max(10.0, float(np.nanpercentile(abs_error.values, 95)))
    lon_min = float(abs_error["lon"].min().values)
    lon_max = float(abs_error["lon"].max().values)
    lat_min = float(abs_error["lat"].min().values)
    lat_max = float(abs_error["lat"].max().values)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        abs_error["lon"].values,
        abs_error["lat"].values,
        abs_error.values,
        cmap="magma",
        vmin=0.0,
        vmax=error_ceiling,
        shading="auto",
    )
    ax.set_title(f"Comfort Index Absolute Error ({valid_date:%Y-%m-%d})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(alpha=0.2, linewidth=0.4)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Absolute score error")
    path = output_dir / f"{file_prefix}_absolute_error_{valid_date:%Y%m%d}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_category_disagreement_map(
    *,
    forecast_daily: xr.Dataset,
    analysis_daily: xr.Dataset,
    valid_date: date,
    output_dir: Path,
    file_prefix: str,
) -> Path:
    category_diff = np.abs(
        forecast_daily["category_index"].values.astype(int) - analysis_daily["category_index"].values.astype(int)
    )
    mismatch_class = np.where(category_diff == 0, 0, np.where(category_diff == 1, 1, 2))
    diff_da = xr.DataArray(
        mismatch_class,
        coords=forecast_daily["category_index"].coords,
        dims=forecast_daily["category_index"].dims,
    ).transpose("lat", "lon")
    lon_min = float(diff_da["lon"].min().values)
    lon_max = float(diff_da["lon"].max().values)
    lat_min = float(diff_da["lat"].min().values)
    lat_max = float(diff_da["lat"].max().values)
    cmap = matplotlib.colors.ListedColormap(["#f7f7f7", "#f4c96f", "#d86b5c"])
    bounds = [-0.5, 0.5, 1.5, 2.5]

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        diff_da["lon"].values,
        diff_da["lat"].values,
        diff_da.values,
        cmap=cmap,
        vmin=bounds[0],
        vmax=bounds[-1],
        shading="auto",
    )
    ax.set_title(f"Comfort Category Disagreement ({valid_date:%Y-%m-%d})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(alpha=0.2, linewidth=0.4)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Match", "Near miss", "Large miss"])
    cbar.set_label("Category mismatch class")
    path = output_dir / f"{file_prefix}_category_disagreement_{valid_date:%Y%m%d}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_high_comfort_mask_map(
    *,
    forecast_daily: xr.Dataset,
    analysis_daily: xr.Dataset,
    valid_date: date,
    output_dir: Path,
    file_prefix: str,
    mode: str,
) -> Path:
    if mode == "missed":
        mask = (
            (analysis_daily["category_index"].values >= VERIFICATION_HIGH_COMFORT_CATEGORY_MIN_INDEX)
            & (forecast_daily["category_index"].values < VERIFICATION_HIGH_COMFORT_CATEGORY_MIN_INDEX)
        )
        title = "Missed High Comfort"
        color = "#d95f0e"
    elif mode == "false":
        mask = (
            (forecast_daily["category_index"].values >= VERIFICATION_HIGH_COMFORT_CATEGORY_MIN_INDEX)
            & (analysis_daily["category_index"].values < VERIFICATION_HIGH_COMFORT_CATEGORY_MIN_INDEX)
        )
        title = "False High Comfort"
        color = "#756bb1"
    else:
        raise ValueError(f"Unsupported high-comfort mask mode '{mode}'.")

    mask_da = xr.DataArray(
        mask.astype(int),
        coords=forecast_daily["category_index"].coords,
        dims=forecast_daily["category_index"].dims,
    ).transpose("lat", "lon")
    lon_min = float(mask_da["lon"].min().values)
    lon_max = float(mask_da["lon"].max().values)
    lat_min = float(mask_da["lat"].min().values)
    lat_max = float(mask_da["lat"].max().values)
    cmap = matplotlib.colors.ListedColormap(["#f7f7f7", color])

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        mask_da["lon"].values,
        mask_da["lat"].values,
        mask_da.values,
        cmap=cmap,
        vmin=-0.5,
        vmax=1.5,
        shading="auto",
    )
    ax.set_title(f"{title} Mask ({valid_date:%Y-%m-%d})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(alpha=0.2, linewidth=0.4)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.02, ticks=[0, 1])
    cbar.ax.set_yticklabels(["No", "Yes"])
    cbar.set_label("Mask")
    path = output_dir / f"{file_prefix}_{mode}_high_comfort_{valid_date:%Y%m%d}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def run_verification(
    *,
    valid_date: date,
    region_name: str,
    output_dir: Path,
    mesh_profile: str,
    forecast_model: str,
    forecast_run_hour_utc: int,
    forecast_lead_days: int,
    analysis_model: str = OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    aggregation_policy: str = "baseline",
    workflow_name: str = "verification_model",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_forecast_model = resolve_openmeteo_verification_forecast_model(
        requested_model=forecast_model,
        forecast_lead_days=forecast_lead_days,
    )
    aggregation_mode = resolve_verification_aggregation_mode(
        policy_name=aggregation_policy,
        region_name=region_name,
        valid_date=valid_date,
        forecast_lead_days=forecast_lead_days,
    )
    file_prefix = build_verification_file_prefix(
        region_name=region_name,
        resolved_forecast_model=resolved_forecast_model,
        analysis_model=analysis_model,
        forecast_lead_days=forecast_lead_days,
        aggregation_policy=aggregation_policy,
    )
    request_report_slug = f"{file_prefix}_{valid_date:%Y%m%d}"
    reset_openmeteo_request_records()
    try:
        loader = OpenMeteoVerificationRegionalLoader(
            region_name=region_name,
            mesh_profile=mesh_profile,
            forecast_model=forecast_model,
            analysis_model=analysis_model,
            forecast_run_hour_utc=forecast_run_hour_utc,
            forecast_lead_days=forecast_lead_days,
        )
        with openmeteo_request_context(
            workflow=workflow_name,
            label=f"region={region_name};date={valid_date.isoformat()}",
            run_slug=request_report_slug,
        ):
            forecast_hourly, analysis_hourly, metadata = loader.load_pair(valid_date)
        forecast_hourly, analysis_hourly = _apply_truth_observability_overrides(
            forecast_hourly=forecast_hourly,
            analysis_hourly=analysis_hourly,
            metadata=metadata,
        )
        forecast_scored_hourly = score_hourly_dataset(forecast_hourly)
        analysis_scored_hourly = score_hourly_dataset(analysis_hourly)
        forecast_daily = aggregate_daily_scores(forecast_scored_hourly, aggregation_mode=aggregation_mode)
        analysis_daily = aggregate_daily_scores(analysis_scored_hourly, aggregation_mode=aggregation_mode)

        forecast_daily_path = output_dir / f"{file_prefix}_forecast_daily_fields_{valid_date:%Y%m%d}.nc"
        analysis_daily_path = output_dir / f"{file_prefix}_analysis_daily_fields_{valid_date:%Y%m%d}.nc"
        forecast_daily.to_netcdf(forecast_daily_path)
        analysis_daily.to_netcdf(analysis_daily_path)

        extent = get_region_definition(region_name).expanded_bounds
        forecast_maps = render_daily_maps(
            daily=forecast_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=f"{file_prefix}_forecast",
            extent=extent,
            map_label=f"Verification Forecast {region_name} ({forecast_model})",
            include_presentation=False,
        )
        analysis_maps = render_daily_maps(
            daily=analysis_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=f"{file_prefix}_analysis",
            extent=extent,
            map_label=f"Verification Analysis {region_name}",
            include_presentation=False,
        )
        diff_map_path = _write_difference_map(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )
        absolute_error_map_path = _write_absolute_error_map(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )
        category_disagreement_map_path = _write_category_disagreement_map(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )
        missed_high_comfort_map_path = _write_high_comfort_mask_map(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=file_prefix,
            mode="missed",
        )
        false_high_comfort_map_path = _write_high_comfort_mask_map(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            valid_date=valid_date,
            output_dir=output_dir,
            file_prefix=file_prefix,
            mode="false",
        )

        summary = _verification_summary(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            metadata={
                **metadata,
                "aggregation_policy": aggregation_policy,
                "aggregation_mode": aggregation_mode,
            },
            valid_date=valid_date,
        )
        component_metrics = _component_metrics(
            forecast_scored_hourly=forecast_scored_hourly,
            analysis_scored_hourly=analysis_scored_hourly,
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
        )
        summary.update(_component_summary_fields(component_metrics))
        summary_path = output_dir / f"{file_prefix}_summary_{valid_date:%Y%m%d}.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

        points_path = output_dir / f"{file_prefix}_points_{valid_date:%Y%m%d}.csv"
        _point_metrics(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            region_name=region_name,
        ).to_csv(points_path, index=False)

        component_metrics_path = output_dir / f"{file_prefix}_components_{valid_date:%Y%m%d}.csv"
        component_metrics.to_csv(component_metrics_path, index=False)
    except Exception:
        write_openmeteo_request_report(output_dir=output_dir, run_slug=request_report_slug)
        raise

    request_summary_path, request_detail_path = write_openmeteo_request_report(output_dir=output_dir, run_slug=request_report_slug)
    return {
        "forecast_daily_fields": forecast_daily_path,
        "analysis_daily_fields": analysis_daily_path,
        "forecast_score_map": forecast_maps["raw_map"],
        "analysis_score_map": analysis_maps["raw_map"],
        "score_difference_map": diff_map_path,
        "absolute_error_map": absolute_error_map_path,
        "category_disagreement_map": category_disagreement_map_path,
        "missed_high_comfort_map": missed_high_comfort_map_path,
        "false_high_comfort_map": false_high_comfort_map_path,
        "summary_csv": summary_path,
        "point_metrics_csv": points_path,
        "component_metrics_csv": component_metrics_path,
        "request_summary_csv": request_summary_path,
        "request_detail_csv": request_detail_path,
        "summary_record": summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare archived forecast Comfort Index against archive-analysis Comfort Index.")
    parser.add_argument("--date", required=True, help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--region", default=OPENMETEO_VERIFICATION_DEFAULT_REGION, choices=list_region_names())
    parser.add_argument("--mesh-profile", default="standard", help="Regional mesh profile. Default: standard.")
    parser.add_argument("--forecast-model", default=OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT)
    parser.add_argument("--analysis-model", default=OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT)
    parser.add_argument("--forecast-run-hour-utc", type=int, default=OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC)
    parser.add_argument("--forecast-lead-days", type=int, default=OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS)
    parser.add_argument(
        "--aggregation-policy",
        default="baseline",
        choices=list_verification_aggregation_policies(),
        help="Verification-only daily aggregation policy. Default: baseline.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    outputs = run_verification(
        valid_date=valid_date,
        region_name=args.region,
        output_dir=Path(args.output_dir),
        mesh_profile=args.mesh_profile,
        forecast_model=args.forecast_model,
        analysis_model=args.analysis_model,
        forecast_run_hour_utc=args.forecast_run_hour_utc,
        forecast_lead_days=args.forecast_lead_days,
        aggregation_policy=args.aggregation_policy,
    )
    print(f"Saved forecast score map: {outputs['forecast_score_map']}")
    print(f"Saved analysis score map: {outputs['analysis_score_map']}")
    print(f"Saved score difference map: {outputs['score_difference_map']}")
    print(f"Saved absolute error map: {outputs['absolute_error_map']}")
    print(f"Saved category disagreement map: {outputs['category_disagreement_map']}")
    print(f"Saved missed high comfort map: {outputs['missed_high_comfort_map']}")
    print(f"Saved false high comfort map: {outputs['false_high_comfort_map']}")
    print(f"Saved verification summary: {outputs['summary_csv']}")
    print(f"Saved point metrics: {outputs['point_metrics_csv']}")
    print(f"Saved component metrics: {outputs['component_metrics_csv']}")
    print(f"Saved Open-Meteo request summary: {outputs['request_summary_csv']}")
    print(f"Saved Open-Meteo request detail: {outputs['request_detail_csv']}")


if __name__ == "__main__":
    main()
