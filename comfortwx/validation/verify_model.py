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
    OPENMETEO_VERIFICATION_DEFAULT_REGION,
    OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS,
    OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC,
    OPENMETEO_VERIFICATION_SAMPLE_POINT_NAMES,
    OUTPUT_DIR,
)
from comfortwx.data.openmeteo_reliability import (
    openmeteo_request_context,
    reset_openmeteo_request_records,
    write_openmeteo_request_report,
)
from comfortwx.data.openmeteo_verification import OpenMeteoVerificationRegionalLoader
from comfortwx.mapping.plotting import render_daily_maps
from comfortwx.mapping.regions import get_region_definition, list_region_names
from comfortwx.scoring.categories import category_name_from_index
from comfortwx.scoring.daily import aggregate_daily_scores
from comfortwx.scoring.hourly import score_hourly_dataset


def _category_counts(prefix: str, daily: xr.Dataset) -> dict[str, object]:
    values = daily["category_index"].values.astype(int).ravel()
    records: dict[str, object] = {}
    for index in sorted(np.unique(values)):
        label = category_name_from_index(int(index)).lower()
        records[f"{prefix}_{label}_count"] = int(np.sum(values == index))
    return records


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

    summary = {
        "valid_date": valid_date.isoformat(),
        "region_name": metadata["region_name"],
        "forecast_source": f"Open-Meteo Single Runs ({metadata['forecast_model']})",
        "analysis_source": f"Open-Meteo Archive ({metadata['analysis_model']})",
        "forecast_run_timestamp_utc": metadata["forecast_run_timestamp_utc"],
        "mesh_profile": metadata["mesh_profile"],
        "mesh_point_count": metadata["mesh_point_count"],
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


def run_verification(
    *,
    valid_date: date,
    region_name: str,
    output_dir: Path,
    mesh_profile: str,
    forecast_model: str,
    forecast_run_hour_utc: int,
    forecast_lead_days: int,
    workflow_name: str = "verification_model",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"comfortwx_verify_{region_name}_{forecast_model}"
    request_report_slug = f"{file_prefix}_{valid_date:%Y%m%d}"
    reset_openmeteo_request_records()
    try:
        loader = OpenMeteoVerificationRegionalLoader(
            region_name=region_name,
            mesh_profile=mesh_profile,
            forecast_model=forecast_model,
            forecast_run_hour_utc=forecast_run_hour_utc,
            forecast_lead_days=forecast_lead_days,
        )
        with openmeteo_request_context(
            workflow=workflow_name,
            label=f"region={region_name};date={valid_date.isoformat()}",
            run_slug=request_report_slug,
        ):
            forecast_hourly, analysis_hourly, metadata = loader.load_pair(valid_date)
        forecast_daily = aggregate_daily_scores(score_hourly_dataset(forecast_hourly))
        analysis_daily = aggregate_daily_scores(score_hourly_dataset(analysis_hourly))

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

        summary = _verification_summary(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            metadata=metadata,
            valid_date=valid_date,
        )
        summary_path = output_dir / f"{file_prefix}_summary_{valid_date:%Y%m%d}.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

        points_path = output_dir / f"{file_prefix}_points_{valid_date:%Y%m%d}.csv"
        _point_metrics(
            forecast_daily=forecast_daily,
            analysis_daily=analysis_daily,
            region_name=region_name,
        ).to_csv(points_path, index=False)
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
        "summary_csv": summary_path,
        "point_metrics_csv": points_path,
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
    parser.add_argument("--forecast-run-hour-utc", type=int, default=OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC)
    parser.add_argument("--forecast-lead-days", type=int, default=OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS)
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
        forecast_run_hour_utc=args.forecast_run_hour_utc,
        forecast_lead_days=args.forecast_lead_days,
    )
    print(f"Saved forecast score map: {outputs['forecast_score_map']}")
    print(f"Saved analysis score map: {outputs['analysis_score_map']}")
    print(f"Saved score difference map: {outputs['score_difference_map']}")
    print(f"Saved verification summary: {outputs['summary_csv']}")
    print(f"Saved point metrics: {outputs['point_metrics_csv']}")
    print(f"Saved Open-Meteo request summary: {outputs['request_summary_csv']}")
    print(f"Saved Open-Meteo request detail: {outputs['request_detail_csv']}")


if __name__ == "__main__":
    main()
