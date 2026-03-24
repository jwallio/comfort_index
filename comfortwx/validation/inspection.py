"""Helpers for point inspection and diagnostic exports."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import xarray as xr

from comfortwx.scoring.categories import category_name_from_index
from comfortwx.validation.explain import explain_point_series


def select_nearest_point(dataset: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """Return the nearest point from a gridded dataset."""

    return dataset.sel(lat=lat, lon=lon, method="nearest")


def point_hourly_breakdown_dataframe(scored_point: xr.Dataset) -> pd.DataFrame:
    """Return an hourly component breakdown for a single point."""

    preferred_columns = [
        "temp_f",
        "dewpoint_f",
        "wind_mph",
        "gust_mph",
        "cloud_pct",
        "pop_pct",
        "qpf_in",
        "thunder",
        "aqi",
        "pm25",
        "smoke",
        "visibility_mi",
        "temp_score",
        "dewpoint_score",
        "wind_score",
        "cloud_score",
        "precip_score",
        "air_quality_penalty",
        "visibility_penalty",
        "interaction_adjustment",
        "hazard_penalty",
        "hourly_score_pre_cap",
        "hourly_score_cap",
        "hourly_score",
    ]
    available_columns = [column for column in preferred_columns if column in scored_point]
    frame = scored_point[available_columns].to_dataframe().reset_index()
    return frame


def point_daily_summary_record(daily_point: xr.Dataset, explanation: str) -> dict[str, object]:
    """Return a compact daily summary record for a single point."""

    return {
        "best_3hr": round(float(daily_point["best_3hr"].values), 1),
        "best_6hr": round(float(daily_point["best_6hr"].values), 1),
        "daytime_weighted_mean": round(float(daily_point["daytime_weighted_mean"].values), 1),
        "reliability_score": round(float(daily_point["reliability_score"].values), 1),
        "disruption_penalty": round(float(daily_point["disruption_penalty"].values), 1),
        "daily_score": round(float(daily_point["daily_score"].values), 1),
        "category": category_name_from_index(int(daily_point["category_index"].values)),
        "prime_measurable_precip_fraction": round(float(daily_point["prime_measurable_precip_fraction"].values), 3),
        "prime_heavy_precip_fraction": round(float(daily_point["prime_heavy_precip_fraction"].values), 3),
        "prime_thunder_fraction": round(float(daily_point["prime_thunder_fraction"].values), 3),
        "prime_gusty_fraction": round(float(daily_point["prime_gusty_fraction"].values), 3),
        "prime_score_drop_fraction": round(float(daily_point["prime_score_drop_fraction"].values), 3),
        "prime_tail_clean_fraction": round(float(daily_point["prime_tail_clean_fraction"].values), 3),
        "daytime_mean_dewpoint": round(float(daily_point["daytime_mean_dewpoint"].values), 1),
        "daytime_mean_gust": round(float(daily_point["daytime_mean_gust"].values), 1),
        "explanation": explanation,
    }


def inspect_point(
    scored_hourly: xr.Dataset,
    daily: xr.Dataset,
    lat: float,
    lon: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str, tuple[float, float]]:
    """Return hourly breakdown, daily summary, explanation, and resolved coords."""

    scored_point = select_nearest_point(scored_hourly, lat=lat, lon=lon)
    daily_point = select_nearest_point(daily, lat=lat, lon=lon)
    actual_lat = float(scored_point["lat"].values)
    actual_lon = float(scored_point["lon"].values)
    explanation = explain_point_series(scored_point=scored_point, daily_point=daily_point)
    hourly_frame = point_hourly_breakdown_dataframe(scored_point)
    summary_frame = pd.DataFrame([point_daily_summary_record(daily_point, explanation)])
    return hourly_frame, summary_frame, explanation, (actual_lat, actual_lon)


def export_point_inspection(
    scored_hourly: xr.Dataset,
    daily: xr.Dataset,
    valid_date: date,
    output_dir: Path,
    lat: float,
    lon: float,
) -> dict[str, object]:
    """Write hourly and daily diagnostics for the nearest grid point."""

    hourly_frame, summary_frame, explanation, (actual_lat, actual_lon) = inspect_point(
        scored_hourly=scored_hourly,
        daily=daily,
        lat=lat,
        lon=lon,
    )
    base_name = f"comfortwx_point_{actual_lat:.2f}_{actual_lon:.2f}_{valid_date:%Y%m%d}"
    hourly_path = output_dir / f"{base_name}_hourly.csv"
    summary_path = output_dir / f"{base_name}_summary.csv"
    hourly_frame.to_csv(hourly_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    return {
        "hourly_csv": hourly_path,
        "summary_csv": summary_path,
        "explanation": explanation,
        "resolved_lat": actual_lat,
        "resolved_lon": actual_lon,
    }
