"""Open-Meteo archived forecast and archive-analysis loaders for verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xarray as xr

from comfortwx.config import (
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_CAPE_THUNDER_THRESHOLD,
    OPENMETEO_POP_THUNDER_THRESHOLD,
    OPENMETEO_REGIONAL_BATCH_SIZE,
    OPENMETEO_SINGLE_RUN_URL,
    OPENMETEO_THUNDER_WEATHER_CODES,
    OPENMETEO_VERIFICATION_ANALYSIS_HOURLY_VARS,
    OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_ANALYSIS_POP_PROXY_QPF_FULL_IN,
    OPENMETEO_VERIFICATION_FORECAST_DAYS,
    OPENMETEO_VERIFICATION_FORECAST_HOURLY_VARS,
    OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS,
    OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_FORECAST_SHORT_LEAD_MODEL,
    OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC,
    get_openmeteo_mesh_settings,
)
from comfortwx.data.openmeteo import (
    _build_hourly_data_array,
    _dewpoint_from_temp_and_rh,
    _expand_axis,
    _fetch_json,
    _payload_list,
    _regional_coordinate_values,
    assemble_point_datasets_to_grid,
)
from comfortwx.mapping.regions import get_region_definition


def _visibility_to_miles(values: np.ndarray, unit: str | None) -> np.ndarray:
    normalized = (unit or "").strip().lower()
    if normalized in {"m", "meter", "meters"}:
        return values / 1609.344
    if normalized in {"ft", "feet"}:
        return values / 5280.0
    return np.full(values.shape, np.nan, dtype=float)


def _derive_pop_proxy_from_qpf(qpf_in: np.ndarray) -> np.ndarray:
    return np.clip((np.clip(qpf_in, 0.0, None) / OPENMETEO_VERIFICATION_ANALYSIS_POP_PROXY_QPF_FULL_IN) * 100.0, 0.0, 100.0)


def _normalize_openmeteo_verification_payload(
    payload: dict[str, object],
    *,
    requested_lat: float,
    requested_lon: float,
    source_label: str,
    derive_pop_proxy: bool,
) -> xr.Dataset:
    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict):
        raise ValueError("Verification payload missing hourly data.")
    hourly_units = payload.get("hourly_units", {})
    if not isinstance(hourly_units, dict):
        hourly_units = {}

    time_values = hourly.get("time")
    if not isinstance(time_values, list) or not time_values:
        raise ValueError("Verification payload missing hourly.time values.")

    times = pd.to_datetime(time_values)
    count = len(times)
    temp_f = _build_hourly_data_array(hourly.get("temperature_2m"), count)
    dewpoint_values = hourly.get("dew_point_2m")
    relative_humidity = _build_hourly_data_array(hourly.get("relative_humidity_2m"), count)
    dewpoint_f = (
        _build_hourly_data_array(dewpoint_values, count)
        if dewpoint_values is not None
        else _dewpoint_from_temp_and_rh(temp_f=temp_f, relative_humidity_pct=relative_humidity)
    )
    wind_mph = _build_hourly_data_array(hourly.get("wind_speed_10m"), count, fill_value=0.0)
    gust_values = hourly.get("wind_gusts_10m")
    gust_mph = np.array(gust_values, dtype=float) if gust_values is not None else wind_mph.copy()
    cloud_pct = _build_hourly_data_array(hourly.get("cloud_cover"), count, fill_value=0.0)
    qpf_in = _build_hourly_data_array(hourly.get("precipitation"), count, fill_value=0.0)

    pop_raw = np.array(hourly.get("precipitation_probability", [np.nan] * count), dtype=float)
    if derive_pop_proxy or np.isnan(pop_raw).all():
        pop_pct = _derive_pop_proxy_from_qpf(qpf_in)
    else:
        pop_pct = np.clip(pop_raw, 0.0, 100.0)

    weather_code = _build_hourly_data_array(hourly.get("weather_code"), count, fill_value=np.nan)
    visibility_values = _build_hourly_data_array(hourly.get("visibility"), count)
    visibility_mi = _visibility_to_miles(visibility_values, hourly_units.get("visibility"))
    cape = _build_hourly_data_array(hourly.get("cape"), count)

    thunder = np.isin(weather_code.astype(int), np.array(OPENMETEO_THUNDER_WEATHER_CODES))
    thunder = thunder | ((cape >= OPENMETEO_CAPE_THUNDER_THRESHOLD) & (pop_pct >= OPENMETEO_POP_THUNDER_THRESHOLD))

    resolved_lat = float(payload.get("latitude", requested_lat))
    resolved_lon = float(payload.get("longitude", requested_lon))
    dataset = xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), _expand_axis(temp_f).astype(np.float32)),
            "dewpoint_f": (("time", "lat", "lon"), _expand_axis(dewpoint_f).astype(np.float32)),
            "wind_mph": (("time", "lat", "lon"), _expand_axis(wind_mph).astype(np.float32)),
            "gust_mph": (("time", "lat", "lon"), _expand_axis(gust_mph).astype(np.float32)),
            "cloud_pct": (("time", "lat", "lon"), _expand_axis(np.clip(cloud_pct, 0.0, 100.0)).astype(np.float32)),
            "pop_pct": (("time", "lat", "lon"), _expand_axis(np.clip(pop_pct, 0.0, 100.0)).astype(np.float32)),
            "qpf_in": (("time", "lat", "lon"), _expand_axis(np.clip(qpf_in, 0.0, None)).astype(np.float32)),
            "weather_code": (("time", "lat", "lon"), _expand_axis(weather_code).astype(np.float32)),
            "thunder": (("time", "lat", "lon"), _expand_axis(thunder).astype(bool)),
        },
        coords={"time": times, "lat": [resolved_lat], "lon": [resolved_lon]},
        attrs={
            "source": source_label,
            "requested_lat": requested_lat,
            "requested_lon": requested_lon,
            "timezone": payload.get("timezone", "auto"),
        },
    )
    if not np.isnan(cape).all():
        dataset["cape"] = (("time", "lat", "lon"), _expand_axis(cape).astype(np.float32))
    if not np.isnan(visibility_mi).all():
        dataset["visibility_mi"] = (("time", "lat", "lon"), _expand_axis(visibility_mi).astype(np.float32))
    return dataset


def _subset_to_valid_local_day(dataset: xr.Dataset, valid_date: date) -> xr.Dataset:
    return dataset.sel(time=dataset["time"].dt.date == valid_date)


def _forecast_run_timestamp(valid_date: date, *, run_hour_utc: int, lead_days: int) -> str:
    run_datetime = datetime.combine(valid_date - timedelta(days=lead_days), datetime.min.time(), tzinfo=timezone.utc)
    run_datetime = run_datetime.replace(hour=run_hour_utc)
    return run_datetime.strftime("%Y-%m-%dT%H:%M")


def resolve_openmeteo_verification_forecast_model(
    *,
    requested_model: str,
    forecast_lead_days: int,
) -> str:
    normalized_model = requested_model.strip().lower()
    if normalized_model == OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT and forecast_lead_days <= 1:
        return OPENMETEO_VERIFICATION_FORECAST_SHORT_LEAD_MODEL
    return normalized_model


def _verification_forecast_days(
    *,
    resolved_forecast_model: str,
    forecast_lead_days: int,
) -> int:
    if resolved_forecast_model == OPENMETEO_VERIFICATION_FORECAST_SHORT_LEAD_MODEL and forecast_lead_days <= 1:
        return max(OPENMETEO_VERIFICATION_FORECAST_DAYS, 2)
    return max(OPENMETEO_VERIFICATION_FORECAST_DAYS, forecast_lead_days + 2)


@dataclass
class OpenMeteoVerificationRegionalLoader:
    """Archived forecast vs historical analysis regional mesh loader."""

    region_name: str
    mesh_profile: str = "standard"
    forecast_model: str = OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT
    analysis_model: str = OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT
    forecast_run_hour_utc: int = OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC
    forecast_lead_days: int = OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS

    def load_pair(self, valid_date: date) -> tuple[xr.Dataset, xr.Dataset, dict[str, object]]:
        region = get_region_definition(self.region_name)
        settings = get_openmeteo_mesh_settings(self.region_name, self.mesh_profile)
        lat_values = _regional_coordinate_values(region.expanded_bounds[2], region.expanded_bounds[3], float(settings["lat_step"]))
        lon_values = _regional_coordinate_values(region.expanded_bounds[0], region.expanded_bounds[1], float(settings["lon_step"]))
        coordinate_pairs = [(lat, lon) for lat in lat_values for lon in lon_values]
        timezone_name = str(settings.get("timezone", "GMT"))
        run_timestamp = _forecast_run_timestamp(
            valid_date,
            run_hour_utc=self.forecast_run_hour_utc,
            lead_days=self.forecast_lead_days,
        )
        resolved_forecast_model = resolve_openmeteo_verification_forecast_model(
            requested_model=self.forecast_model,
            forecast_lead_days=self.forecast_lead_days,
        )
        forecast_days = _verification_forecast_days(
            resolved_forecast_model=resolved_forecast_model,
            forecast_lead_days=self.forecast_lead_days,
        )

        forecast_point_datasets: dict[tuple[float, float], xr.Dataset] = {}
        analysis_point_datasets: dict[tuple[float, float], xr.Dataset] = {}

        for start in range(0, len(coordinate_pairs), OPENMETEO_REGIONAL_BATCH_SIZE):
            batch = coordinate_pairs[start : start + OPENMETEO_REGIONAL_BATCH_SIZE]
            forecast_query = {
                "latitude": ",".join(str(lat) for lat, _ in batch),
                "longitude": ",".join(str(lon) for _, lon in batch),
                "run": run_timestamp,
                "forecast_days": forecast_days,
                "timezone": timezone_name,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "models": resolved_forecast_model,
                "hourly": list(OPENMETEO_VERIFICATION_FORECAST_HOURLY_VARS),
            }
            analysis_query = {
                "latitude": ",".join(str(lat) for lat, _ in batch),
                "longitude": ",".join(str(lon) for _, lon in batch),
                "start_date": valid_date.isoformat(),
                "end_date": valid_date.isoformat(),
                "timezone": timezone_name,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "models": self.analysis_model,
                "hourly": list(OPENMETEO_VERIFICATION_ANALYSIS_HOURLY_VARS),
            }

            forecast_payloads = _payload_list(_fetch_json(OPENMETEO_SINGLE_RUN_URL, forecast_query))
            analysis_payloads = _payload_list(_fetch_json(OPENMETEO_ARCHIVE_URL, analysis_query))
            if len(forecast_payloads) != len(batch) or len(analysis_payloads) != len(batch):
                raise ValueError("Verification batch response size did not match requested mesh points.")

            for (requested_lat, requested_lon), forecast_payload, analysis_payload in zip(
                batch, forecast_payloads, analysis_payloads, strict=False
            ):
                forecast_dataset = _normalize_openmeteo_verification_payload(
                    forecast_payload,
                    requested_lat=requested_lat,
                    requested_lon=requested_lon,
                    source_label=f"openmeteo_single_run:{resolved_forecast_model}",
                    derive_pop_proxy=False,
                )
                forecast_point_datasets[(requested_lat, requested_lon)] = _subset_to_valid_local_day(forecast_dataset, valid_date)
                analysis_point_datasets[(requested_lat, requested_lon)] = _normalize_openmeteo_verification_payload(
                    analysis_payload,
                    requested_lat=requested_lat,
                    requested_lon=requested_lon,
                    source_label=f"openmeteo_archive:{self.analysis_model}",
                    derive_pop_proxy=True,
                )

        forecast_grid = assemble_point_datasets_to_grid(
            point_datasets=forecast_point_datasets,
            lat_values=lat_values,
            lon_values=lon_values,
            region_name=self.region_name,
        )
        analysis_grid = assemble_point_datasets_to_grid(
            point_datasets=analysis_point_datasets,
            lat_values=lat_values,
            lon_values=lon_values,
            region_name=self.region_name,
        )
        forecast_grid.attrs.update(
            {
                "source": f"openmeteo_single_run:{resolved_forecast_model}",
                "verification_region": self.region_name,
                "verification_run_timestamp_utc": run_timestamp,
                "mesh_profile": self.mesh_profile,
            }
        )
        analysis_grid.attrs.update(
            {
                "source": f"openmeteo_archive:{self.analysis_model}",
                "verification_region": self.region_name,
                "verification_valid_date": valid_date.isoformat(),
                "mesh_profile": self.mesh_profile,
            }
        )
        metadata = {
            "region_name": self.region_name,
            "mesh_profile": self.mesh_profile,
            "forecast_model_requested": self.forecast_model,
            "forecast_model": resolved_forecast_model,
            "analysis_model": self.analysis_model,
            "forecast_run_timestamp_utc": run_timestamp,
            "forecast_lead_days": self.forecast_lead_days,
            "mesh_point_count": len(coordinate_pairs),
            "timezone": timezone_name,
        }
        return forecast_grid, analysis_grid, metadata
