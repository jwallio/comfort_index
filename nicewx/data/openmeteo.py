"""Open-Meteo point forecast loader and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

from nicewx.config import (
    OPENMETEO_AIR_QUALITY_URL,
    OPENMETEO_AIR_QUALITY_VARS,
    OPENMETEO_CAPE_THUNDER_THRESHOLD,
    OPENMETEO_DEFAULT_MESH_PROFILE,
    OPENMETEO_FORECAST_URL,
    OPENMETEO_HOURLY_VARS,
    OPENMETEO_POP_THUNDER_THRESHOLD,
    OPENMETEO_REGIONAL_BATCH_SIZE,
    OPENMETEO_REQUEST_TIMEOUT_SECONDS,
    OPENMETEO_THUNDER_WEATHER_CODES,
    get_openmeteo_mesh_settings,
)
from nicewx.data.openmeteo_reliability import fetch_with_retries
from nicewx.mapping.regions import get_region_definition


def _dewpoint_from_temp_and_rh(temp_f: np.ndarray, relative_humidity_pct: np.ndarray) -> np.ndarray:
    """Derive dew point from temperature and relative humidity."""

    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    rh = np.clip(relative_humidity_pct, 1.0, 100.0)
    gamma = np.log(rh / 100.0) + (17.625 * temp_c) / (243.04 + temp_c)
    dewpoint_c = 243.04 * gamma / (17.625 - gamma)
    return dewpoint_c * 9.0 / 5.0 + 32.0


def _meters_to_miles(meters: np.ndarray) -> np.ndarray:
    return meters / 1609.344


def _fetch_json(base_url: str, query: dict[str, object]) -> dict[str, object] | list[dict[str, object]]:
    """Fetch JSON from an Open-Meteo endpoint."""

    def _request(base_url_inner: str, query_inner: dict[str, object]) -> dict[str, object] | list[dict[str, object]]:
        url = f"{base_url_inner}?{urlencode(query_inner, doseq=True)}"
        with urlopen(url, timeout=float(OPENMETEO_REQUEST_TIMEOUT_SECONDS)) as response:
            return json.loads(response.read().decode("utf-8"))

    return fetch_with_retries(base_url=base_url, query=query, request_func=_request)


def _build_hourly_data_array(values: list[float] | list[int] | list[bool] | None, size: int, fill_value: float = np.nan) -> np.ndarray:
    """Return a numeric array of the requested size, filling missing fields gracefully."""

    if values is None:
        return np.full(size, fill_value, dtype=float)
    return np.array(values, dtype=float)


def _expand_axis(values: np.ndarray) -> np.ndarray:
    """Expand a 1-D hourly series to point-grid shape."""

    return values[:, None, None]


def _regional_coordinate_values(min_value: float, max_value: float, step: float) -> list[float]:
    """Return inclusive regional mesh coordinates."""

    values = list(np.arange(min_value, max_value + 0.0001, step))
    if not np.isclose(values[-1], max_value):
        values.append(max_value)
    return [round(value, 4) for value in values]


def _payload_list(payload: dict[str, object] | list[dict[str, object]]) -> list[dict[str, object]]:
    """Normalize single or multi-location responses to a list."""

    return payload if isinstance(payload, list) else [payload]


def normalize_openmeteo_forecast_response(payload: dict[str, object], requested_lat: float, requested_lon: float) -> xr.Dataset:
    """Normalize an Open-Meteo forecast payload into the project point schema."""

    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict):
        raise ValueError("Open-Meteo forecast payload missing hourly data.")
    time_values = hourly.get("time")
    if not isinstance(time_values, list) or not time_values:
        raise ValueError("Open-Meteo forecast payload missing hourly.time values.")

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
    pop_pct = _build_hourly_data_array(hourly.get("precipitation_probability"), count, fill_value=0.0)
    qpf_in = _build_hourly_data_array(hourly.get("precipitation"), count, fill_value=0.0)
    weather_code = _build_hourly_data_array(hourly.get("weather_code"), count, fill_value=np.nan)
    visibility_mi = _meters_to_miles(_build_hourly_data_array(hourly.get("visibility"), count))
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
            "cape": (("time", "lat", "lon"), _expand_axis(cape).astype(np.float32)),
        },
        coords={"time": times, "lat": [resolved_lat], "lon": [resolved_lon]},
        attrs={
            "source": "openmeteo_forecast",
            "requested_lat": requested_lat,
            "requested_lon": requested_lon,
            "timezone": payload.get("timezone", "auto"),
            "normalization_notes": (
                "Temperature requested in Fahrenheit, wind in mph, precipitation in inches, "
                "visibility converted from meters to miles, thunder derived from weather_code and CAPE proxy."
            ),
        },
    )
    if not np.isnan(visibility_mi).all():
        dataset["visibility_mi"] = (("time", "lat", "lon"), _expand_axis(visibility_mi).astype(np.float32))
    return dataset


def merge_openmeteo_air_quality(dataset: xr.Dataset, payload: dict[str, object]) -> xr.Dataset:
    """Merge optional Open-Meteo air-quality payload fields into a normalized dataset."""

    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict):
        return dataset

    merged = dataset.copy()
    if "us_aqi" in hourly:
        aqi = np.array(hourly["us_aqi"], dtype=float)
        merged["aqi"] = (("time", "lat", "lon"), aqi[:, None, None].astype(np.float32))
    if "pm2_5" in hourly:
        pm25 = np.array(hourly["pm2_5"], dtype=float)
        merged["pm25"] = (("time", "lat", "lon"), pm25[:, None, None].astype(np.float32))
    return merged


@dataclass
class OpenMeteoPointLoader:
    """Fetch and normalize Open-Meteo hourly point forecasts."""

    lat: float
    lon: float

    def _fetch_json(self, base_url: str, query: dict[str, object]) -> dict[str, object]:
        payload = _fetch_json(base_url, query)
        if isinstance(payload, list):
            raise ValueError("Expected a single-location response for point loading.")
        return payload

    def load_hourly_grid(self, valid_date: date) -> xr.Dataset:
        forecast_query = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": valid_date.isoformat(),
            "end_date": valid_date.isoformat(),
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "hourly": list(OPENMETEO_HOURLY_VARS),
        }
        forecast_payload = self._fetch_json(OPENMETEO_FORECAST_URL, forecast_query)
        dataset = normalize_openmeteo_forecast_response(
            forecast_payload,
            requested_lat=self.lat,
            requested_lon=self.lon,
        )

        air_quality_query = {
            "latitude": float(dataset["lat"].values.squeeze()),
            "longitude": float(dataset["lon"].values.squeeze()),
            "start_date": valid_date.isoformat(),
            "end_date": valid_date.isoformat(),
            "timezone": "auto",
            "hourly": list(OPENMETEO_AIR_QUALITY_VARS),
        }
        try:
            air_payload = self._fetch_json(OPENMETEO_AIR_QUALITY_URL, air_quality_query)
        except Exception:
            return dataset
        return merge_openmeteo_air_quality(dataset, air_payload)


def assemble_point_datasets_to_grid(
    point_datasets: dict[tuple[float, float], xr.Dataset],
    lat_values: list[float],
    lon_values: list[float],
    region_name: str,
) -> xr.Dataset:
    """Assemble normalized point datasets into a gridded regional mesh dataset."""

    if not point_datasets:
        raise ValueError("At least one point dataset is required to assemble a regional grid.")

    first_dataset = next(iter(point_datasets.values()))
    times = first_dataset["time"].values
    variables = sorted({variable for dataset in point_datasets.values() for variable in dataset.data_vars})
    data_vars: dict[str, tuple[tuple[str, str, str], np.ndarray]] = {}

    for variable in variables:
        sample = next(dataset[variable] for dataset in point_datasets.values() if variable in dataset)
        dtype = bool if sample.dtype == bool else np.float32
        fill_value = False if dtype == bool else np.nan
        values = np.full((len(times), len(lat_values), len(lon_values)), fill_value, dtype=dtype)
        for lat_index, lat in enumerate(lat_values):
            for lon_index, lon in enumerate(lon_values):
                dataset = point_datasets.get((lat, lon))
                if dataset is None:
                    continue
                if variable not in dataset:
                    continue
                values[:, lat_index, lon_index] = dataset[variable].values.squeeze().astype(dtype)
        data_vars[variable] = (("time", "lat", "lon"), values)

    assembled = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "lat": lat_values, "lon": lon_values},
        attrs={
            "source": "openmeteo_regional_mesh",
            "region_name": region_name,
            "mesh_note": "Coarse regional mesh assembled from repeated Open-Meteo point forecasts.",
        },
    )
    return assembled


@dataclass
class OpenMeteoRegionalMeshLoader:
    """Pilot coarse regional mesh loader using batched Open-Meteo point forecasts."""

    region_name: str
    mesh_profile: str = OPENMETEO_DEFAULT_MESH_PROFILE

    def load_hourly_grid(self, valid_date: date) -> xr.Dataset:
        region = get_region_definition(self.region_name)
        settings = get_openmeteo_mesh_settings(self.region_name, self.mesh_profile)
        lat_values = _regional_coordinate_values(region.expanded_bounds[2], region.expanded_bounds[3], float(settings["lat_step"]))
        lon_values = _regional_coordinate_values(region.expanded_bounds[0], region.expanded_bounds[1], float(settings["lon_step"]))
        coordinate_pairs = [(lat, lon) for lat in lat_values for lon in lon_values]

        point_datasets: dict[tuple[float, float], xr.Dataset] = {}
        timezone = str(settings.get("timezone", "GMT"))

        for start in range(0, len(coordinate_pairs), OPENMETEO_REGIONAL_BATCH_SIZE):
            batch = coordinate_pairs[start : start + OPENMETEO_REGIONAL_BATCH_SIZE]
            query = {
                "latitude": ",".join(str(lat) for lat, _ in batch),
                "longitude": ",".join(str(lon) for _, lon in batch),
                "start_date": valid_date.isoformat(),
                "end_date": valid_date.isoformat(),
                "timezone": timezone,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "hourly": list(OPENMETEO_HOURLY_VARS),
            }
            payloads = _payload_list(_fetch_json(OPENMETEO_FORECAST_URL, query))
            if len(payloads) != len(batch):
                raise ValueError("Open-Meteo regional batch response size did not match requested mesh points.")

            for (requested_lat, requested_lon), payload in zip(batch, payloads, strict=False):
                point_datasets[(requested_lat, requested_lon)] = normalize_openmeteo_forecast_response(
                    payload,
                    requested_lat=requested_lat,
                    requested_lon=requested_lon,
                )

        dataset = assemble_point_datasets_to_grid(
            point_datasets=point_datasets,
            lat_values=lat_values,
            lon_values=lon_values,
            region_name=self.region_name,
        )
        dataset.attrs["mesh_lat_step"] = float(settings["lat_step"])
        dataset.attrs["mesh_lon_step"] = float(settings["lon_step"])
        dataset.attrs["mesh_point_count"] = len(coordinate_pairs)
        dataset.attrs["mesh_profile"] = self.mesh_profile
        dataset.attrs["timezone"] = timezone
        return dataset
