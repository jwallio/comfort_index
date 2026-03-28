"""Archived NDFD forecast scaffold for verification experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

from comfortwx.config import (
    NDFD_ARCHIVE_BASE_URL,
    NDFD_FORECAST_CACHE_DIR,
    NDFD_FORECAST_CACHE_VERSION,
    NDFD_FORECAST_POP_PROXY_QPF_FULL_IN,
    NDFD_FORECAST_SUPPORTED_REGIONS,
    NDFD_FORECAST_TIMEOUT_SECONDS,
    OPENMETEO_VERIFICATION_FORECAST_MODEL_NWS_NDFD_HOURLY,
    OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC,
    get_openmeteo_mesh_settings,
)
from comfortwx.data.noaa_analysis import _nearest_point_lookup, _normalize_longitudes, _utc_hour_schedule
from comfortwx.data.openmeteo import _regional_coordinate_values, assemble_point_datasets_to_grid
from comfortwx.mapping.regions import get_region_definition

_K_TO_F_OFFSET = 459.67
_MS_TO_MPH = 2.2369362920544
_CATALOG_ENTRY_PATTERN = re.compile(
    r'dataset=(?P<dataset_id>[^"]+)"><code>(?P<label>[^<]+) run on (?P<run_ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) '
    r'WMO: (?P<wmo>[A-Z0-9]+) CC: (?P<center>[A-Z0-9]+)</code>',
    re.IGNORECASE,
)

_FIELD_SPECS: dict[str, dict[str, str]] = {
    "temp_f": {"label": "Surface Temperature", "converter": "kelvin_to_fahrenheit"},
    "dewpoint_f": {"label": "Dew Point", "converter": "kelvin_to_fahrenheit"},
    "wind_mph": {"label": "Wind Speed", "converter": "ms_to_mph"},
    "gust_mph": {"label": "Wind Speed Gust At The Surface", "converter": "ms_to_mph"},
    "cloud_pct": {"label": "Total Cloud Cover", "converter": "identity"},
    "pop_pct": {"label": "Probability of Precipitation (12 Hour)", "converter": "identity"},
}


def _kelvin_to_fahrenheit(values: np.ndarray) -> np.ndarray:
    return values * 9.0 / 5.0 - _K_TO_F_OFFSET


def _ms_to_mph(values: np.ndarray) -> np.ndarray:
    return values * _MS_TO_MPH


def _identity(values: np.ndarray) -> np.ndarray:
    return values


def _pop_to_qpf_proxy(pop_pct: np.ndarray) -> np.ndarray:
    return np.clip(np.array(pop_pct, dtype=float), 0.0, 100.0) / 100.0 * NDFD_FORECAST_POP_PROXY_QPF_FULL_IN


def _catalog_url(run_date: date) -> str:
    return (
        f"{NDFD_ARCHIVE_BASE_URL}/catalog/model-ndfd-file_kwbn/access/"
        f"{run_date:%Y%m}/{run_date:%Y%m%d}/catalog.html"
    )


def _file_url(dataset_id: str) -> str:
    file_server_id = dataset_id
    if file_server_id.startswith("NDFD_kwbn/"):
        file_server_id = "model-ndfd-file_kwbn/" + file_server_id.removeprefix("NDFD_kwbn/")
    return f"{NDFD_ARCHIVE_BASE_URL}/fileServer/{file_server_id}"


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=float(NDFD_FORECAST_TIMEOUT_SECONDS)) as response:
        return response.read().decode("utf-8", "replace")


def _catalog_entries(run_date: date) -> list[dict[str, object]]:
    html = _fetch_text(_catalog_url(run_date))
    entries: list[dict[str, object]] = []
    for match in _CATALOG_ENTRY_PATTERN.finditer(html):
        entries.append(
            {
                "dataset_id": match.group("dataset_id"),
                "label": match.group("label"),
                "run_timestamp_utc": datetime.strptime(match.group("run_ts"), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc),
                "wmo_code": match.group("wmo"),
                "center_code": match.group("center"),
            }
        )
    return entries


def _select_catalog_entry(
    entries: list[dict[str, object]],
    *,
    label: str,
    target_run_timestamp_utc: datetime,
    region_name: str | None = None,
) -> dict[str, object]:
    candidates = [entry for entry in entries if str(entry["label"]).startswith(label)]
    if not candidates:
        raise FileNotFoundError(f"No NDFD archive entry found for '{label}'.")
    normalized_region = (region_name or "").strip().lower()
    if normalized_region == "west_coast":
        national = [entry for entry in candidates if str(entry["wmo_code"]).endswith("UZ98")]
        if not national:
            national = [entry for entry in candidates if str(entry["wmo_code"]).endswith("UZ97")]
        candidates = national or candidates
    else:
        preferred = [entry for entry in candidates if str(entry["wmo_code"]).endswith("88")]
        candidates = preferred or candidates
    earlier = [entry for entry in candidates if entry["run_timestamp_utc"] <= target_run_timestamp_utc]
    if earlier:
        return max(earlier, key=lambda entry: entry["run_timestamp_utc"])
    return min(candidates, key=lambda entry: abs(entry["run_timestamp_utc"] - target_run_timestamp_utc))


def _cache_path(*, dataset_id: str) -> Path:
    filename = dataset_id.split("/")[-1]
    return NDFD_FORECAST_CACHE_DIR / NDFD_FORECAST_CACHE_VERSION / f"{filename}.grb2"


def _download_cached_file(*, dataset_id: str) -> Path:
    cache_path = _cache_path(dataset_id=dataset_id)
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(_file_url(dataset_id), timeout=float(NDFD_FORECAST_TIMEOUT_SECONDS)) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _open_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def _dataset_var_name(dataset: xr.Dataset) -> str:
    if not dataset.data_vars:
        raise ValueError("NDFD dataset had no data variables.")
    return next(iter(dataset.data_vars))


def _dataset_lookup(dataset: xr.Dataset, coordinate_pairs: list[tuple[float, float]], bounds: tuple[float, float, float, float]) -> dict[tuple[float, float], object]:
    latitude = np.asarray(dataset["latitude"].values)
    longitude = _normalize_longitudes(np.asarray(dataset["longitude"].values))
    if latitude.ndim == 1 and longitude.ndim == 1 and latitude.shape == longitude.shape:
        lookup: dict[tuple[float, float], int] = {}
        for requested_lat, requested_lon in coordinate_pairs:
            distance = (latitude - requested_lat) ** 2 + (longitude - requested_lon) ** 2
            lookup[(requested_lat, requested_lon)] = int(np.nanargmin(distance))
        return lookup
    if latitude.ndim == 2 and longitude.ndim == 2:
        return _nearest_point_lookup(
            latitude=latitude,
            longitude=longitude,
            coordinate_pairs=coordinate_pairs,
            bounds=bounds,
        )
    raise ValueError("Unsupported NDFD latitude/longitude layout.")


def _nearest_time_index(valid_times: pd.DatetimeIndex, target_utc: datetime) -> int:
    target_naive = pd.Timestamp(target_utc.replace(tzinfo=None))
    delta = np.abs(valid_times - target_naive)
    return int(np.argmin(delta))


def _dataset_value(dataset: xr.Dataset, *, point_index: object, utc_dt: datetime, converter: str) -> float:
    var_name = _dataset_var_name(dataset)
    valid_times = pd.to_datetime(dataset["valid_time"].values)
    time_index = _nearest_time_index(valid_times, utc_dt)
    values = dataset[var_name].values
    if isinstance(point_index, tuple):
        raw_value = values[time_index, point_index[0], point_index[1]]
    else:
        raw_value = values[time_index, int(point_index)]
    if not np.isfinite(raw_value):
        return np.nan
    converter_func = {
        "kelvin_to_fahrenheit": _kelvin_to_fahrenheit,
        "ms_to_mph": _ms_to_mph,
        "identity": _identity,
    }[converter]
    return float(converter_func(np.array(raw_value, dtype=float)))


def _point_dataset(
    *,
    requested_lat: float,
    requested_lon: float,
    local_times: list[datetime],
    temp_f: list[float],
    dewpoint_f: list[float],
    wind_mph: list[float],
    gust_mph: list[float],
    cloud_pct: list[float],
    pop_pct: list[float],
    source_label: str,
) -> xr.Dataset:
    time_index = pd.to_datetime(local_times)
    qpf_proxy = _pop_to_qpf_proxy(np.array(pop_pct, dtype=float))
    return xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), np.array(temp_f, dtype=np.float32)[:, None, None]),
            "dewpoint_f": (("time", "lat", "lon"), np.array(dewpoint_f, dtype=np.float32)[:, None, None]),
            "wind_mph": (("time", "lat", "lon"), np.array(wind_mph, dtype=np.float32)[:, None, None]),
            "gust_mph": (("time", "lat", "lon"), np.array(gust_mph, dtype=np.float32)[:, None, None]),
            "cloud_pct": (("time", "lat", "lon"), np.array(cloud_pct, dtype=np.float32)[:, None, None]),
            "pop_pct": (("time", "lat", "lon"), np.array(pop_pct, dtype=np.float32)[:, None, None]),
            "qpf_in": (("time", "lat", "lon"), qpf_proxy.astype(np.float32)[:, None, None]),
            "thunder": (("time", "lat", "lon"), np.zeros((len(local_times), 1, 1), dtype=bool)),
        },
        coords={"time": time_index, "lat": [requested_lat], "lon": [requested_lon]},
        attrs={"source": source_label},
    )


@dataclass
class NdfdForecastRegionalLoader:
    """Archived NDFD forecast loader for first-pass west-coast verification experiments."""

    region_name: str
    mesh_profile: str = "standard"
    forecast_run_hour_utc: int = OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC
    forecast_lead_days: int = 1

    def load_hourly_grid(self, valid_date: date) -> tuple[xr.Dataset, dict[str, object]]:
        if self.region_name not in NDFD_FORECAST_SUPPORTED_REGIONS:
            raise ValueError(
                f"NDFD forecast scaffold currently supports {', '.join(NDFD_FORECAST_SUPPORTED_REGIONS)} only."
            )
        if self.forecast_lead_days != 1:
            raise ValueError("NDFD forecast scaffold currently supports D+1 verification only.")

        region = get_region_definition(self.region_name)
        settings = get_openmeteo_mesh_settings(self.region_name, self.mesh_profile)
        lat_values = _regional_coordinate_values(region.expanded_bounds[2], region.expanded_bounds[3], float(settings["lat_step"]))
        lon_values = _regional_coordinate_values(region.expanded_bounds[0], region.expanded_bounds[1], float(settings["lon_step"]))
        coordinate_pairs = [(lat, lon) for lat in lat_values for lon in lon_values]
        timezone_name = str(settings.get("timezone", "GMT"))
        schedule = _utc_hour_schedule(valid_date, timezone_name)
        run_timestamp_utc = datetime.combine(valid_date, datetime.min.time(), tzinfo=timezone.utc).replace(
            hour=self.forecast_run_hour_utc
        ) - timedelta(days=self.forecast_lead_days)
        run_date = run_timestamp_utc.date()

        entries = _catalog_entries(run_date)
        field_datasets: dict[str, xr.Dataset] = {}
        field_lookups: dict[str, dict[tuple[float, float], object]] = {}
        selected_entries: dict[str, dict[str, object]] = {}
        for field_name, spec in _FIELD_SPECS.items():
            entry = _select_catalog_entry(
                entries,
                label=spec["label"],
                target_run_timestamp_utc=run_timestamp_utc,
                region_name=self.region_name,
            )
            selected_entries[field_name] = entry
            dataset = _open_dataset(_download_cached_file(dataset_id=str(entry["dataset_id"])))
            field_datasets[field_name] = dataset
            field_lookups[field_name] = _dataset_lookup(dataset, coordinate_pairs, region.expanded_bounds)

        point_datasets: dict[tuple[float, float], xr.Dataset] = {}
        source_label = "ndfd_archive_core_kwbn"
        for requested_lat, requested_lon in coordinate_pairs:
            point_series: dict[str, list[float]] = {
                "temp_f": [],
                "dewpoint_f": [],
                "wind_mph": [],
                "gust_mph": [],
                "cloud_pct": [],
                "pop_pct": [],
            }
            local_times: list[datetime] = []
            for local_dt, utc_dt in schedule:
                local_times.append(local_dt)
                for field_name, spec in _FIELD_SPECS.items():
                    point_index = field_lookups[field_name][(requested_lat, requested_lon)]
                    point_series[field_name].append(
                        _dataset_value(
                            field_datasets[field_name],
                            point_index=point_index,
                            utc_dt=utc_dt,
                            converter=spec["converter"],
                        )
                    )
            point_datasets[(requested_lat, requested_lon)] = _point_dataset(
                requested_lat=requested_lat,
                requested_lon=requested_lon,
                local_times=local_times,
                temp_f=point_series["temp_f"],
                dewpoint_f=point_series["dewpoint_f"],
                wind_mph=point_series["wind_mph"],
                gust_mph=point_series["gust_mph"],
                cloud_pct=point_series["cloud_pct"],
                pop_pct=np.clip(np.array(point_series["pop_pct"], dtype=float), 0.0, 100.0).tolist(),
                source_label=source_label,
            )

        grid = assemble_point_datasets_to_grid(
            point_datasets=point_datasets,
            lat_values=lat_values,
            lon_values=lon_values,
            region_name=self.region_name,
        )
        metadata = {
            "forecast_model": OPENMETEO_VERIFICATION_FORECAST_MODEL_NWS_NDFD_HOURLY,
            "forecast_model_requested": OPENMETEO_VERIFICATION_FORECAST_MODEL_NWS_NDFD_HOURLY,
            "forecast_model_mode": "exact",
            "forecast_run_timestamp_utc": min(
                entry["run_timestamp_utc"].strftime("%Y-%m-%dT%H:%MZ")
                for entry in selected_entries.values()
            ),
            "forecast_source_label": "NWS NDFD Archive (core KWBN scaffold)",
            "forecast_grid_source": source_label,
            "forecast_selected_entries": {field_name: entry["dataset_id"] for field_name, entry in selected_entries.items()},
        }
        grid.attrs.update(
            {
                "source": source_label,
                "verification_region": self.region_name,
                "verification_run_timestamp_utc": metadata["forecast_run_timestamp_utc"],
                "mesh_profile": self.mesh_profile,
            }
        )
        return grid, metadata
