"""NOAA URMA/RTMA regional analysis loader for verification truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

from comfortwx.config import (
    NOAA_ANALYSIS_CACHE_DIR,
    NOAA_ANALYSIS_CACHE_VERSION,
    NOAA_ANALYSIS_LOCAL_HOURS,
    NOAA_ANALYSIS_TIMEOUT_SECONDS,
    NOAA_RTMA_ANALYSIS_BASE_URL,
    NOAA_URMA_ANALYSIS_BASE_URL,
    OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_ANALYSIS_POP_PROXY_QPF_FULL_IN,
    OPENMETEO_VERIFICATION_ANALYSIS_MODEL_NOAA_URMA_RTMA,
    OPENMETEO_VERIFICATION_ANALYSIS_MODEL_OPENMETEO_ARCHIVE,
    get_openmeteo_mesh_settings,
)
from comfortwx.data.openmeteo import _regional_coordinate_values, assemble_point_datasets_to_grid
from comfortwx.mapping.regions import get_region_definition

_MS_TO_MPH = 2.2369362920544
_M_TO_MI = 1.0 / 1609.344
_MM_TO_IN = 1.0 / 25.4


def _kelvin_to_fahrenheit(values: np.ndarray) -> np.ndarray:
    return values * 9.0 / 5.0 - 459.67


def _utc_hour_schedule(valid_date: date, timezone_name: str) -> list[tuple[datetime, datetime]]:
    local_zone = ZoneInfo(timezone_name)
    start_hour, end_hour = NOAA_ANALYSIS_LOCAL_HOURS
    schedule: list[tuple[datetime, datetime]] = []
    for hour in range(start_hour, end_hour + 1):
        local_dt = datetime.combine(valid_date, time(hour), tzinfo=local_zone)
        utc_dt = local_dt.astimezone(timezone.utc)
        schedule.append((local_dt.replace(tzinfo=None), utc_dt))
    return schedule


def _analysis_cache_path(*, source: str, utc_dt: datetime, suffix: str) -> Path:
    day_stem = utc_dt.strftime("%Y%m%d")
    hour_stem = utc_dt.strftime("%H")
    return NOAA_ANALYSIS_CACHE_DIR / NOAA_ANALYSIS_CACHE_VERSION / source / day_stem / f"{hour_stem}_{suffix}.grb2"


def _surface_analysis_url(*, source: str, utc_dt: datetime) -> str:
    day_stem = utc_dt.strftime("%Y%m%d")
    hour_stem = utc_dt.strftime("%H")
    if source == "urma":
        return (
            f"{NOAA_URMA_ANALYSIS_BASE_URL}/urma2p5.{day_stem}/"
            f"urma2p5.t{hour_stem}z.2dvaranl_ndfd.grb2_wexp"
        )
    if source == "rtma":
        return (
            f"{NOAA_RTMA_ANALYSIS_BASE_URL}/rtma2p5.{day_stem}/"
            f"rtma2p5.t{hour_stem}z.2dvaranl_ndfd.grb2_wexp"
        )
    raise ValueError(f"Unsupported NOAA analysis source '{source}'.")


def _precip_analysis_url(*, source: str, utc_dt: datetime) -> str:
    day_stem = utc_dt.strftime("%Y%m%d")
    hour_stamp = utc_dt.strftime("%Y%m%d%H")
    if source == "urma":
        return (
            f"{NOAA_URMA_ANALYSIS_BASE_URL}/urma2p5.{day_stem}/"
            f"urma2p5.{hour_stamp}.pcp_01h.wexp.grb2"
        )
    if source == "rtma":
        return (
            f"{NOAA_RTMA_ANALYSIS_BASE_URL}/rtma2p5.{day_stem}/"
            f"rtma2p5.{hour_stamp}.pcp.184.grb2"
        )
    raise ValueError(f"Unsupported NOAA precipitation source '{source}'.")


def _download_cached_file(url: str, cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=float(NOAA_ANALYSIS_TIMEOUT_SECONDS)) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _open_surface_dataset(path: Path) -> tuple[xr.Dataset, xr.Dataset]:
    common_kwargs = {"indexpath": ""}
    base = xr.open_dataset(path, engine="cfgrib", backend_kwargs={**common_kwargs, "errors": "ignore"})
    wind = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={**common_kwargs, "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10}},
    )
    return base, wind


def _open_precip_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def _bbox_subset_indices(latitude: np.ndarray, longitude: np.ndarray, bounds: tuple[float, float, float, float]) -> tuple[slice, slice]:
    lon_min, lon_max, lat_min, lat_max = bounds
    mask = (
        (latitude >= lat_min - 1.0)
        & (latitude <= lat_max + 1.0)
        & (longitude >= lon_min - 1.0)
        & (longitude <= lon_max + 1.0)
    )
    if not np.any(mask):
        return slice(0, latitude.shape[0]), slice(0, latitude.shape[1])
    y_indices, x_indices = np.where(mask)
    y0 = max(int(y_indices.min()) - 2, 0)
    y1 = min(int(y_indices.max()) + 3, latitude.shape[0])
    x0 = max(int(x_indices.min()) - 2, 0)
    x1 = min(int(x_indices.max()) + 3, latitude.shape[1])
    return slice(y0, y1), slice(x0, x1)


def _nearest_point_lookup(
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    coordinate_pairs: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> dict[tuple[float, float], tuple[int, int]]:
    y_slice, x_slice = _bbox_subset_indices(latitude, longitude, bounds)
    lat_subset = latitude[y_slice, x_slice]
    lon_subset = longitude[y_slice, x_slice]
    lookup: dict[tuple[float, float], tuple[int, int]] = {}
    for requested_lat, requested_lon in coordinate_pairs:
        distance = (lat_subset - requested_lat) ** 2 + (lon_subset - requested_lon) ** 2
        y_local, x_local = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
        lookup[(requested_lat, requested_lon)] = (y_local + y_slice.start, x_local + x_slice.start)
    return lookup


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
    qpf_in: list[float],
    visibility_mi: list[float],
    source_label: str,
) -> xr.Dataset:
    time_index = pd.to_datetime(local_times)
    qpf_array = np.array(qpf_in, dtype=float)
    pop_pct = np.clip((np.clip(qpf_array, 0.0, None) / OPENMETEO_VERIFICATION_ANALYSIS_POP_PROXY_QPF_FULL_IN) * 100.0, 0.0, 100.0)
    return xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), np.array(temp_f, dtype=np.float32)[:, None, None]),
            "dewpoint_f": (("time", "lat", "lon"), np.array(dewpoint_f, dtype=np.float32)[:, None, None]),
            "wind_mph": (("time", "lat", "lon"), np.array(wind_mph, dtype=np.float32)[:, None, None]),
            "gust_mph": (("time", "lat", "lon"), np.array(gust_mph, dtype=np.float32)[:, None, None]),
            "cloud_pct": (("time", "lat", "lon"), np.array(cloud_pct, dtype=np.float32)[:, None, None]),
            "pop_pct": (("time", "lat", "lon"), pop_pct.astype(np.float32)[:, None, None]),
            "qpf_in": (("time", "lat", "lon"), qpf_array.astype(np.float32)[:, None, None]),
            "thunder": (("time", "lat", "lon"), np.zeros((len(local_times), 1, 1), dtype=bool)),
            "visibility_mi": (("time", "lat", "lon"), np.array(visibility_mi, dtype=np.float32)[:, None, None]),
        },
        coords={"time": time_index, "lat": [requested_lat], "lon": [requested_lon]},
        attrs={"source": source_label},
    )


def _fetch_noaa_hourly_grids(utc_dt: datetime) -> tuple[str, xr.Dataset, xr.Dataset, xr.Dataset]:
    last_error: Exception | None = None
    for source in ("urma", "rtma"):
        try:
            surface_path = _download_cached_file(
                _surface_analysis_url(source=source, utc_dt=utc_dt),
                _analysis_cache_path(source=source, utc_dt=utc_dt, suffix="surface"),
            )
            precip_path = _download_cached_file(
                _precip_analysis_url(source=source, utc_dt=utc_dt),
                _analysis_cache_path(source=source, utc_dt=utc_dt, suffix="precip"),
            )
            base_ds, wind_ds = _open_surface_dataset(surface_path)
            precip_ds = _open_precip_dataset(precip_path)
            return source, base_ds, wind_ds, precip_ds
        except HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                continue
            raise
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No NOAA truth files available for {utc_dt:%Y-%m-%d %H}Z.")


@dataclass
class NoaaAnalysisRegionalLoader:
    """NOAA URMA/RTMA daytime analysis mesh loader for verification truth."""

    region_name: str
    mesh_profile: str = "standard"
    analysis_model: str = OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT

    def load_hourly_grid(self, valid_date: date) -> tuple[xr.Dataset, dict[str, object]]:
        if self.analysis_model not in {
            OPENMETEO_VERIFICATION_ANALYSIS_MODEL_NOAA_URMA_RTMA,
            OPENMETEO_VERIFICATION_ANALYSIS_MODEL_OPENMETEO_ARCHIVE,
        }:
            raise ValueError(f"Unsupported NOAA analysis model '{self.analysis_model}'.")

        region = get_region_definition(self.region_name)
        settings = get_openmeteo_mesh_settings(self.region_name, self.mesh_profile)
        lat_values = _regional_coordinate_values(region.expanded_bounds[2], region.expanded_bounds[3], float(settings["lat_step"]))
        lon_values = _regional_coordinate_values(region.expanded_bounds[0], region.expanded_bounds[1], float(settings["lon_step"]))
        coordinate_pairs = [(lat, lon) for lat in lat_values for lon in lon_values]
        timezone_name = str(settings.get("timezone", "GMT"))
        schedule = _utc_hour_schedule(valid_date, timezone_name)

        point_records: dict[tuple[float, float], dict[str, list[float] | list[datetime]]] = {
            pair: {
                "time": [],
                "temp_f": [],
                "dewpoint_f": [],
                "wind_mph": [],
                "gust_mph": [],
                "cloud_pct": [],
                "qpf_in": [],
                "visibility_mi": [],
            }
            for pair in coordinate_pairs
        }
        lookup_cache: dict[str, dict[tuple[float, float], tuple[int, int]]] = {}
        source_sequence: list[str] = []

        for local_dt, utc_dt in schedule:
            source_name, base_ds, wind_ds, precip_ds = _fetch_noaa_hourly_grids(utc_dt)
            source_sequence.append(source_name)
            if source_name not in lookup_cache:
                lookup_cache[source_name] = _nearest_point_lookup(
                    latitude=np.asarray(base_ds["latitude"].values),
                    longitude=np.asarray(base_ds["longitude"].values),
                    coordinate_pairs=coordinate_pairs,
                    bounds=region.expanded_bounds,
                )

            lookup = lookup_cache[source_name]
            base_temp = np.asarray(base_ds["t2m"].values)
            base_dew = np.asarray(base_ds["d2m"].values)
            base_vis = np.asarray(base_ds["vis"].values)
            base_cloud = np.asarray(base_ds["tcc"].values)
            wind_speed = np.asarray(wind_ds["si10"].values)
            wind_gust = np.asarray(wind_ds["i10fg"].values if "i10fg" in wind_ds else wind_ds["si10"].values)
            precip = np.asarray(precip_ds["tp"].values)

            for pair in coordinate_pairs:
                y_index, x_index = lookup[pair]
                point_records[pair]["time"].append(local_dt)
                point_records[pair]["temp_f"].append(float(_kelvin_to_fahrenheit(base_temp[y_index, x_index])))
                point_records[pair]["dewpoint_f"].append(float(_kelvin_to_fahrenheit(base_dew[y_index, x_index])))
                point_records[pair]["wind_mph"].append(float(wind_speed[y_index, x_index] * _MS_TO_MPH))
                point_records[pair]["gust_mph"].append(float(max(wind_speed[y_index, x_index], wind_gust[y_index, x_index]) * _MS_TO_MPH))
                point_records[pair]["cloud_pct"].append(float(np.clip(base_cloud[y_index, x_index], 0.0, 100.0)))
                point_records[pair]["qpf_in"].append(float(max(0.0, precip[y_index, x_index]) * _MM_TO_IN))
                point_records[pair]["visibility_mi"].append(float(max(0.0, base_vis[y_index, x_index]) * _M_TO_MI))

            base_ds.close()
            wind_ds.close()
            precip_ds.close()

        source_label = "noaa_urma" if source_sequence and all(source == "urma" for source in source_sequence) else "noaa_urma+rtma_fallback"
        point_datasets = {
            pair: _point_dataset(
                requested_lat=pair[0],
                requested_lon=pair[1],
                local_times=point_records[pair]["time"],
                temp_f=point_records[pair]["temp_f"],
                dewpoint_f=point_records[pair]["dewpoint_f"],
                wind_mph=point_records[pair]["wind_mph"],
                gust_mph=point_records[pair]["gust_mph"],
                cloud_pct=point_records[pair]["cloud_pct"],
                qpf_in=point_records[pair]["qpf_in"],
                visibility_mi=point_records[pair]["visibility_mi"],
                source_label=source_label,
            )
            for pair in coordinate_pairs
        }
        grid = assemble_point_datasets_to_grid(
            point_datasets=point_datasets,
            lat_values=lat_values,
            lon_values=lon_values,
            region_name=self.region_name,
        )
        grid.attrs.update(
            {
                "source": source_label,
                "verification_region": self.region_name,
                "verification_valid_date": valid_date.isoformat(),
                "mesh_profile": self.mesh_profile,
            }
        )
        metadata = {
            "analysis_model": source_label,
            "analysis_timezone": timezone_name,
            "analysis_hour_count": len(schedule),
            "mesh_point_count": len(coordinate_pairs),
        }
        return grid, metadata
