"""Loader interfaces for hourly forecast grids."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import xarray as xr

from nicewx.data.mock_data import generate_mock_hourly_grid
from nicewx.data.openmeteo import OpenMeteoPointLoader, OpenMeteoRegionalMeshLoader


class ForecastLoader(ABC):
    """Abstract loader for model-agnostic hourly forecast grids."""

    @abstractmethod
    def load_hourly_grid(self, valid_date: date) -> xr.Dataset:
        """Return a local-day hourly gridded forecast dataset."""


class MockForecastLoader(ForecastLoader):
    """Deterministic synthetic loader for end-to-end development."""

    def __init__(self, lat_points: int = 65, lon_points: int = 115) -> None:
        self.lat_points = lat_points
        self.lon_points = lon_points

    def load_hourly_grid(self, valid_date: date) -> xr.Dataset:
        return generate_mock_hourly_grid(
            valid_date=valid_date,
            lat_points=self.lat_points,
            lon_points=self.lon_points,
        )


def get_loader(
    loader_name: str = "mock",
    lat_points: int = 65,
    lon_points: int = 115,
    lat: float | None = None,
    lon: float | None = None,
    region_name: str | None = None,
    mesh_profile: str = "standard",
) -> ForecastLoader:
    """Return the configured loader implementation."""

    if loader_name == "mock":
        return MockForecastLoader(lat_points=lat_points, lon_points=lon_points)
    if loader_name == "openmeteo":
        if region_name is not None:
            return OpenMeteoRegionalMeshLoader(region_name=region_name, mesh_profile=mesh_profile)
        if lat is None or lon is None:
            raise ValueError("Open-Meteo point loading requires both lat and lon.")
        return OpenMeteoPointLoader(lat=lat, lon=lon)
    raise ValueError(
        f"Unsupported loader '{loader_name}'. Supported loaders are 'mock' and 'openmeteo'."
    )
