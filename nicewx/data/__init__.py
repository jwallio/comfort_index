"""Forecast data loading interfaces."""

from nicewx.data.loaders import ForecastLoader, get_loader
from nicewx.data.openmeteo import OpenMeteoPointLoader

__all__ = ["ForecastLoader", "OpenMeteoPointLoader", "get_loader"]
