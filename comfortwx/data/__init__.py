"""Forecast data loading interfaces."""

from comfortwx.data.loaders import ForecastLoader, get_loader
from comfortwx.data.openmeteo import OpenMeteoPointLoader

__all__ = ["ForecastLoader", "OpenMeteoPointLoader", "get_loader"]
