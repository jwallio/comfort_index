"""Composition of hourly comfort scores."""

from __future__ import annotations

import xarray as xr

from comfortwx.config import HOURLY_REQUIRED_FIELDS
from comfortwx.scoring.air_quality import score_optional_air_quality
from comfortwx.scoring.clouds import score_clouds
from comfortwx.scoring.hazards import hazard_cap, hazard_penalty
from comfortwx.scoring.humidity import score_dewpoint
from comfortwx.scoring.interactions import total_interaction_adjustment
from comfortwx.scoring.precip import score_precipitation
from comfortwx.scoring.temperature import score_temperature
from comfortwx.scoring.wind import score_wind


def _validate_hourly_dataset(dataset: xr.Dataset) -> None:
    missing = [field for field in HOURLY_REQUIRED_FIELDS if field not in dataset]
    if missing:
        raise ValueError(f"Hourly dataset missing required fields: {missing}")


def score_hourly_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Score each hour for each grid cell and return component fields."""

    _validate_hourly_dataset(dataset)
    thunder = dataset["thunder"] if "thunder" in dataset else None

    temp_component = score_temperature(dataset["temp_f"])
    dewpoint_component = score_dewpoint(dataset["dewpoint_f"])
    wind_component = score_wind(dataset["wind_mph"])
    cloud_component = score_clouds(dataset["temp_f"], dataset["cloud_pct"])
    precip_component = score_precipitation(
        pop_pct=dataset["pop_pct"],
        qpf_in=dataset["qpf_in"],
        thunder=thunder,
    )
    interaction_component = total_interaction_adjustment(
        temp_f=dataset["temp_f"],
        dewpoint_f=dataset["dewpoint_f"],
        wind_mph=dataset["wind_mph"],
        cloud_pct=dataset["cloud_pct"],
    )
    hazard_component = hazard_penalty(dataset["gust_mph"], thunder=thunder)
    cap_component = hazard_cap(dataset["temp_f"], thunder=thunder)
    optional_components, optional_cap = score_optional_air_quality(dataset)
    combined_cap = xr.where(optional_cap < cap_component, optional_cap, cap_component)

    pre_cap_score = (
        temp_component
        + dewpoint_component
        + wind_component
        + cloud_component
        + precip_component
        + interaction_component
        - optional_components["air_quality_penalty"]
        - optional_components["visibility_penalty"]
        - hazard_component
    )
    hourly_score = pre_cap_score.clip(min=0.0, max=100.0)
    hourly_score = xr.where(hourly_score > combined_cap, combined_cap, hourly_score)
    hourly_score = hourly_score.clip(min=0.0, max=100.0)

    scored = dataset.copy()
    scored["temp_score"] = temp_component
    scored["dewpoint_score"] = dewpoint_component
    scored["wind_score"] = wind_component
    scored["cloud_score"] = cloud_component
    scored["precip_score"] = precip_component
    scored["interaction_adjustment"] = interaction_component
    scored["air_quality_penalty"] = optional_components["air_quality_penalty"]
    scored["visibility_penalty"] = optional_components["visibility_penalty"]
    scored["hazard_penalty"] = hazard_component
    scored["hazard_cap"] = cap_component
    scored["air_quality_cap"] = optional_components["air_quality_cap"]
    scored["hourly_score_cap"] = combined_cap
    scored["hourly_score_pre_cap"] = pre_cap_score
    scored["hourly_score"] = hourly_score
    scored["hourly_score"].attrs["long_name"] = "Hourly comfort score"
    scored["hourly_score"].attrs["units"] = "0-100"
    return scored
