"""Mismatch-audit helpers for real-world validation cases."""

from __future__ import annotations

import xarray as xr


def _scalar(dataset: xr.Dataset, name: str) -> float:
    return float(dataset[name].values.squeeze())


def build_point_contribution_summary(scored_point: xr.Dataset, daily_point: xr.Dataset) -> dict[str, float]:
    """Return compact component and penalty summaries for a single point."""

    daytime = scored_point.sel(time=scored_point["time"].dt.hour.isin(range(8, 21)))
    mean_air_quality_effect = -(
        float(daytime["air_quality_penalty"].mean().values) + float(daytime["visibility_penalty"].mean().values)
    )
    total_interaction_adjustment = float(daytime["interaction_adjustment"].sum().values)
    total_hazard_disruption_effect = float(daytime["hazard_penalty"].sum().values) + _scalar(daily_point, "disruption_penalty")

    return {
        "mean_temp_score": round(float(daytime["temp_score"].mean().values), 2),
        "mean_dewpoint_score": round(float(daytime["dewpoint_score"].mean().values), 2),
        "mean_wind_score": round(float(daytime["wind_score"].mean().values), 2),
        "mean_cloud_score": round(float(daytime["cloud_score"].mean().values), 2),
        "mean_precip_score": round(float(daytime["precip_score"].mean().values), 2),
        "mean_air_quality_effect": round(mean_air_quality_effect, 2),
        "total_interaction_adjustment": round(total_interaction_adjustment, 2),
        "total_hazard_disruption_effect": round(total_hazard_disruption_effect, 2),
    }


def audit_point_mismatch(scored_point: xr.Dataset, daily_point: xr.Dataset) -> dict[str, object]:
    """Return dominant limiting factor and top reasons for a validation case."""

    daytime = scored_point.sel(time=scored_point["time"].dt.hour.isin(range(8, 21)))
    contributions = build_point_contribution_summary(scored_point, daily_point)

    temperature_loss = 35.0 - contributions["mean_temp_score"]
    humidity_loss = 20.0 - contributions["mean_dewpoint_score"]
    wind_loss = 10.0 - contributions["mean_wind_score"]
    cloud_loss = 10.0 - contributions["mean_cloud_score"]
    precip_loss = 15.0 - contributions["mean_precip_score"]
    air_quality_loss = max(0.0, -contributions["mean_air_quality_effect"])
    hazard_loss = float(daytime["hazard_penalty"].mean().values)
    aggregation_loss = _scalar(daily_point, "disruption_penalty") + max(0.0, 70.0 - _scalar(daily_point, "reliability_score")) * 0.15

    limiting_factors = {
        "temperature": temperature_loss,
        "humidity": humidity_loss,
        "wind": wind_loss,
        "clouds": cloud_loss,
        "precipitation": precip_loss,
        "air quality": air_quality_loss,
        "thunder/hazard": hazard_loss,
        "aggregation/reliability/disruption": aggregation_loss,
    }
    dominant_limiting_factor = max(limiting_factors, key=limiting_factors.get)

    candidates: list[tuple[float, str]] = []
    if temperature_loss >= 10.0:
        candidates.append((temperature_loss, "Temperature profile held the score down"))
    else:
        candidates.append((contributions["mean_temp_score"], "Temperature profile boosted the score"))

    if humidity_loss >= 8.0:
        candidates.append((humidity_loss, "Humidity held the score down"))
    elif contributions["mean_dewpoint_score"] >= 15.0:
        candidates.append((contributions["mean_dewpoint_score"], "Humidity stayed manageable"))

    if cloud_loss >= 5.0:
        candidates.append((cloud_loss, "Cloud cover suppressed the score"))
    elif contributions["mean_cloud_score"] >= 7.0:
        candidates.append((contributions["mean_cloud_score"], "Cloud cover helped the score"))

    if _scalar(daily_point, "prime_thunder_fraction") > 0.0:
        candidates.append((_scalar(daily_point, "prime_thunder_fraction") * 30.0, "Prime-hour thunderstorms cut deeply into reliability"))
    elif _scalar(daily_point, "prime_measurable_precip_fraction") > 0.0:
        candidates.append((_scalar(daily_point, "prime_measurable_precip_fraction") * 20.0, "Prime-hour precipitation reduced the usable window"))
    else:
        candidates.append((contributions["mean_precip_score"], "Dry prime hours supported the score"))

    if hazard_loss >= 3.0 or _scalar(daily_point, "prime_gusty_fraction") > 0.15:
        candidates.append((max(hazard_loss, _scalar(daily_point, "prime_gusty_fraction") * 20.0), "Wind and hazard effects reduced outdoor comfort"))

    if air_quality_loss > 0.0:
        candidates.append((air_quality_loss, "Air quality or visibility suppressed the score"))

    if aggregation_loss >= 6.0:
        candidates.append((aggregation_loss, "Aggregation penalized weak reliability or disruption"))
    else:
        candidates.append((max(_scalar(daily_point, "reliability_score") / 10.0, 0.0), "Reliable prime hours supported the final score"))

    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    top_reasons = [reason for _, reason in ranked[:3]]

    return {
        "dominant_limiting_factor": dominant_limiting_factor,
        "top_reason_1": top_reasons[0] if len(top_reasons) > 0 else "",
        "top_reason_2": top_reasons[1] if len(top_reasons) > 1 else "",
        "top_reason_3": top_reasons[2] if len(top_reasons) > 2 else "",
        "top_3_reasons": " | ".join(top_reasons),
        **contributions,
    }
