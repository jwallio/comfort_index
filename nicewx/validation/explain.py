"""Rule-based explainability helpers for point diagnostics."""

from __future__ import annotations

import xarray as xr


def _scalar(dataset: xr.Dataset, name: str) -> float:
    """Return a scalar float from a daily point dataset."""

    return float(dataset[name].values.squeeze())


def explain_point_series(scored_point: xr.Dataset, daily_point: xr.Dataset) -> str:
    """Return a concise rule-based explanation for a scored point."""

    daytime = scored_point.sel(time=scored_point["time"].dt.hour.isin(range(8, 21)))
    mean_temp_score = float(daytime["temp_score"].mean().values)
    mean_dewpoint_score = float(daytime["dewpoint_score"].mean().values)
    mean_cloud_score = float(daytime["cloud_score"].mean().values)
    mean_cloud_pct = float(daytime["cloud_pct"].mean().values)
    mean_temp = float(daytime["temp_f"].mean().values)
    mean_hazard_penalty = float(daytime["hazard_penalty"].mean().values)
    mean_air_quality_penalty = float(daytime["air_quality_penalty"].mean().values) if "air_quality_penalty" in daytime else 0.0
    mean_visibility_penalty = float(daytime["visibility_penalty"].mean().values) if "visibility_penalty" in daytime else 0.0

    clauses: list[str] = []

    if mean_temp_score >= 28.0:
        clauses.append("Excellent temperature profile")
    elif mean_temp_score <= 16.0:
        clauses.append("Temperature profile limited outdoor comfort")

    if _scalar(daily_point, "daytime_mean_dewpoint") >= 64.0 or mean_dewpoint_score <= 8.0:
        clauses.append("Humidity prevented elite score")

    if _scalar(daily_point, "prime_thunder_fraction") > 0.05:
        clauses.append("Prime-hour thunderstorms heavily reduced reliability")
    elif _scalar(daily_point, "prime_measurable_precip_fraction") > 0.2:
        clauses.append("Repeated prime-hour precipitation reduced the usable window")
    elif (
        _scalar(daily_point, "prime_measurable_precip_fraction") > 0.0
        and _scalar(daily_point, "prime_tail_clean_fraction") >= 0.6
    ):
        clauses.append("Early precipitation faded and the day recovered later")

    if mean_temp >= 82.0 and 20.0 <= mean_cloud_pct <= 55.0 and mean_cloud_score >= 7.0:
        clauses.append("Clouds helped afternoon heat slightly")
    elif mean_cloud_pct >= 82.0 or mean_cloud_score <= 4.5:
        clauses.append("Persistent overcast suppressed elite potential")

    if _scalar(daily_point, "prime_gusty_fraction") > 0.15 or mean_hazard_penalty >= 3.0:
        clauses.append("Gusty conditions reduced outdoor comfort")

    if mean_air_quality_penalty >= 4.0 or mean_visibility_penalty >= 4.0:
        clauses.append("Smoke or low visibility suppressed elite potential")

    if (
        _scalar(daily_point, "reliability_score") >= 90.0
        and _scalar(daily_point, "prime_measurable_precip_fraction") <= 0.0
        and _scalar(daily_point, "prime_thunder_fraction") <= 0.0
    ):
        clauses.append("Clean prime hours preserved high reliability")

    if not clauses:
        clauses.append("Balanced thermodynamics and clean prime hours supported a solid outdoor day")

    return ". ".join(clauses[:3]) + "."
