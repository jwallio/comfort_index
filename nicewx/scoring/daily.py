"""Daily aggregation logic for nice-weather scores."""

from __future__ import annotations

import numpy as np
import xarray as xr

from nicewx.config import (
    DAILY_AGGREGATION_DEFAULT_MODE,
    DAILY_DISRUPTION_WEIGHTS,
    DAILY_SCORE_WEIGHTS,
    DAYTIME_HOUR_WEIGHTS,
    get_daily_aggregation_mode_config,
    LOCAL_DAY_HOURS,
    MAX_DAILY_DISRUPTION_PENALTY,
    PRIME_INTERRUPTION_THRESHOLDS,
    PRIME_DAY_HOURS,
    PRIME_RECOVERY_SETTINGS,
    PRISTINE_GATE_THRESHOLDS,
    RELIABILITY_COMPONENT_WEIGHTS,
    RELIABILITY_HIGH_THRESHOLD,
    RELIABILITY_THRESHOLD,
)
from nicewx.scoring.categories import categorize_scores


def _select_hours(dataset: xr.Dataset, hour_bounds: tuple[int, int]) -> xr.Dataset:
    start_hour, end_hour = hour_bounds
    hour_values = dataset["time"].dt.hour
    mask = (hour_values >= start_hour) & (hour_values <= end_hour)
    return dataset.where(mask, drop=True)


def _weights_for_subset(dataset: xr.Dataset) -> xr.DataArray:
    hours = dataset["time"].dt.hour.values
    weights = np.array([DAYTIME_HOUR_WEIGHTS[int(hour)] for hour in hours], dtype=float)
    return xr.DataArray(weights, dims=("time",), coords={"time": dataset["time"]})


def _best_rolling_mean(scores: xr.DataArray, window: int) -> xr.DataArray:
    rolling = scores.rolling(time=window, min_periods=window).mean()
    return rolling.max("time", skipna=True).fillna(0.0)


def _weighted_fraction(indicator: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    return indicator.astype(float).weighted(weights).mean("time")


def _weighted_mean(values: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    return values.weighted(weights).mean("time")


def _linear_ramp(values: xr.DataArray, start: float, end: float) -> xr.DataArray:
    """Return a 0..1 ramp between two thresholds."""

    if end <= start:
        return xr.where(values >= end, 1.0, 0.0)
    return ((values - start) / (end - start)).clip(0.0, 1.0)


def _inverse_linear_ramp(values: xr.DataArray, low: float, high: float) -> xr.DataArray:
    """Return a 1..0 ramp as values rise from low to high."""

    if high <= low:
        return xr.where(values <= low, 1.0, 0.0)
    return ((high - values) / (high - low)).clip(0.0, 1.0)


def _score_drop_fraction(scores: xr.DataArray, weights: xr.DataArray, drop_threshold: float) -> xr.DataArray:
    """Return weighted fraction of prime-hour score drops larger than a threshold."""

    score_delta = scores.diff("time")
    drop_indicator = score_delta <= -drop_threshold
    drop_weights = weights.isel(time=slice(1, None))
    return _weighted_fraction(drop_indicator, drop_weights)


def _score_drop_signal(scores: xr.DataArray, weights: xr.DataArray, min_drop: float, full_drop: float) -> xr.DataArray:
    """Return a graded weighted score-drop signal."""

    drop_magnitude = (-scores.diff("time")).clip(min=0.0)
    drop_signal = _linear_ramp(drop_magnitude, min_drop, full_drop)
    drop_weights = weights.isel(time=slice(1, None))
    return _weighted_mean(drop_signal, drop_weights)


def _tail_clean_fraction(
    prime: xr.Dataset,
    tail_hours: int,
    precip_threshold: float,
    pop_threshold: float,
) -> xr.DataArray:
    """Return the fraction of the last prime hours that are clean and dry."""

    tail = prime.isel(time=slice(-tail_hours, None))
    tail_weights = _weights_for_subset(tail)
    clean_indicator = (
        (tail["qpf_in"] < precip_threshold)
        & (tail["pop_pct"] < pop_threshold)
        & (~tail["thunder"].astype(bool))
    )
    return _weighted_fraction(clean_indicator, tail_weights)


def _pristine_gate(
    daily_score: xr.DataArray,
    best_6hr: xr.DataArray,
    daytime_weighted_mean: xr.DataArray,
    reliability_score: xr.DataArray,
    prime_thunder_fraction: xr.DataArray,
    prime_measurable_precip_fraction: xr.DataArray,
    daytime_mean_dewpoint: xr.DataArray,
    daytime_mean_gust: xr.DataArray,
    prime_score_drop_fraction: xr.DataArray,
) -> xr.DataArray:
    """Return whether a grid cell clears the stricter pristine criteria."""

    return (
        (daily_score >= PRISTINE_GATE_THRESHOLDS["raw_score_min"])
        & (best_6hr >= PRISTINE_GATE_THRESHOLDS["best_6hr_min"])
        & (daytime_weighted_mean >= PRISTINE_GATE_THRESHOLDS["daytime_weighted_mean_min"])
        & (reliability_score >= PRISTINE_GATE_THRESHOLDS["reliability_score_min"])
        & (prime_thunder_fraction <= 0.0)
        & (prime_measurable_precip_fraction <= 0.0)
        & (daytime_mean_dewpoint <= PRISTINE_GATE_THRESHOLDS["daytime_mean_dewpoint_max"])
        & (daytime_mean_gust <= PRISTINE_GATE_THRESHOLDS["daytime_mean_gust_max"])
        & ((best_6hr - daytime_weighted_mean) <= PRISTINE_GATE_THRESHOLDS["best_6hr_daytime_gap_max"])
        & (prime_score_drop_fraction <= PRISTINE_GATE_THRESHOLDS["prime_score_drop_fraction_max"])
    )


def aggregate_daily_scores(
    scored_hourly: xr.Dataset,
    aggregation_mode: str = DAILY_AGGREGATION_DEFAULT_MODE,
) -> xr.Dataset:
    """Aggregate hourly scores to a daily grid score and category."""

    if "hourly_score" not in scored_hourly:
        raise ValueError("Expected scored hourly dataset with 'hourly_score'.")
    mode_config = get_daily_aggregation_mode_config(aggregation_mode)

    daytime = _select_hours(scored_hourly, LOCAL_DAY_HOURS)
    daytime_weights = _weights_for_subset(daytime)
    prime = _select_hours(scored_hourly, PRIME_DAY_HOURS)
    prime_weights = _weights_for_subset(prime)

    best_3hr = _best_rolling_mean(daytime["hourly_score"], 3)
    best_6hr = _best_rolling_mean(daytime["hourly_score"], 6)
    daytime_weighted_mean = _weighted_mean(daytime["hourly_score"], daytime_weights)
    daytime_mean_dewpoint = _weighted_mean(daytime["dewpoint_f"], daytime_weights)
    daytime_mean_gust = _weighted_mean(daytime["gust_mph"], daytime_weights)

    if bool(mode_config["soft_rain_signal"]):
        qpf_signal = _linear_ramp(
            prime["qpf_in"],
            float(mode_config["measurable_rain_qpf_min"]),
            float(mode_config["measurable_rain_qpf_full"]),
        )
        pop_signal = _linear_ramp(
            prime["pop_pct"],
            float(mode_config["measurable_rain_pop_min"]),
            float(mode_config["measurable_rain_pop_full"]),
        )
        measurable_rain = _weighted_mean(xr.where(qpf_signal > pop_signal, qpf_signal, pop_signal), prime_weights)
    else:
        measurable_rain = _weighted_fraction(
            (prime["qpf_in"] >= PRIME_INTERRUPTION_THRESHOLDS["qpf_in"])
            | (prime["pop_pct"] >= PRIME_INTERRUPTION_THRESHOLDS["pop_pct"]),
            prime_weights,
        )
    heavy_precip = _weighted_fraction(
        prime["qpf_in"] >= PRIME_INTERRUPTION_THRESHOLDS["heavy_qpf_in"],
        prime_weights,
    )
    thunder = _weighted_fraction(prime["thunder"], prime_weights) if "thunder" in prime else 0.0
    if bool(mode_config["soft_gust_signal"]):
        strong_gusts = _weighted_mean(
            _linear_ramp(
                prime["gust_mph"],
                float(mode_config["gust_soft_min"]),
                float(mode_config["gust_soft_full"]),
            ),
            prime_weights,
        )
    else:
        strong_gusts = _weighted_fraction(
            prime["gust_mph"] >= PRIME_INTERRUPTION_THRESHOLDS["gust_mph"],
            prime_weights,
        )
    if bool(mode_config["soft_score_crash_signal"]):
        score_crash = _weighted_mean(
            _inverse_linear_ramp(
                prime["hourly_score"],
                float(mode_config["score_crash_floor"]),
                float(mode_config["score_crash_ceiling"]),
            ),
            prime_weights,
        )
    else:
        score_crash = _weighted_fraction(
            prime["hourly_score"] < PRIME_INTERRUPTION_THRESHOLDS["score_crash"],
            prime_weights,
        )
    if bool(mode_config["soft_score_drop_signal"]):
        score_drop = _score_drop_signal(
            prime["hourly_score"],
            prime_weights,
            float(mode_config["score_drop_min"]),
            float(mode_config["score_drop_full"]),
        )
    else:
        score_drop = _score_drop_fraction(
            prime["hourly_score"],
            prime_weights,
            PRIME_INTERRUPTION_THRESHOLDS["score_drop"],
        )
    tail_clean_fraction = _tail_clean_fraction(
        prime=prime,
        tail_hours=int(PRIME_RECOVERY_SETTINGS["tail_hours"]),
        precip_threshold=PRIME_INTERRUPTION_THRESHOLDS["qpf_in"],
        pop_threshold=PRIME_INTERRUPTION_THRESHOLDS["pop_pct"],
    )

    prime_penalty_weights = mode_config["prime_clean_penalty_weights"]
    assert isinstance(prime_penalty_weights, dict)
    prime_clean_fraction = 1.0 - (
        float(prime_penalty_weights["rain"]) * measurable_rain
        + float(prime_penalty_weights["thunder"]) * thunder
        + float(prime_penalty_weights["gust"]) * strong_gusts
    ).clip(0.0, 1.0)
    if bool(mode_config["graded_reliability"]):
        usable_hours = _weighted_mean(
            _linear_ramp(
                daytime["hourly_score"],
                float(mode_config["usable_score_min"]),
                float(mode_config["usable_score_full"]),
            ),
            daytime_weights,
        )
        strong_hours = _weighted_mean(
            _linear_ramp(
                daytime["hourly_score"],
                float(mode_config["strong_score_min"]),
                float(mode_config["strong_score_full"]),
            ),
            daytime_weights,
        )
    else:
        usable_hours = _weighted_fraction(daytime["hourly_score"] >= RELIABILITY_THRESHOLD, daytime_weights)
        strong_hours = _weighted_fraction(daytime["hourly_score"] >= RELIABILITY_HIGH_THRESHOLD, daytime_weights)
    reliability_score = (
        RELIABILITY_COMPONENT_WEIGHTS["usable_hours"]
        * usable_hours
        + RELIABILITY_COMPONENT_WEIGHTS["strong_hours"] * strong_hours
        + RELIABILITY_COMPONENT_WEIGHTS["prime_clean_hours"] * prime_clean_fraction
    ) * 100.0

    disruption_weights = mode_config["disruption_weights"]
    assert isinstance(disruption_weights, dict)
    disruption_penalty = (
        float(disruption_weights["measurable_rain"]) * measurable_rain
        + float(disruption_weights.get("heavy_precip", 0.0)) * heavy_precip
        + float(disruption_weights["thunder"]) * thunder
        + float(disruption_weights["strong_gusts"]) * strong_gusts
        + float(disruption_weights["score_crash"]) * score_crash
        + float(disruption_weights["score_drop"]) * score_drop
        - PRIME_RECOVERY_SETTINGS["tail_clean_credit"] * tail_clean_fraction * (1.0 - thunder)
    ).clip(min=0.0, max=MAX_DAILY_DISRUPTION_PENALTY)

    daily_score = (
        DAILY_SCORE_WEIGHTS["best_3hr"] * best_3hr
        + DAILY_SCORE_WEIGHTS["best_6hr"] * best_6hr
        + DAILY_SCORE_WEIGHTS["daytime_weighted_mean"] * daytime_weighted_mean
        + DAILY_SCORE_WEIGHTS["reliability_score"] * reliability_score
        - disruption_penalty
    ).clip(min=0.0, max=100.0)

    pristine_allowed = _pristine_gate(
        daily_score=daily_score,
        best_6hr=best_6hr,
        daytime_weighted_mean=daytime_weighted_mean,
        reliability_score=reliability_score,
        prime_thunder_fraction=thunder,
        prime_measurable_precip_fraction=measurable_rain,
        daytime_mean_dewpoint=daytime_mean_dewpoint,
        daytime_mean_gust=daytime_mean_gust,
        prime_score_drop_fraction=score_drop,
    )

    return xr.Dataset(
        data_vars={
            "daily_score": daily_score,
            "best_3hr": best_3hr,
            "best_6hr": best_6hr,
            "daytime_weighted_mean": daytime_weighted_mean,
            "daytime_mean_dewpoint": daytime_mean_dewpoint,
            "daytime_mean_gust": daytime_mean_gust,
            "reliability_score": reliability_score,
            "disruption_penalty": disruption_penalty,
            "prime_measurable_precip_fraction": measurable_rain,
            "prime_heavy_precip_fraction": heavy_precip,
            "prime_thunder_fraction": thunder,
            "prime_gusty_fraction": strong_gusts,
            "prime_score_crash_fraction": score_crash,
            "prime_score_drop_fraction": score_drop,
            "prime_tail_clean_fraction": tail_clean_fraction,
            "pristine_allowed": pristine_allowed,
            "category_index": categorize_scores(daily_score, pristine_allowed=pristine_allowed),
        },
        coords={"lat": scored_hourly["lat"], "lon": scored_hourly["lon"]},
        attrs={
            "description": "Daily nice weather aggregate score",
            "category_labels": "Poor,Fair,Pleasant,Ideal,Exceptional",
            "aggregation_mode": aggregation_mode,
        },
    )
