"""Sanity-check demo cases for known weather regimes."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from nicewx.scoring.categories import category_name_from_index
from nicewx.scoring.daily import aggregate_daily_scores
from nicewx.scoring.hourly import score_hourly_dataset
from nicewx.validation.explain import explain_point_series
from nicewx.validation.inspection import point_hourly_breakdown_dataframe


def _build_case_dataset(
    valid_date: date,
    name: str,
    temp_f: np.ndarray,
    dewpoint_f: np.ndarray,
    wind_mph: np.ndarray,
    gust_mph: np.ndarray,
    cloud_pct: np.ndarray,
    pop_pct: np.ndarray,
    qpf_in: np.ndarray,
    thunder: np.ndarray,
) -> xr.Dataset:
    times = pd.date_range(pd.Timestamp(valid_date), periods=24, freq="1h")
    coords = {"time": times, "lat": [35.0], "lon": [-80.0]}
    return xr.Dataset(
        data_vars={
            "temp_f": (("time", "lat", "lon"), temp_f[:, None, None]),
            "dewpoint_f": (("time", "lat", "lon"), dewpoint_f[:, None, None]),
            "wind_mph": (("time", "lat", "lon"), wind_mph[:, None, None]),
            "gust_mph": (("time", "lat", "lon"), gust_mph[:, None, None]),
            "cloud_pct": (("time", "lat", "lon"), cloud_pct[:, None, None]),
            "pop_pct": (("time", "lat", "lon"), pop_pct[:, None, None]),
            "qpf_in": (("time", "lat", "lon"), qpf_in[:, None, None]),
            "thunder": (("time", "lat", "lon"), thunder[:, None, None]),
        },
        coords=coords,
        attrs={"case_name": name},
    )


def _series(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def build_demo_cases(valid_date: date) -> dict[str, xr.Dataset]:
    """Return four fixed point-case datasets for sanity checks."""

    return {
        "raleigh": _build_case_dataset(
            valid_date=valid_date,
            name="Raleigh-like spring day",
            temp_f=_series([47, 46, 45, 44, 44, 45, 48, 52, 57, 62, 66, 70, 73, 75, 76, 76, 74, 71, 67, 62, 58, 54, 51, 49]),
            dewpoint_f=_series([46, 45, 44, 44, 43, 43, 45, 47, 49, 50, 51, 52, 53, 54, 54, 55, 55, 54, 53, 52, 51, 49, 48, 47]),
            wind_mph=_series([3, 3, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 8, 8, 8, 7, 7, 6, 5, 4, 4, 3, 3, 3]),
            gust_mph=_series([6, 6, 5, 5, 5, 5, 6, 7, 9, 10, 11, 12, 12, 12, 11, 10, 10, 9, 8, 7, 7, 6, 6, 6]),
            cloud_pct=_series([20, 18, 16, 14, 14, 15, 18, 20, 22, 24, 25, 28, 30, 32, 34, 32, 28, 24, 22, 20, 18, 16, 16, 18]),
            pop_pct=_series([5, 5, 5, 5, 5, 5, 4, 4, 4, 5, 6, 6, 8, 8, 10, 10, 9, 8, 6, 5, 5, 5, 5, 5]),
            qpf_in=_series([0] * 24),
            thunder=np.zeros(24, dtype=bool),
        ),
        "miami": _build_case_dataset(
            valid_date=valid_date,
            name="Miami-like hot humid day",
            temp_f=_series([77, 76, 75, 75, 75, 76, 78, 80, 82, 84, 86, 87, 88, 89, 89, 88, 87, 85, 83, 81, 80, 79, 78, 77]),
            dewpoint_f=_series([74, 74, 73, 73, 73, 73, 74, 74, 75, 75, 75, 76, 76, 76, 76, 75, 75, 74, 74, 74, 74, 74, 74, 74]),
            wind_mph=_series([8, 8, 8, 7, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 13, 13, 12, 11, 10, 10, 9, 9, 8, 8]),
            gust_mph=_series([12, 12, 12, 11, 11, 12, 13, 15, 17, 18, 20, 22, 24, 25, 23, 22, 21, 19, 17, 16, 15, 14, 13, 12]),
            cloud_pct=_series([35, 35, 36, 36, 38, 40, 42, 45, 48, 52, 55, 58, 60, 62, 60, 58, 56, 54, 50, 46, 42, 40, 38, 36]),
            pop_pct=_series([20, 20, 20, 20, 20, 22, 25, 28, 30, 32, 35, 40, 50, 60, 65, 60, 55, 45, 35, 30, 25, 22, 20, 20]),
            qpf_in=_series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00, 0.01, 0.03, 0.05, 0.02, 0.01, 0.00, 0.00, 0, 0, 0, 0, 0]),
            thunder=np.array([False] * 13 + [True, True, False, False] + [False] * 7, dtype=bool),
        ),
        "denver": _build_case_dataset(
            valid_date=valid_date,
            name="Denver-like dry pleasant day",
            temp_f=_series([39, 37, 35, 34, 33, 34, 38, 45, 53, 61, 67, 72, 75, 77, 78, 77, 74, 69, 63, 56, 50, 45, 42, 40]),
            dewpoint_f=_series([20, 20, 19, 19, 18, 18, 20, 22, 24, 26, 28, 30, 31, 32, 33, 32, 31, 30, 28, 26, 24, 22, 21, 20]),
            wind_mph=_series([4, 4, 3, 3, 3, 3, 4, 5, 7, 8, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 5, 4, 4]),
            gust_mph=_series([7, 7, 6, 6, 6, 6, 7, 8, 11, 12, 14, 16, 17, 17, 16, 15, 13, 12, 11, 10, 9, 8, 7, 7]),
            cloud_pct=_series([5, 5, 5, 5, 5, 5, 7, 8, 10, 12, 14, 16, 18, 20, 22, 22, 20, 18, 14, 10, 8, 7, 6, 5]),
            pop_pct=_series([3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 7, 7, 6, 5, 5, 4, 4, 3, 3, 3, 3]),
            qpf_in=_series([0] * 24),
            thunder=np.zeros(24, dtype=bool),
        ),
        "seattle": _build_case_dataset(
            valid_date=valid_date,
            name="Seattle-like cloudy cool day",
            temp_f=_series([47, 46, 46, 45, 45, 45, 46, 47, 49, 51, 53, 55, 56, 57, 58, 57, 56, 54, 52, 50, 49, 48, 48, 47]),
            dewpoint_f=_series([44, 44, 44, 43, 43, 43, 44, 45, 45, 46, 46, 47, 48, 48, 48, 47, 47, 46, 46, 45, 45, 45, 44, 44]),
            wind_mph=_series([5, 5, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 9, 9, 8, 7, 6, 6, 5, 5, 5]),
            gust_mph=_series([8, 8, 8, 8, 8, 8, 9, 10, 11, 12, 12, 13, 13, 14, 14, 13, 13, 12, 11, 10, 9, 8, 8, 8]),
            cloud_pct=_series([88, 90, 92, 92, 93, 94, 94, 93, 92, 92, 90, 88, 86, 85, 84, 85, 86, 88, 90, 92, 93, 94, 92, 90]),
            pop_pct=_series([55, 55, 58, 60, 60, 58, 55, 52, 50, 48, 45, 40, 38, 38, 40, 42, 45, 48, 52, 55, 58, 60, 58, 56]),
            qpf_in=_series([0.01, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01]),
            thunder=np.zeros(24, dtype=bool),
        ),
    }


def build_demo_case_hourly_breakdown(valid_date: date) -> pd.DataFrame:
    """Return concatenated hourly component diagnostics for all demo cases."""

    frames: list[pd.DataFrame] = []
    for key, dataset in build_demo_cases(valid_date).items():
        scored = score_hourly_dataset(dataset)
        frame = point_hourly_breakdown_dataframe(scored)
        frame.insert(0, "case_key", key)
        frame.insert(1, "case_name", str(dataset.attrs["case_name"]))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_demo_case_validation(valid_date: date) -> tuple[pd.DataFrame, str]:
    """Score demo cases and return a summary table plus a printable report."""

    records: list[dict[str, object]] = []
    report_lines = ["Demo case sanity checks:"]

    for key, dataset in build_demo_cases(valid_date).items():
        scored = score_hourly_dataset(dataset)
        daily = aggregate_daily_scores(scored)

        daily_score = float(daily["daily_score"].values.squeeze())
        case_name = str(dataset.attrs["case_name"])
        category = category_name_from_index(int(daily["category_index"].values.squeeze()))
        explanation = explain_point_series(scored_point=scored, daily_point=daily)
        daytime_scores = scored["hourly_score"].sel(time=scored["time"].dt.hour.isin(range(8, 21)))
        hourly_pairs = [
            f"{pd.Timestamp(timestamp).hour:02d}:{int(round(value))}"
            for timestamp, value in zip(daytime_scores["time"].values, daytime_scores.values.squeeze(), strict=False)
        ]

        records.append(
            {
                "case_key": key,
                "case_name": case_name,
                "best_3hr": round(float(daily["best_3hr"].values.squeeze()), 1),
                "best_6hr": round(float(daily["best_6hr"].values.squeeze()), 1),
                "daytime_weighted_mean": round(float(daily["daytime_weighted_mean"].values.squeeze()), 1),
                "reliability_score": round(float(daily["reliability_score"].values.squeeze()), 1),
                "disruption_penalty": round(float(daily["disruption_penalty"].values.squeeze()), 1),
                "daily_score": round(daily_score, 1),
                "category": category,
                "daytime_mean_dewpoint": round(float(daily["daytime_mean_dewpoint"].values.squeeze()), 1),
                "daytime_mean_gust": round(float(daily["daytime_mean_gust"].values.squeeze()), 1),
                "prime_measurable_precip_fraction": round(float(daily["prime_measurable_precip_fraction"].values.squeeze()), 3),
                "prime_thunder_fraction": round(float(daily["prime_thunder_fraction"].values.squeeze()), 3),
                "prime_gusty_fraction": round(float(daily["prime_gusty_fraction"].values.squeeze()), 3),
                "prime_score_crash_fraction": round(float(daily["prime_score_crash_fraction"].values.squeeze()), 3),
                "prime_score_drop_fraction": round(float(daily["prime_score_drop_fraction"].values.squeeze()), 3),
                "prime_tail_clean_fraction": round(float(daily["prime_tail_clean_fraction"].values.squeeze()), 3),
                "pristine_allowed": bool(daily["pristine_allowed"].values.squeeze()),
                "explanation": explanation,
            }
        )
        report_lines.append(f"- {case_name}: {daily_score:.1f} ({category})")
        report_lines.append(
            "  Metrics: "
            f"best_3hr={float(daily['best_3hr'].values.squeeze()):.1f}, "
            f"best_6hr={float(daily['best_6hr'].values.squeeze()):.1f}, "
            f"day_mean={float(daily['daytime_weighted_mean'].values.squeeze()):.1f}, "
            f"reliability={float(daily['reliability_score'].values.squeeze()):.1f}, "
            f"disruption={float(daily['disruption_penalty'].values.squeeze()):.1f}"
        )
        report_lines.append(f"  Why: {explanation}")
        report_lines.append(f"  Daytime hourly scores: {', '.join(hourly_pairs)}")

    summary = pd.DataFrame.from_records(records).sort_values("daily_score", ascending=False)
    report_lines.append("")
    report_lines.append("Calibration table:")
    report_lines.append(
        summary.loc[
            :,
            [
                "case_name",
                "best_3hr",
                "best_6hr",
                "daytime_weighted_mean",
                "reliability_score",
                "disruption_penalty",
                "daily_score",
                "category",
            ],
        ].to_string(index=False)
    )
    return summary, "\n".join(report_lines)
