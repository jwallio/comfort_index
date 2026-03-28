"""Benchmark harness for proxy forecast-vs-analysis verification."""

from __future__ import annotations

import argparse
import html
import shutil
import time
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from comfortwx.config import (
    OPENMETEO_VERIFICATION_BENCHMARK_LEAD_DAYS,
    OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS,
    OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC,
    OUTPUT_DIR,
    VERIFICATION_CALIBRATION_INTERCEPT_RANGE,
    VERIFICATION_CALIBRATION_BIAS_SHRINKAGE_OFFSET,
    VERIFICATION_CALIBRATION_LINEAR_MIN_CASES,
    VERIFICATION_CALIBRATION_LINEAR_MIN_POINTS,
    VERIFICATION_CALIBRATION_MIN_CASES,
    VERIFICATION_CALIBRATION_MIN_POINTS,
    VERIFICATION_CALIBRATION_SLOPE_RANGE,
    VERIFICATION_BENCHMARK_LEAD_THRESHOLDS,
    VERIFICATION_BENCHMARK_THRESHOLDS,
    VERIFICATION_INCREMENTAL_CASE_COOLDOWN_SECONDS,
    VERIFICATION_INCREMENTAL_MAX_FRESH_CASES_BY_TIER,
    list_verification_aggregation_policies,
)
from comfortwx.data.openmeteo_verification import resolve_openmeteo_verification_forecast_model
from comfortwx.scoring.categories import categorize_scores
from comfortwx.validation.verify_benchmark_cases import (
    DEFAULT_VERIFICATION_BENCHMARK_CASES,
    VerificationBenchmarkCase,
    VERIFICATION_BENCHMARK_TIER_DEFAULT,
    VERIFICATION_BENCHMARK_TIERS,
    get_benchmark_case_set,
)
from comfortwx.validation.verify_model import build_verification_file_prefix, run_verification


def _resolved_cases(
    date_override: date | None,
    benchmark_tier: str = VERIFICATION_BENCHMARK_TIER_DEFAULT,
    lead_days_override: tuple[int, ...] | None = None,
) -> list[VerificationBenchmarkCase]:
    base_cases = get_benchmark_case_set(benchmark_tier)
    lead_days = lead_days_override or tuple(
        dict.fromkeys(case.forecast_lead_days for case in base_cases)
    ) or OPENMETEO_VERIFICATION_BENCHMARK_LEAD_DAYS
    if date_override is None:
        if lead_days_override is None:
            return list(base_cases)
        unique_region_dates: list[tuple[str, date]] = []
        for case in base_cases:
            key = (case.region_name, case.valid_date)
            if key not in unique_region_dates:
                unique_region_dates.append(key)
        return [
            VerificationBenchmarkCase(
                region_name=region_name,
                valid_date=valid_date,
                forecast_lead_days=lead_day,
            )
            for region_name, valid_date in unique_region_dates
            for lead_day in lead_days
        ]
    regions_in_order: list[str] = []
    for case in base_cases:
        if case.region_name not in regions_in_order:
            regions_in_order.append(case.region_name)
    return [
        VerificationBenchmarkCase(
            region_name=region_name,
            valid_date=date_override,
            forecast_lead_days=lead_day,
        )
        for region_name in regions_in_order
        for lead_day in lead_days
    ]


def _parse_aggregation_policies(value: str) -> tuple[str, ...]:
    policies = tuple(part.strip() for part in value.split(",") if part.strip())
    if not policies:
        raise ValueError("At least one aggregation policy is required.")
    available = set(list_verification_aggregation_policies())
    invalid = [policy for policy in policies if policy not in available]
    if invalid:
        raise ValueError(
            f"Unknown aggregation policies: {', '.join(invalid)}. Available policies: {', '.join(sorted(available))}."
        )
    return tuple(dict.fromkeys(policies))


def _max_fresh_cases_for_tier(benchmark_tier: str) -> int:
    return int(VERIFICATION_INCREMENTAL_MAX_FRESH_CASES_BY_TIER.get(benchmark_tier, 0))


def _filter_cases(
    cases: list[VerificationBenchmarkCase],
    *,
    region_filter: tuple[str, ...] | None,
    date_filter: tuple[date, ...] | None,
) -> list[VerificationBenchmarkCase]:
    filtered = list(cases)
    if region_filter:
        allowed_regions = {value.strip().lower() for value in region_filter if value.strip()}
        filtered = [case for case in filtered if case.region_name.lower() in allowed_regions]
    if date_filter:
        allowed_dates = set(date_filter)
        filtered = [case for case in filtered if case.valid_date in allowed_dates]
    return filtered


def _lead_label(lead_day: int) -> str:
    return f"D+{lead_day}"


def _thresholds_for_lead(lead_day: int) -> dict[str, float]:
    return VERIFICATION_BENCHMARK_LEAD_THRESHOLDS.get(lead_day, VERIFICATION_BENCHMARK_THRESHOLDS)


def _ok_cases(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "status" not in summary.columns:
        return summary.iloc[0:0].copy()
    normalized = summary.copy()
    required_columns = {"improvement_priority_score", "forecast_lead_days", "benchmark_threshold_status"}
    if not required_columns.issubset(set(normalized.columns)):
        normalized = _apply_threshold_flags(normalized)
    ok = normalized.loc[normalized["status"] == "ok"].copy()
    if ok.empty:
        return ok
    ok["date"] = pd.to_datetime(ok["date"])
    ok["forecast_lead_days"] = ok["forecast_lead_days"].astype(int)
    ok["lead_label"] = ok["forecast_lead_days"].map(_lead_label)
    ok["case_label"] = ok.apply(
        lambda row: (
            f"{row['region']} {row['date']:%Y-%m-%d} {row['lead_label']}"
            + (
                f" [{row['verification_aggregation_policy']}]"
                if "verification_aggregation_policy" in row and str(row["verification_aggregation_policy"]).strip()
                else ""
            )
        ),
        axis=1,
    )
    ok["score_bias_mean"] = ok["score_bias_mean"].astype(float)
    ok["score_mae"] = ok["score_mae"].astype(float)
    ok["score_rmse"] = ok["score_rmse"].astype(float)
    ok["exact_category_agreement_fraction"] = ok["exact_category_agreement_fraction"].astype(float)
    ok["near_category_agreement_fraction"] = ok["near_category_agreement_fraction"].astype(float)
    if "high_comfort_precision" in ok.columns:
        ok["high_comfort_precision"] = pd.to_numeric(ok["high_comfort_precision"], errors="coerce")
    if "high_comfort_recall" in ok.columns:
        ok["high_comfort_recall"] = pd.to_numeric(ok["high_comfort_recall"], errors="coerce")
    ok["abs_score_bias_mean"] = ok["score_bias_mean"].abs()
    ok["verification_rank_score"] = (
        ok["score_mae"]
        + ok["score_rmse"] * 0.35
        + ok["abs_score_bias_mean"] * 0.75
        + (1.0 - ok["near_category_agreement_fraction"]) * 20.0
    )
    return ok


def _safe_path(value: object) -> Path | None:
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.exists() and path.is_file() else None


def _existing_verification_case_outputs(
    *,
    valid_date: date,
    region_name: str,
    forecast_model: str,
    forecast_model_mode: str = "auto",
    analysis_model: str = OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    forecast_lead_days: int,
    aggregation_policy: str,
    output_dir: Path,
) -> dict[str, object] | None:
    resolved_forecast_model = resolve_openmeteo_verification_forecast_model(
        requested_model=forecast_model,
        forecast_lead_days=forecast_lead_days,
        forecast_model_mode=forecast_model_mode,
        region_name=region_name,
    )
    file_prefix = build_verification_file_prefix(
        region_name=region_name,
        resolved_forecast_model=resolved_forecast_model,
        analysis_model=analysis_model,
        forecast_lead_days=forecast_lead_days,
        aggregation_policy=aggregation_policy,
    )
    date_stem = f"{valid_date:%Y%m%d}"
    output_paths = {
        "forecast_daily_fields": output_dir / f"{file_prefix}_forecast_daily_fields_{date_stem}.nc",
        "analysis_daily_fields": output_dir / f"{file_prefix}_analysis_daily_fields_{date_stem}.nc",
        "forecast_score_map": output_dir / f"{file_prefix}_forecast_score_{date_stem}.png",
        "analysis_score_map": output_dir / f"{file_prefix}_analysis_score_{date_stem}.png",
        "score_difference_map": output_dir / f"{file_prefix}_score_diff_{date_stem}.png",
        "absolute_error_map": output_dir / f"{file_prefix}_absolute_error_{date_stem}.png",
        "category_disagreement_map": output_dir / f"{file_prefix}_category_disagreement_{date_stem}.png",
        "missed_high_comfort_map": output_dir / f"{file_prefix}_missed_high_comfort_{date_stem}.png",
        "false_high_comfort_map": output_dir / f"{file_prefix}_false_high_comfort_{date_stem}.png",
        "summary_csv": output_dir / f"{file_prefix}_summary_{date_stem}.csv",
        "point_metrics_csv": output_dir / f"{file_prefix}_points_{date_stem}.csv",
        "component_metrics_csv": output_dir / f"{file_prefix}_components_{date_stem}.csv",
        "request_summary_csv": output_dir / f"{file_prefix}_{date_stem}_openmeteo_request_summary.csv",
        "request_detail_csv": output_dir / f"{file_prefix}_{date_stem}_openmeteo_request_detail.csv",
    }
    if not output_paths["summary_csv"].exists():
        return None
    if not output_paths["forecast_daily_fields"].exists() or not output_paths["analysis_daily_fields"].exists():
        return None

    summary_row = pd.read_csv(output_paths["summary_csv"]).iloc[0].to_dict()
    return {
        "summary_record": summary_row,
        **output_paths,
    }


def _load_daily_pair(row: pd.Series, cache: dict[tuple[str, str], tuple[xr.Dataset, xr.Dataset]]) -> tuple[xr.Dataset, xr.Dataset] | None:
    forecast_path = _safe_path(row.get("forecast_daily_fields_path", ""))
    analysis_path = _safe_path(row.get("analysis_daily_fields_path", ""))
    if forecast_path is None or analysis_path is None:
        return None
    cache_key = (str(forecast_path), str(analysis_path))
    if cache_key not in cache:
        cache[cache_key] = (xr.load_dataset(forecast_path), xr.load_dataset(analysis_path))
    return cache[cache_key]


def _daily_score_metrics(forecast_daily: xr.Dataset, analysis_daily: xr.Dataset) -> dict[str, float]:
    diff = forecast_daily["daily_score"] - analysis_daily["daily_score"]
    abs_diff = np.abs(diff.values)
    category_diff = np.abs(forecast_daily["category_index"].values - analysis_daily["category_index"].values)
    valid_mask = np.isfinite(diff.values)
    return {
        "score_bias_mean": round(float(diff.mean().values), 2),
        "score_mae": round(float(np.nanmean(abs_diff)), 2),
        "score_rmse": round(float(np.sqrt(np.nanmean(diff.values**2))), 2),
        "exact_category_agreement_fraction": round(float(np.mean((category_diff == 0)[valid_mask])), 4),
        "near_category_agreement_fraction": round(float(np.mean((category_diff <= 1)[valid_mask])), 4),
    }


def _fit_case_calibration(
    target_row: pd.Series,
    training_rows: pd.DataFrame,
    daily_pair_cache: dict[tuple[str, str], tuple[xr.Dataset, xr.Dataset]],
) -> dict[str, object] | None:
    if training_rows.empty:
        return None

    target_region = str(target_row["region"])
    target_lead = int(target_row["forecast_lead_days"])
    selected_scope = "region+lead"
    selected_rows = training_rows.loc[
        (training_rows["region"] == target_region) & (training_rows["forecast_lead_days"] == target_lead)
    ].copy()
    if len(selected_rows) < VERIFICATION_CALIBRATION_MIN_CASES:
        return None

    forecast_values: list[np.ndarray] = []
    analysis_values: list[np.ndarray] = []
    for _, training_row in selected_rows.iterrows():
        pair = _load_daily_pair(training_row, daily_pair_cache)
        if pair is None:
            continue
        forecast_daily, analysis_daily = pair
        forecast_score = forecast_daily["daily_score"].values.ravel()
        analysis_score = analysis_daily["daily_score"].values.ravel()
        valid_mask = np.isfinite(forecast_score) & np.isfinite(analysis_score)
        if np.any(valid_mask):
            forecast_values.append(forecast_score[valid_mask])
            analysis_values.append(analysis_score[valid_mask])

    if not forecast_values:
        return None

    x = np.concatenate(forecast_values)
    y = np.concatenate(analysis_values)
    if x.size < VERIFICATION_CALIBRATION_MIN_POINTS:
        return None

    if len(selected_rows) >= VERIFICATION_CALIBRATION_LINEAR_MIN_CASES and x.size >= VERIFICATION_CALIBRATION_LINEAR_MIN_POINTS:
        slope, intercept = np.polyfit(x, y, 1)
        method = "linear"
    else:
        slope = 1.0
        shrinkage = len(selected_rows) / (len(selected_rows) + VERIFICATION_CALIBRATION_BIAS_SHRINKAGE_OFFSET)
        intercept = float(np.mean(y - x) * shrinkage)
        method = "bias_only"

    slope = float(np.clip(slope, VERIFICATION_CALIBRATION_SLOPE_RANGE[0], VERIFICATION_CALIBRATION_SLOPE_RANGE[1]))
    intercept = float(
        np.clip(intercept, VERIFICATION_CALIBRATION_INTERCEPT_RANGE[0], VERIFICATION_CALIBRATION_INTERCEPT_RANGE[1])
    )
    return {
        "calibration_scope": selected_scope,
        "calibration_method": method,
        "calibration_training_case_count": int(len(selected_rows)),
        "calibration_training_point_count": int(x.size),
        "calibration_slope": round(slope, 4),
        "calibration_intercept": round(intercept, 4),
    }


def _apply_score_calibration(forecast_daily: xr.Dataset, *, slope: float, intercept: float) -> xr.Dataset:
    calibrated = forecast_daily.copy(deep=True)
    calibrated_score = (forecast_daily["daily_score"] * slope + intercept).clip(min=0.0, max=100.0)
    calibrated["daily_score"] = calibrated_score
    pristine_allowed = forecast_daily["pristine_allowed"] if "pristine_allowed" in forecast_daily else None
    calibrated["category_index"] = categorize_scores(calibrated_score, pristine_allowed=pristine_allowed)
    return calibrated


def _build_calibration_summary(
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    stem: str,
) -> tuple[pd.DataFrame, Path | None]:
    ok_cases = _ok_cases(summary)
    if ok_cases.empty:
        return pd.DataFrame(), None

    daily_pair_cache: dict[tuple[str, str], tuple[xr.Dataset, xr.Dataset]] = {}
    calibrated_records: list[dict[str, object]] = []

    for index, row in ok_cases.iterrows():
        training_rows = ok_cases.drop(index=index)
        calibration = _fit_case_calibration(row, training_rows, daily_pair_cache)
        if calibration is None:
            continue
        pair = _load_daily_pair(row, daily_pair_cache)
        if pair is None:
            continue
        forecast_daily, analysis_daily = pair
        calibrated_daily = _apply_score_calibration(
            forecast_daily,
            slope=float(calibration["calibration_slope"]),
            intercept=float(calibration["calibration_intercept"]),
        )
        calibrated_metrics = _daily_score_metrics(calibrated_daily, analysis_daily)
        calibrated_records.append(
            {
                "case_label": row["case_label"],
                "region": row["region"],
                "date": row["date"].date().isoformat() if hasattr(row["date"], "date") else str(row["date"]),
                "forecast_lead_days": int(row["forecast_lead_days"]),
                "lead_label": row["lead_label"],
                **calibration,
                "raw_score_bias_mean": float(row["score_bias_mean"]),
                "raw_score_mae": float(row["score_mae"]),
                "raw_score_rmse": float(row["score_rmse"]),
                "raw_exact_category_agreement_fraction": float(row["exact_category_agreement_fraction"]),
                "raw_near_category_agreement_fraction": float(row["near_category_agreement_fraction"]),
                "calibrated_score_bias_mean": calibrated_metrics["score_bias_mean"],
                "calibrated_score_mae": calibrated_metrics["score_mae"],
                "calibrated_score_rmse": calibrated_metrics["score_rmse"],
                "calibrated_exact_category_agreement_fraction": calibrated_metrics["exact_category_agreement_fraction"],
                "calibrated_near_category_agreement_fraction": calibrated_metrics["near_category_agreement_fraction"],
                "score_mae_improvement": round(float(row["score_mae"]) - calibrated_metrics["score_mae"], 2),
                "score_rmse_improvement": round(float(row["score_rmse"]) - calibrated_metrics["score_rmse"], 2),
                "near_category_agreement_improvement": round(
                    calibrated_metrics["near_category_agreement_fraction"] - float(row["near_category_agreement_fraction"]),
                    4,
                ),
            }
        )

    if not calibrated_records:
        return pd.DataFrame(), None

    calibrated_summary = pd.DataFrame.from_records(calibrated_records).sort_values(
        ["score_mae_improvement", "forecast_lead_days", "region", "date"],
        ascending=[False, True, True, True],
    )
    csv_path = output_dir / f"comfortwx_verify_benchmark_{stem}_calibration_summary.csv"
    calibrated_summary.to_csv(csv_path, index=False)
    return calibrated_summary, csv_path


def _apply_threshold_flags(summary: pd.DataFrame) -> pd.DataFrame:
    flagged = summary.copy()
    if "forecast_lead_days" not in flagged.columns:
        flagged["forecast_lead_days"] = OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS
    flagged["passes_mae_threshold"] = False
    flagged["passes_near_category_threshold"] = False
    flagged["passes_bias_threshold"] = False
    flagged["passes_benchmark_thresholds"] = False
    flagged["benchmark_threshold_status"] = "error"
    flagged["threshold_score_mae_max"] = np.nan
    flagged["threshold_near_category_agreement_min"] = np.nan
    flagged["threshold_abs_score_bias_mean_max"] = np.nan
    flagged["score_mae_excess"] = np.nan
    flagged["near_category_agreement_shortfall"] = np.nan
    flagged["abs_score_bias_mean_excess"] = np.nan
    flagged["improvement_priority_score"] = np.nan

    ok_mask = flagged["status"] == "ok"
    if not ok_mask.any():
        return flagged

    ok_indices = list(flagged.index[ok_mask])
    for index in ok_indices:
        lead_day = int(flagged.at[index, "forecast_lead_days"])
        thresholds = _thresholds_for_lead(lead_day)
        score_mae = float(flagged.at[index, "score_mae"])
        near_agreement = float(flagged.at[index, "near_category_agreement_fraction"])
        abs_bias = abs(float(flagged.at[index, "score_bias_mean"]))

        flagged.at[index, "threshold_score_mae_max"] = thresholds["score_mae_max"]
        flagged.at[index, "threshold_near_category_agreement_min"] = thresholds["near_category_agreement_min"]
        flagged.at[index, "threshold_abs_score_bias_mean_max"] = thresholds["abs_score_bias_mean_max"]
        flagged.at[index, "score_mae_excess"] = max(0.0, score_mae - thresholds["score_mae_max"])
        flagged.at[index, "near_category_agreement_shortfall"] = max(
            0.0,
            thresholds["near_category_agreement_min"] - near_agreement,
        )
        flagged.at[index, "abs_score_bias_mean_excess"] = max(
            0.0,
            abs_bias - thresholds["abs_score_bias_mean_max"],
        )
        flagged.at[index, "passes_mae_threshold"] = score_mae <= thresholds["score_mae_max"]
        flagged.at[index, "passes_near_category_threshold"] = near_agreement >= thresholds["near_category_agreement_min"]
        flagged.at[index, "passes_bias_threshold"] = abs_bias <= thresholds["abs_score_bias_mean_max"]

    flagged.loc[ok_mask, "passes_benchmark_thresholds"] = (
        flagged.loc[ok_mask, "passes_mae_threshold"]
        & flagged.loc[ok_mask, "passes_near_category_threshold"]
        & flagged.loc[ok_mask, "passes_bias_threshold"]
    )
    flagged.loc[ok_mask, "benchmark_threshold_status"] = np.where(
        flagged.loc[ok_mask, "passes_benchmark_thresholds"],
        "pass",
        "warn",
    )
    flagged.loc[ok_mask, "improvement_priority_score"] = (
        flagged.loc[ok_mask, "score_mae_excess"].astype(float) * 3.0
        + flagged.loc[ok_mask, "near_category_agreement_shortfall"].astype(float) * 30.0
        + flagged.loc[ok_mask, "abs_score_bias_mean_excess"].astype(float) * 1.5
    )
    return flagged


def format_benchmark_table(summary: pd.DataFrame) -> str:
    columns = [
        "case_label",
        "verification_aggregation_policy",
        "region",
        "date",
        "forecast_lead_days",
        "status",
        "benchmark_threshold_status",
        "score_bias_mean",
        "score_mae",
        "score_rmse",
        "exact_category_agreement_fraction",
        "near_category_agreement_fraction",
    ]
    present_columns = [column for column in columns if column in summary.columns]
    ranking_columns = [
        column
        for column in ["forecast_lead_days", "score_mae", "score_rmse", "region", "date"]
        if column in summary.columns
    ]
    return summary.loc[:, present_columns].sort_values(ranking_columns).to_string(index=False)


def _case_colors(frame: pd.DataFrame) -> list[str]:
    return ["#2e8b57" if bool(value) else "#d95f02" for value in frame["passes_benchmark_thresholds"]]


def _save_chart(fig, path: Path) -> Path:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_mae_bar_chart(ok_cases: pd.DataFrame, output_dir: Path, stem: str) -> Path | None:
    if ok_cases.empty:
        return None
    chart_data = ok_cases.sort_values(["score_mae", "score_rmse", "forecast_lead_days", "region", "date"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(chart_data["case_label"], chart_data["score_mae"], color=_case_colors(chart_data))
    ax.axhline(VERIFICATION_BENCHMARK_THRESHOLDS["score_mae_max"], color="#6b7280", linestyle="--", linewidth=1.1)
    ax.set_title("Benchmark MAE by Region, Date, and Lead")
    ax.set_ylabel("Score MAE")
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_mae_bar.png")


def _write_agreement_bar_chart(ok_cases: pd.DataFrame, output_dir: Path, stem: str) -> Path | None:
    if ok_cases.empty:
        return None
    chart_data = ok_cases.sort_values(
        ["near_category_agreement_fraction", "exact_category_agreement_fraction", "forecast_lead_days"],
        ascending=[False, False, True],
    )
    positions = np.arange(len(chart_data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(
        positions - width / 2,
        chart_data["exact_category_agreement_fraction"],
        width=width,
        color="#5b8ff9",
        label="Exact agreement",
    )
    ax.bar(
        positions + width / 2,
        chart_data["near_category_agreement_fraction"],
        width=width,
        color="#61d9a2",
        label="Near agreement",
    )
    ax.axhline(
        VERIFICATION_BENCHMARK_THRESHOLDS["near_category_agreement_min"],
        color="#6b7280",
        linestyle="--",
        linewidth=1.1,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(chart_data["case_label"], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Category Agreement by Region, Date, and Lead")
    ax.set_ylabel("Agreement fraction")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_agreement_bar.png")


def _write_bias_rmse_scatter(ok_cases: pd.DataFrame, output_dir: Path, stem: str) -> Path | None:
    if ok_cases.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    regions = list(dict.fromkeys(ok_cases["region"].tolist()))
    cmap = plt.get_cmap("tab10")
    for index, region in enumerate(regions):
        subset = ok_cases.loc[ok_cases["region"] == region]
        ax.scatter(
            subset["score_bias_mean"],
            subset["score_rmse"],
            s=70,
            alpha=0.9,
            color=cmap(index % 10),
            label=region,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['date']:%m/%d} {row['lead_label']}",
                (row["score_bias_mean"], row["score_rmse"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7.5,
            )
    ax.axvline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7)
    ax.axvline(VERIFICATION_BENCHMARK_THRESHOLDS["abs_score_bias_mean_max"], color="#b0b4b9", linestyle="--", linewidth=1.0)
    ax.axvline(-VERIFICATION_BENCHMARK_THRESHOLDS["abs_score_bias_mean_max"], color="#b0b4b9", linestyle="--", linewidth=1.0)
    ax.set_title("Bias vs RMSE")
    ax.set_xlabel("Mean score bias")
    ax.set_ylabel("Score RMSE")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_bias_rmse_scatter.png")


def _write_ranked_case_chart(ok_cases: pd.DataFrame, output_dir: Path, stem: str) -> Path | None:
    if ok_cases.empty:
        return None
    chart_data = ok_cases.sort_values(["verification_rank_score", "score_mae", "forecast_lead_days", "region", "date"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.barh(chart_data["case_label"], chart_data["verification_rank_score"], color=_case_colors(chart_data))
    ax.set_title("Ranked Verification Cases (Best to Worst)")
    ax.set_xlabel("Composite verification score (lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_ranked_cases.png")


def _aggregate_timeseries_by_lead(ok_cases: pd.DataFrame, y_column: str) -> pd.DataFrame:
    return (
        ok_cases.groupby(["date", "forecast_lead_days"], as_index=False)[y_column]
        .mean()
        .sort_values(["forecast_lead_days", "date"])
    )


def _write_timeseries_chart(
    ok_cases: pd.DataFrame,
    *,
    y_column: str,
    y_label: str,
    title: str,
    filename_suffix: str,
    output_dir: Path,
    stem: str,
    threshold: float | None = None,
) -> Path | None:
    if ok_cases.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    series = _aggregate_timeseries_by_lead(ok_cases, y_column)
    cmap = plt.get_cmap("viridis")
    lead_days = sorted(series["forecast_lead_days"].unique())
    for index, lead_day in enumerate(lead_days):
        subset = series.loc[series["forecast_lead_days"] == lead_day]
        ax.plot(
            subset["date"],
            subset[y_column],
            marker="o",
            linewidth=1.8,
            color=cmap(index / max(1, len(lead_days) - 1)),
            label=_lead_label(int(lead_day)),
        )
    if threshold is not None:
        ax.axhline(threshold, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Valid date")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2, title="Forecast lead")
    fig.autofmt_xdate(rotation=25)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_{filename_suffix}.png")


def _write_lead_summary(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> tuple[pd.DataFrame, Path | None]:
    ok_cases = _ok_cases(summary)
    if ok_cases.empty:
        return pd.DataFrame(), None

    lead_summary = ok_cases.groupby("forecast_lead_days", as_index=False).agg(
        case_count=("forecast_lead_days", "size"),
        mean_score_bias=("score_bias_mean", "mean"),
        mean_score_mae=("score_mae", "mean"),
        mean_score_rmse=("score_rmse", "mean"),
        mean_exact_category_agreement=("exact_category_agreement_fraction", "mean"),
        mean_near_category_agreement=("near_category_agreement_fraction", "mean"),
        pass_rate=("passes_benchmark_thresholds", "mean"),
    ).sort_values("forecast_lead_days")
    lead_summary["lead_label"] = lead_summary["forecast_lead_days"].map(lambda value: _lead_label(int(value)))
    csv_path = output_dir / f"comfortwx_verify_benchmark_{stem}_lead_summary.csv"
    lead_summary.to_csv(csv_path, index=False)
    return lead_summary, csv_path


def _write_lead_summary_chart(lead_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if lead_summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    positions = np.arange(len(lead_summary))
    width = 0.38
    ax.bar(
        positions - width / 2,
        lead_summary["mean_score_mae"],
        width=width,
        color="#4f8ad9",
        label="Mean MAE",
    )
    ax.bar(
        positions + width / 2,
        lead_summary["mean_score_rmse"],
        width=width,
        color="#c96b50",
        label="Mean RMSE",
    )
    ax.axhline(VERIFICATION_BENCHMARK_THRESHOLDS["score_mae_max"], color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(lead_summary["lead_label"])
    ax.set_ylabel("Score error")
    ax.set_title("Verification Summary by Forecast Lead")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_lead_summary.png")


def _write_calibration_mae_chart(calibration_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if calibration_summary.empty:
        return None
    chart_data = calibration_summary.sort_values(
        ["score_mae_improvement", "forecast_lead_days", "region", "date"],
        ascending=[False, True, True, True],
    )
    positions = np.arange(len(chart_data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(
        positions - width / 2,
        chart_data["raw_score_mae"],
        width=width,
        color="#9aa4b2",
        label="Raw MAE",
    )
    ax.bar(
        positions + width / 2,
        chart_data["calibrated_score_mae"],
        width=width,
        color="#2f855a",
        label="Calibrated MAE",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(chart_data["case_label"], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Score MAE")
    ax.set_title("Held-Out Calibration Impact by Case")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_calibration_mae.png")


def _write_calibration_lead_chart(calibration_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if calibration_summary.empty:
        return None
    lead_summary = calibration_summary.groupby("forecast_lead_days", as_index=False).agg(
        raw_score_mae=("raw_score_mae", "mean"),
        calibrated_score_mae=("calibrated_score_mae", "mean"),
        raw_score_rmse=("raw_score_rmse", "mean"),
        calibrated_score_rmse=("calibrated_score_rmse", "mean"),
    ).sort_values("forecast_lead_days")
    positions = np.arange(len(lead_summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(positions - width / 2, lead_summary["raw_score_mae"], width=width, color="#6b7280", label="Raw MAE")
    ax.bar(
        positions + width / 2,
        lead_summary["calibrated_score_mae"],
        width=width,
        color="#2b6cb0",
        label="Calibrated MAE",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([_lead_label(int(value)) for value in lead_summary["forecast_lead_days"]])
    ax.set_ylabel("Mean score MAE")
    ax.set_title("Held-Out Calibration Impact by Forecast Lead")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_calibration_lead.png")


def _primary_component_name(row: pd.Series) -> str:
    component_columns = {
        "temp_mae": "temperature",
        "dewpoint_mae": "dew point",
        "cloud_mae": "cloud",
        "precip_mae": "precipitation",
        "reliability_score_mae": "reliability",
        "disruption_penalty_mae": "disruption",
    }
    present = {
        label: float(row[column])
        for column, label in component_columns.items()
        if column in row and pd.notna(row[column])
    }
    if not present:
        return "unknown"
    return max(present.items(), key=lambda item: item[1])[0]


def _problem_tags(row: pd.Series) -> str:
    tags: list[str] = []
    if float(row.get("score_bias_mean", 0.0)) >= 2.5:
        tags.append("positive bias")
    elif float(row.get("score_bias_mean", 0.0)) <= -2.5:
        tags.append("negative bias")

    missed = float(row.get("missed_high_comfort_cell_count", 0.0))
    false = float(row.get("false_high_comfort_cell_count", 0.0))
    if missed >= false + 3:
        tags.append("underforecasted high comfort")
    elif false >= missed + 3:
        tags.append("overforecasted high comfort")

    primary_component = _primary_component_name(row)
    if primary_component != "unknown":
        tags.append(f"{primary_component} driven")
    return ", ".join(tags)


def _write_priority_case_summary(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> tuple[pd.DataFrame, Path | None]:
    ok_cases = _ok_cases(summary)
    if ok_cases.empty:
        return pd.DataFrame(), None

    priority_cases = ok_cases.copy()
    priority_cases["problem_tags"] = priority_cases.apply(_problem_tags, axis=1)
    priority_cases["primary_component"] = priority_cases.apply(_primary_component_name, axis=1)
    priority_cases = priority_cases.sort_values(
        ["improvement_priority_score", "score_mae", "forecast_lead_days", "region", "date"],
        ascending=[False, False, True, True, True],
    )
    columns = [
        "case_label",
        "region",
        "date",
        "forecast_lead_days",
        "lead_label",
        "benchmark_threshold_status",
        "improvement_priority_score",
        "score_mae",
        "score_rmse",
        "score_bias_mean",
        "near_category_agreement_fraction",
        "score_mae_excess",
        "near_category_agreement_shortfall",
        "abs_score_bias_mean_excess",
        "problem_tags",
        "primary_component",
        "forecast_score_map_path",
        "analysis_score_map_path",
        "score_difference_map_path",
        "absolute_error_map_path",
        "category_disagreement_map_path",
        "missed_high_comfort_map_path",
        "false_high_comfort_map_path",
        "component_metrics_csv_path",
    ]
    priority_cases = priority_cases.loc[:, [column for column in columns if column in priority_cases.columns]]
    csv_path = output_dir / f"comfortwx_verify_benchmark_{stem}_priority_cases.csv"
    priority_cases.to_csv(csv_path, index=False)
    return priority_cases, csv_path


def _write_priority_case_chart(priority_cases: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if priority_cases.empty:
        return None
    chart_data = priority_cases.head(10).sort_values(
        ["improvement_priority_score", "score_mae"],
        ascending=[True, True],
    )
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.45 * len(chart_data) + 1.8)))
    ax.barh(chart_data["case_label"], chart_data["improvement_priority_score"], color="#c96b50")
    ax.set_title("Top Verification Cases To Improve First")
    ax.set_xlabel("Improvement priority score")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_priority_cases.png")


def _write_region_lead_summary(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> tuple[pd.DataFrame, Path | None]:
    ok_cases = _ok_cases(summary)
    if ok_cases.empty:
        return pd.DataFrame(), None

    region_lead_summary = ok_cases.groupby(["region", "forecast_lead_days"], as_index=False).agg(
        case_count=("region", "size"),
        mean_score_mae=("score_mae", "mean"),
        mean_score_rmse=("score_rmse", "mean"),
        mean_near_category_agreement=("near_category_agreement_fraction", "mean"),
        pass_rate=("passes_benchmark_thresholds", "mean"),
        mean_priority_score=("improvement_priority_score", "mean"),
    ).sort_values(["mean_priority_score", "mean_score_mae", "forecast_lead_days"], ascending=[False, False, True])
    region_lead_summary["lead_label"] = region_lead_summary["forecast_lead_days"].map(lambda value: _lead_label(int(value)))
    csv_path = output_dir / f"comfortwx_verify_benchmark_{stem}_region_lead_summary.csv"
    region_lead_summary.to_csv(csv_path, index=False)
    return region_lead_summary, csv_path


def _write_region_lead_heatmap(region_lead_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if region_lead_summary.empty:
        return None
    pivot = region_lead_summary.pivot(index="region", columns="lead_label", values="mean_score_mae").sort_index()
    fig, ax = plt.subplots(figsize=(7.8, max(3.8, 0.7 * len(pivot.index) + 1.8)))
    image = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean MAE by Region and Forecast Lead")
    cbar = fig.colorbar(image, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Mean MAE")
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_region_lead_heatmap.png")


def _write_component_priority_summary(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> tuple[pd.DataFrame, Path | None]:
    ok_cases = _ok_cases(summary)
    if ok_cases.empty:
        return pd.DataFrame(), None

    component_columns = {
        "temp_mae": "temperature",
        "dewpoint_mae": "dew point",
        "cloud_mae": "cloud",
        "precip_mae": "precipitation",
        "reliability_score_mae": "reliability",
        "disruption_penalty_mae": "disruption",
    }
    rows: list[dict[str, object]] = []
    for column, label in component_columns.items():
        if column not in ok_cases.columns:
            continue
        values = pd.to_numeric(ok_cases[column], errors="coerce")
        weighted = values * ok_cases["improvement_priority_score"].fillna(0.0)
        rows.append(
            {
                "component_name": label,
                "mean_component_mae": round(float(values.mean()), 4),
                "warn_case_mean_component_mae": round(float(values.loc[ok_cases["benchmark_threshold_status"] == "warn"].mean()), 4)
                if (ok_cases["benchmark_threshold_status"] == "warn").any()
                else np.nan,
                "priority_weighted_component_mae": round(float(weighted.sum()), 4),
                "top_driver_case_count": int((ok_cases.apply(_primary_component_name, axis=1) == label).sum()),
            }
        )
    component_summary = pd.DataFrame.from_records(rows).sort_values(
        ["priority_weighted_component_mae", "mean_component_mae"],
        ascending=[False, False],
    )
    csv_path = output_dir / f"comfortwx_verify_benchmark_{stem}_component_priority.csv"
    component_summary.to_csv(csv_path, index=False)
    return component_summary, csv_path


def _write_component_priority_chart(component_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if component_summary.empty:
        return None
    chart_data = component_summary.sort_values(
        ["priority_weighted_component_mae", "mean_component_mae"],
        ascending=[True, True],
    )
    fig, ax = plt.subplots(figsize=(8.6, max(4.0, 0.6 * len(chart_data) + 1.6)))
    ax.barh(chart_data["component_name"], chart_data["priority_weighted_component_mae"], color="#5b8ff9")
    ax.set_title("Components Most Worth Improving First")
    ax.set_xlabel("Priority-weighted component error")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_component_priority.png")


def _write_benchmark_charts(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> dict[str, Path]:
    ok_cases = _ok_cases(summary)
    charts: dict[str, Path] = {}
    candidates = {
        "mae_bar_chart": _write_mae_bar_chart(ok_cases, output_dir, stem),
        "agreement_bar_chart": _write_agreement_bar_chart(ok_cases, output_dir, stem),
        "bias_rmse_scatter": _write_bias_rmse_scatter(ok_cases, output_dir, stem),
        "ranked_case_chart": _write_ranked_case_chart(ok_cases, output_dir, stem),
        "mae_timeseries_chart": _write_timeseries_chart(
            ok_cases,
            y_column="score_mae",
            y_label="Score MAE",
            title="Mean MAE Over Benchmark Dates by Forecast Lead",
            filename_suffix="mae_timeseries",
            output_dir=output_dir,
            stem=stem,
            threshold=VERIFICATION_BENCHMARK_THRESHOLDS["score_mae_max"],
        ),
        "agreement_timeseries_chart": _write_timeseries_chart(
            ok_cases,
            y_column="near_category_agreement_fraction",
            y_label="Near category agreement",
            title="Mean Near Category Agreement Over Benchmark Dates by Forecast Lead",
            filename_suffix="agreement_timeseries",
            output_dir=output_dir,
            stem=stem,
            threshold=VERIFICATION_BENCHMARK_THRESHOLDS["near_category_agreement_min"],
        ),
        "bias_timeseries_chart": _write_timeseries_chart(
            ok_cases,
            y_column="score_bias_mean",
            y_label="Mean score bias",
            title="Mean Bias Over Benchmark Dates by Forecast Lead",
            filename_suffix="bias_timeseries",
            output_dir=output_dir,
            stem=stem,
            threshold=0.0,
        ),
    }
    for key, value in candidates.items():
        if value is not None:
            charts[key] = value
    return charts


def _write_component_heatmap(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    ok_cases = _ok_cases(summary)
    component_columns = [
        ("temp_mae", "Temp"),
        ("dewpoint_mae", "Dew point"),
        ("cloud_mae", "Cloud"),
        ("precip_mae", "Precip"),
        ("reliability_score_mae", "Reliability"),
        ("disruption_penalty_mae", "Disruption"),
    ]
    present = [(column, label) for column, label in component_columns if column in ok_cases.columns]
    if ok_cases.empty or not present:
        return None

    chart_data = ok_cases.sort_values(["verification_rank_score", "forecast_lead_days", "region", "date"])
    matrix = np.array([[float(row[column]) for column, _ in present] for _, row in chart_data.iterrows()], dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, max(3.8, 0.5 * len(chart_data) + 1.8)))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([label for _, label in present], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(chart_data)))
    ax.set_yticklabels(chart_data["case_label"])
    ax.set_title("Component MAE Heatmap")
    cbar = fig.colorbar(image, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Mean absolute component error")
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_component_heatmap.png")


def _write_region_summary(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> tuple[pd.DataFrame, Path | None]:
    ok_cases = _ok_cases(summary)
    if ok_cases.empty:
        return pd.DataFrame(), None

    aggregations: dict[str, tuple[str, str]] = {
        "case_count": ("region", "size"),
        "mean_score_bias": ("score_bias_mean", "mean"),
        "mean_score_mae": ("score_mae", "mean"),
        "mean_score_rmse": ("score_rmse", "mean"),
        "mean_exact_category_agreement": ("exact_category_agreement_fraction", "mean"),
        "mean_near_category_agreement": ("near_category_agreement_fraction", "mean"),
        "pass_rate": ("passes_benchmark_thresholds", "mean"),
    }
    if "high_comfort_precision" in ok_cases.columns:
        aggregations["mean_high_comfort_precision"] = ("high_comfort_precision", "mean")
    if "high_comfort_recall" in ok_cases.columns:
        aggregations["mean_high_comfort_recall"] = ("high_comfort_recall", "mean")

    region_summary = ok_cases.groupby("region", as_index=False).agg(**aggregations).sort_values(
        ["mean_score_mae", "mean_score_rmse", "region"]
    )
    csv_path = output_dir / f"comfortwx_verify_benchmark_{stem}_region_summary.csv"
    region_summary.to_csv(csv_path, index=False)
    return region_summary, csv_path


def _write_region_summary_chart(region_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if region_summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(region_summary))
    width = 0.38
    ax.bar(
        positions - width / 2,
        region_summary["mean_score_mae"],
        width=width,
        color="#4f8ad9",
        label="Mean MAE",
    )
    ax.bar(
        positions + width / 2,
        region_summary["mean_score_rmse"],
        width=width,
        color="#c96b50",
        label="Mean RMSE",
    )
    ax.axhline(VERIFICATION_BENCHMARK_THRESHOLDS["score_mae_max"], color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(region_summary["region"], rotation=20, ha="right")
    ax.set_ylabel("Score error")
    ax.set_title("Regional Verification Summary")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_region_summary.png")


def _html_table(summary: pd.DataFrame) -> str:
    ordered_columns = [
        "region",
        "date",
        "forecast_lead_days",
        "benchmark_threshold_status",
        "score_bias_mean",
        "score_mae",
        "score_rmse",
        "exact_category_agreement_fraction",
        "near_category_agreement_fraction",
        "missed_high_comfort_cell_count",
        "false_high_comfort_cell_count",
    ]
    present_columns = [column for column in ordered_columns if column in summary.columns]
    display = summary.loc[:, present_columns].copy()
    return display.to_html(index=False, border=0, classes="summary-table")


def _priority_table(priority_cases: pd.DataFrame) -> str:
    if priority_cases.empty:
        return "<p>No priority cases available.</p>"
    display_columns = [
        "region",
        "date",
        "lead_label",
        "benchmark_threshold_status",
        "improvement_priority_score",
        "score_mae",
        "score_rmse",
        "problem_tags",
        "primary_component",
    ]
    return priority_cases.loc[:, display_columns].head(12).to_html(index=False, border=0, classes="summary-table")


def _write_benchmark_html_report(
    summary: pd.DataFrame,
    *,
    charts: dict[str, Path],
    benchmark_tier: str,
    region_summary: pd.DataFrame,
    region_summary_csv_path: Path | None,
    lead_summary: pd.DataFrame,
    lead_summary_csv_path: Path | None,
    region_lead_summary: pd.DataFrame,
    region_lead_summary_csv_path: Path | None,
    component_priority_summary: pd.DataFrame,
    component_priority_csv_path: Path | None,
    priority_cases: pd.DataFrame,
    priority_cases_csv_path: Path | None,
    calibration_summary: pd.DataFrame,
    calibration_summary_csv_path: Path | None,
    output_dir: Path,
    stem: str,
) -> Path:
    ok_cases = _ok_cases(summary)
    report_path = output_dir / f"comfortwx_verify_benchmark_{stem}.html"

    chart_titles = {
        "mae_bar_chart": "MAE by case",
        "agreement_bar_chart": "Category agreement by case",
        "bias_rmse_scatter": "Bias vs RMSE",
        "ranked_case_chart": "Ranked verification cases",
        "mae_timeseries_chart": "MAE over time",
        "agreement_timeseries_chart": "Agreement over time",
        "bias_timeseries_chart": "Bias over time",
        "component_heatmap": "Component MAE heatmap",
        "region_summary_chart": "Regional verification summary",
        "lead_summary_chart": "Forecast lead summary",
        "region_lead_heatmap": "Region and lead MAE heatmap",
        "component_priority_chart": "Components most worth improving",
        "priority_case_chart": "Top cases to improve first",
        "calibration_mae_chart": "Held-out calibration MAE by case",
        "calibration_lead_chart": "Held-out calibration MAE by lead",
    }
    chart_blocks = [
        f'<figure class="chart-card"><img src="{html.escape(path.name)}" alt="{html.escape(chart_titles.get(key, key))}"><figcaption>{html.escape(chart_titles.get(key, key))}</figcaption></figure>'
        for key, path in charts.items()
    ]

    case_cards: list[str] = []
    for _, row in ok_cases.sort_values(["forecast_lead_days", "date", "region"]).iterrows():
        thumbnails = []
        thumb_columns = [
            ("forecast_score_map_path", "Forecast score"),
            ("analysis_score_map_path", "Analysis score"),
            ("score_difference_map_path", "Score difference"),
            ("absolute_error_map_path", "Absolute error"),
            ("category_disagreement_map_path", "Category disagreement"),
            ("missed_high_comfort_map_path", "Missed high comfort"),
            ("false_high_comfort_map_path", "False high comfort"),
        ]
        for column, label in thumb_columns:
            if column in row and pd.notna(row[column]) and str(row[column]).strip():
                rel = Path(str(row[column])).name
                thumbnails.append(
                    f'<a class="thumb" href="{html.escape(rel)}"><img src="{html.escape(rel)}" alt="{html.escape(label)}"><span>{html.escape(label)}</span></a>'
                )
        case_cards.append(
            "\n".join(
                [
                    '<section class="case-card">',
                    f"<h3>{html.escape(str(row['region']))} | {row['date']:%Y-%m-%d} | {_lead_label(int(row['forecast_lead_days']))}</h3>",
                    (
                        f"<p>MAE {float(row['score_mae']):.2f} | RMSE {float(row['score_rmse']):.2f} | "
                        f"Near agreement {float(row['near_category_agreement_fraction']):.1%} | "
                        f"Status {html.escape(str(row['benchmark_threshold_status']))} | "
                        f"<a href=\"{html.escape(Path(str(row['component_metrics_csv_path'])).name)}\">component metrics CSV</a></p>"
                    ),
                    '<div class="thumb-grid">',
                    *thumbnails,
                    "</div>",
                    "</section>",
                ]
            )
        )

    ok_count = int((summary["status"] == "ok").sum()) if "status" in summary.columns else 0
    passing_count = int(summary.get("passes_benchmark_thresholds", pd.Series(dtype=bool)).fillna(False).sum())
    tier_label = benchmark_tier.replace("-", " ").title()
    if ok_cases.empty:
        best_case = ok_cases
        worst_case = ok_cases
    else:
        best_case = ok_cases.sort_values(["verification_rank_score", "score_mae"]).head(1)
        worst_case = ok_cases.sort_values(["verification_rank_score", "score_mae"], ascending=[False, False]).head(1)
    best_case_text = (
        f"Best case: {best_case.iloc[0]['region']} {best_case.iloc[0]['date']:%Y-%m-%d} {_lead_label(int(best_case.iloc[0]['forecast_lead_days']))} (MAE {best_case.iloc[0]['score_mae']:.2f})"
        if not best_case.empty
        else "Best case: n/a"
    )
    worst_case_text = (
        f"Worst case: {worst_case.iloc[0]['region']} {worst_case.iloc[0]['date']:%Y-%m-%d} {_lead_label(int(worst_case.iloc[0]['forecast_lead_days']))} (MAE {worst_case.iloc[0]['score_mae']:.2f})"
        if not worst_case.empty
        else "Worst case: n/a"
    )
    region_table_html = region_summary.to_html(index=False, border=0, classes="summary-table") if not region_summary.empty else "<p>No successful regional summary available.</p>"
    region_summary_link = (
        f"<p><a href='{html.escape(region_summary_csv_path.name)}'>Download regional summary CSV</a></p>"
        if region_summary_csv_path is not None
        else ""
    )
    lead_table_html = lead_summary.to_html(index=False, border=0, classes="summary-table") if not lead_summary.empty else "<p>No successful lead summary available.</p>"
    lead_summary_link = (
        f"<p><a href='{html.escape(lead_summary_csv_path.name)}'>Download forecast lead summary CSV</a></p>"
        if lead_summary_csv_path is not None
        else ""
    )
    region_lead_table_html = region_lead_summary.to_html(index=False, border=0, classes="summary-table") if not region_lead_summary.empty else "<p>No successful region/lead summary available.</p>"
    region_lead_summary_link = (
        f"<p><a href='{html.escape(region_lead_summary_csv_path.name)}'>Download region/lead summary CSV</a></p>"
        if region_lead_summary_csv_path is not None
        else ""
    )
    component_priority_table_html = component_priority_summary.to_html(index=False, border=0, classes="summary-table") if not component_priority_summary.empty else "<p>No component priority summary available.</p>"
    component_priority_link = (
        f"<p><a href='{html.escape(component_priority_csv_path.name)}'>Download component priority CSV</a></p>"
        if component_priority_csv_path is not None
        else ""
    )
    priority_cases_link = (
        f"<p><a href='{html.escape(priority_cases_csv_path.name)}'>Download ranked priority cases CSV</a></p>"
        if priority_cases_csv_path is not None
        else ""
    )
    calibration_summary_table_html = (
        calibration_summary.to_html(index=False, border=0, classes="summary-table")
        if not calibration_summary.empty
        else "<p>No held-out calibration summary available.</p>"
    )
    calibration_summary_link = (
        f"<p><a href='{html.escape(calibration_summary_csv_path.name)}'>Download held-out calibration summary CSV</a></p>"
        if calibration_summary_csv_path is not None
        else ""
    )
    report_path.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html lang='en'>",
                "<head>",
                "<meta charset='utf-8'>",
                "<title>Comfort Index Verification Benchmark</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 24px; color: #1c2330; background: #fafbfc; }",
                "h1, h2, h3 { margin-bottom: 0.35rem; }",
                ".meta { color: #556070; margin-bottom: 1.2rem; }",
                ".summary-table { border-collapse: collapse; width: 100%; background: white; }",
                ".summary-table th, .summary-table td { border: 1px solid #d9dee5; padding: 8px 10px; font-size: 0.92rem; }",
                ".summary-table th { background: #eef2f6; }",
                ".chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin: 22px 0; }",
                ".chart-card, .case-card { background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 12px; }",
                ".chart-card img { width: 100%; height: auto; display: block; }",
                ".chart-card figcaption { margin-top: 8px; color: #46505a; font-size: 0.9rem; }",
                ".thumb-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }",
                ".thumb { text-decoration: none; color: inherit; font-size: 0.84rem; }",
                ".thumb img { width: 100%; height: auto; border: 1px solid #d9dee5; border-radius: 6px; display: block; margin-bottom: 6px; }",
                ".status-line { display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0 16px; }",
                ".status-pill { background: #eef2f6; border-radius: 999px; padding: 6px 12px; font-size: 0.9rem; }",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Comfort Index Verification Benchmark</h1>",
                f"<p class='meta'>Tier: {html.escape(tier_label)} | Cases attempted: {len(summary)} | Successful cases: {ok_count} | Cases meeting thresholds: {passing_count}</p>",
                "<div class='status-line'>",
                f"<div class='status-pill'>MAE threshold: ≤ {VERIFICATION_BENCHMARK_THRESHOLDS['score_mae_max']:.1f}</div>",
                f"<div class='status-pill'>Near agreement threshold: ≥ {VERIFICATION_BENCHMARK_THRESHOLDS['near_category_agreement_min']:.0%}</div>",
                f"<div class='status-pill'>Absolute bias threshold: ≤ {VERIFICATION_BENCHMARK_THRESHOLDS['abs_score_bias_mean_max']:.1f}</div>",
                "</div>",
                f"<p>{html.escape(best_case_text)} | {html.escape(worst_case_text)}</p>",
                "<h2>Summary</h2>",
                _html_table(summary),
                "<h2>Regional Rollup</h2>",
                region_summary_link,
                region_table_html,
                "<h2>Forecast Lead Rollup</h2>",
                lead_summary_link,
                lead_table_html,
                "<h2>Region and Lead Rollup</h2>",
                region_lead_summary_link,
                region_lead_table_html,
                "<h2>Improvement Priorities</h2>",
                priority_cases_link,
                _priority_table(priority_cases),
                "<h2>Component Priorities</h2>",
                component_priority_link,
                component_priority_table_html,
                "<h2>Held-Out Calibration Review</h2>",
                calibration_summary_link,
                calibration_summary_table_html,
                "<h2>Benchmark Charts</h2>",
                "<div class='chart-grid'>",
                *chart_blocks,
                "</div>",
                "<h2>Case Maps</h2>",
                *case_cards,
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def _write_verification_site(
    *,
    summary: pd.DataFrame,
    summary_path: Path,
    charts: dict[str, Path],
    report_path: Path,
    region_summary_csv_path: Path | None,
    lead_summary_csv_path: Path | None,
    region_lead_summary_csv_path: Path | None,
    component_priority_csv_path: Path | None,
    priority_cases_csv_path: Path | None,
    calibration_summary_csv_path: Path | None,
    output_dir: Path,
    stem: str,
) -> Path:
    site_dir = output_dir / "verification_site" / stem
    site_dir.mkdir(parents=True, exist_ok=True)

    asset_paths: set[Path] = {summary_path, report_path}
    asset_paths.update(charts.values())
    if region_summary_csv_path is not None:
        asset_paths.add(region_summary_csv_path)
    if lead_summary_csv_path is not None:
        asset_paths.add(lead_summary_csv_path)
    if region_lead_summary_csv_path is not None:
        asset_paths.add(region_lead_summary_csv_path)
    if component_priority_csv_path is not None:
        asset_paths.add(component_priority_csv_path)
    if priority_cases_csv_path is not None:
        asset_paths.add(priority_cases_csv_path)
    if calibration_summary_csv_path is not None:
        asset_paths.add(calibration_summary_csv_path)

    for column in [
        "forecast_score_map_path",
        "analysis_score_map_path",
        "score_difference_map_path",
        "absolute_error_map_path",
        "category_disagreement_map_path",
        "missed_high_comfort_map_path",
        "false_high_comfort_map_path",
        "summary_csv_path",
        "point_metrics_csv_path",
        "component_metrics_csv_path",
        "request_summary_csv_path",
        "request_detail_csv_path",
    ]:
        if column in summary.columns:
            for value in summary[column].dropna():
                value_text = str(value).strip()
                if not value_text:
                    continue
                path = Path(value_text)
                if path.exists() and path.is_file():
                    asset_paths.add(path)

    for asset in asset_paths:
        if asset.exists() and asset.is_file():
            shutil.copy2(asset, site_dir / asset.name)

    primary_index = site_dir / "index.html"
    shutil.copy2(report_path, primary_index)

    latest_dir = output_dir / "verification_site" / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(site_dir, latest_dir)
    return primary_index


def run_verification_benchmark(
    *,
    cases: list[VerificationBenchmarkCase],
    output_dir: Path,
    mesh_profile: str,
    forecast_model: str,
    forecast_model_mode: str = "auto",
    analysis_model: str = OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    forecast_run_hour_utc: int,
    benchmark_tier: str,
    aggregation_policies: tuple[str, ...] = ("baseline",),
    case_cache_mode: str = "reuse",
    max_fresh_cases: int | None = None,
    case_cooldown_seconds: float = VERIFICATION_INCREMENTAL_CASE_COOLDOWN_SECONDS,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    fresh_case_count = 0
    effective_max_fresh_cases = _max_fresh_cases_for_tier(benchmark_tier) if max_fresh_cases is None else max(0, int(max_fresh_cases))
    for case in cases:
        attempted_fresh_case = False
        try:
            outputs_by_policy: dict[str, tuple[str, dict[str, object]]] = {}
            missing_policies: list[str] = []
            for aggregation_policy in aggregation_policies:
                outputs = None
                build_source = "fresh"
                if case_cache_mode == "reuse":
                    outputs = _existing_verification_case_outputs(
                        valid_date=case.valid_date,
                        region_name=case.region_name,
                        forecast_model=forecast_model,
                        forecast_model_mode=forecast_model_mode,
                        analysis_model=analysis_model,
                        forecast_lead_days=case.forecast_lead_days,
                        aggregation_policy=aggregation_policy,
                        output_dir=output_dir,
                    )
                    if outputs is not None:
                        build_source = "cache"
                if outputs is None:
                    missing_policies.append(aggregation_policy)
                else:
                    outputs_by_policy[aggregation_policy] = (build_source, outputs)

            if missing_policies and effective_max_fresh_cases and fresh_case_count >= effective_max_fresh_cases:
                for aggregation_policy in missing_policies:
                    records.append(
                        {
                            "benchmark_tier": benchmark_tier,
                            "verification_aggregation_policy": aggregation_policy,
                            "build_source": "deferred",
                            "region": case.region_name,
                            "date": case.valid_date.isoformat(),
                            "forecast_lead_days": case.forecast_lead_days,
                            "forecast_model": forecast_model,
                            "score_bias_mean": "",
                            "score_mae": "",
                            "score_rmse": "",
                            "exact_category_agreement_fraction": "",
                            "near_category_agreement_fraction": "",
                            "status": "deferred",
                            "error": f"Deferred after reaching fresh-case cap ({effective_max_fresh_cases}) for this run.",
                            "forecast_score_map_path": "",
                            "analysis_score_map_path": "",
                            "score_difference_map_path": "",
                            "absolute_error_map_path": "",
                            "category_disagreement_map_path": "",
                            "missed_high_comfort_map_path": "",
                            "false_high_comfort_map_path": "",
                            "forecast_daily_fields_path": "",
                            "analysis_daily_fields_path": "",
                            "summary_csv_path": "",
                            "point_metrics_csv_path": "",
                            "component_metrics_csv_path": "",
                            "request_summary_csv_path": "",
                            "request_detail_csv_path": "",
                        }
                    )
                for aggregation_policy, (build_source, outputs) in outputs_by_policy.items():
                    summary_record = dict(outputs["summary_record"])
                    summary_record.update(
                        {
                            "benchmark_tier": benchmark_tier,
                            "build_source": build_source,
                            "region": case.region_name,
                            "date": case.valid_date.isoformat(),
                            "forecast_lead_days": case.forecast_lead_days,
                            "status": "ok",
                            "forecast_score_map_path": str(outputs["forecast_score_map"]),
                            "analysis_score_map_path": str(outputs["analysis_score_map"]),
                            "score_difference_map_path": str(outputs["score_difference_map"]),
                            "absolute_error_map_path": str(outputs["absolute_error_map"]),
                            "category_disagreement_map_path": str(outputs["category_disagreement_map"]),
                            "missed_high_comfort_map_path": str(outputs["missed_high_comfort_map"]),
                            "false_high_comfort_map_path": str(outputs["false_high_comfort_map"]),
                            "forecast_daily_fields_path": str(outputs["forecast_daily_fields"]),
                            "analysis_daily_fields_path": str(outputs["analysis_daily_fields"]),
                            "summary_csv_path": str(outputs["summary_csv"]),
                            "point_metrics_csv_path": str(outputs["point_metrics_csv"]),
                            "component_metrics_csv_path": str(outputs["component_metrics_csv"]),
                            "request_summary_csv_path": str(outputs["request_summary_csv"]),
                            "request_detail_csv_path": str(outputs["request_detail_csv"]),
                        }
                    )
                    records.append(summary_record)
                continue

            for aggregation_policy in missing_policies:
                attempted_fresh_case = True
                outputs = run_verification(
                    valid_date=case.valid_date,
                    region_name=case.region_name,
                    output_dir=output_dir,
                    mesh_profile=mesh_profile,
                    forecast_model=forecast_model,
                    forecast_model_mode=forecast_model_mode,
                    analysis_model=analysis_model,
                    forecast_run_hour_utc=forecast_run_hour_utc,
                    forecast_lead_days=case.forecast_lead_days,
                    aggregation_policy=aggregation_policy,
                    workflow_name="verification_benchmark",
                )
                outputs_by_policy[aggregation_policy] = ("fresh", outputs)

            if missing_policies:
                fresh_case_count += 1
                if case_cooldown_seconds > 0:
                    time.sleep(case_cooldown_seconds)

            for aggregation_policy in aggregation_policies:
                build_source, outputs = outputs_by_policy[aggregation_policy]
                summary_record = dict(outputs["summary_record"])
                summary_record.update(
                    {
                        "benchmark_tier": benchmark_tier,
                        "build_source": build_source,
                        "region": case.region_name,
                        "date": case.valid_date.isoformat(),
                        "forecast_lead_days": case.forecast_lead_days,
                        "status": "ok",
                        "forecast_score_map_path": str(outputs["forecast_score_map"]),
                        "analysis_score_map_path": str(outputs["analysis_score_map"]),
                        "score_difference_map_path": str(outputs["score_difference_map"]),
                        "absolute_error_map_path": str(outputs["absolute_error_map"]),
                        "category_disagreement_map_path": str(outputs["category_disagreement_map"]),
                        "missed_high_comfort_map_path": str(outputs["missed_high_comfort_map"]),
                        "false_high_comfort_map_path": str(outputs["false_high_comfort_map"]),
                        "forecast_daily_fields_path": str(outputs["forecast_daily_fields"]),
                        "analysis_daily_fields_path": str(outputs["analysis_daily_fields"]),
                        "summary_csv_path": str(outputs["summary_csv"]),
                        "point_metrics_csv_path": str(outputs["point_metrics_csv"]),
                        "component_metrics_csv_path": str(outputs["component_metrics_csv"]),
                        "request_summary_csv_path": str(outputs["request_summary_csv"]),
                        "request_detail_csv_path": str(outputs["request_detail_csv"]),
                    }
                )
                records.append(summary_record)
        except Exception as exc:
            if attempted_fresh_case and case_cooldown_seconds > 0:
                time.sleep(case_cooldown_seconds)
            for aggregation_policy in aggregation_policies:
                records.append(
                    {
                        "benchmark_tier": benchmark_tier,
                        "verification_aggregation_policy": aggregation_policy,
                        "build_source": "error",
                        "region": case.region_name,
                        "date": case.valid_date.isoformat(),
                        "forecast_lead_days": case.forecast_lead_days,
                        "forecast_model": forecast_model,
                        "score_bias_mean": "",
                        "score_mae": "",
                        "score_rmse": "",
                        "exact_category_agreement_fraction": "",
                        "near_category_agreement_fraction": "",
                        "status": "error",
                        "error": str(exc),
                        "forecast_score_map_path": "",
                        "analysis_score_map_path": "",
                        "score_difference_map_path": "",
                        "absolute_error_map_path": "",
                        "category_disagreement_map_path": "",
                        "missed_high_comfort_map_path": "",
                        "false_high_comfort_map_path": "",
                        "forecast_daily_fields_path": "",
                        "analysis_daily_fields_path": "",
                        "summary_csv_path": "",
                        "point_metrics_csv_path": "",
                        "component_metrics_csv_path": "",
                        "request_summary_csv_path": "",
                        "request_detail_csv_path": "",
                    }
                )
    return _apply_threshold_flags(pd.DataFrame.from_records(records))


def _write_aggregation_policy_summary(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> tuple[pd.DataFrame, Path | None]:
    ok = _ok_cases(summary)
    if ok.empty or "verification_aggregation_policy" not in ok.columns or ok["verification_aggregation_policy"].nunique() <= 1:
        return pd.DataFrame(), None
    policy_summary = (
        ok.groupby(["forecast_lead_days", "verification_aggregation_policy"], dropna=False)
        .agg(
            case_count=("case_label", "count"),
            mean_score_bias=("score_bias_mean", "mean"),
            mean_score_mae=("score_mae", "mean"),
            mean_score_rmse=("score_rmse", "mean"),
            mean_exact_category_agreement=("exact_category_agreement_fraction", "mean"),
            mean_near_category_agreement=("near_category_agreement_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["forecast_lead_days", "mean_score_mae", "verification_aggregation_policy"])
    )
    baseline = policy_summary.loc[
        policy_summary["verification_aggregation_policy"] == "baseline",
        ["forecast_lead_days", "mean_score_mae", "mean_score_rmse", "mean_near_category_agreement"],
    ].rename(
        columns={
            "mean_score_mae": "baseline_mean_score_mae",
            "mean_score_rmse": "baseline_mean_score_rmse",
            "mean_near_category_agreement": "baseline_mean_near_category_agreement",
        }
    )
    if not baseline.empty:
        policy_summary = policy_summary.merge(baseline, on="forecast_lead_days", how="left")
        policy_summary["score_mae_improvement_vs_baseline"] = (
            policy_summary["baseline_mean_score_mae"] - policy_summary["mean_score_mae"]
        ).round(3)
        policy_summary["score_rmse_improvement_vs_baseline"] = (
            policy_summary["baseline_mean_score_rmse"] - policy_summary["mean_score_rmse"]
        ).round(3)
        policy_summary["near_category_agreement_change_vs_baseline"] = (
            policy_summary["mean_near_category_agreement"] - policy_summary["baseline_mean_near_category_agreement"]
        ).round(4)
    csv_path = output_dir / f"comfortwx_verify_benchmark_{stem}_aggregation_policy_summary.csv"
    policy_summary.to_csv(csv_path, index=False)
    return policy_summary, csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small benchmark suite for proxy Comfort Index verification.")
    parser.add_argument(
        "--benchmark-tier",
        default=VERIFICATION_BENCHMARK_TIER_DEFAULT,
        choices=list(VERIFICATION_BENCHMARK_TIERS),
        help="Benchmark case tier. Default: default.",
    )
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD override for all benchmark regions.")
    parser.add_argument("--mesh-profile", default="standard", help="Regional mesh profile. Default: standard.")
    parser.add_argument("--forecast-model", default=OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT)
    parser.add_argument("--forecast-model-mode", choices=("auto", "exact"), default="auto")
    parser.add_argument("--analysis-model", default=OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT)
    parser.add_argument("--forecast-run-hour-utc", type=int, default=OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC)
    parser.add_argument(
        "--case-cache-mode",
        default="reuse",
        choices=["reuse", "refresh"],
        help="Reuse existing per-case verification artifacts when available. Default: reuse.",
    )
    parser.add_argument(
        "--lead-days",
        default=",".join(str(value) for value in OPENMETEO_VERIFICATION_BENCHMARK_LEAD_DAYS),
        help="Comma-separated verification forecast lead days. Default: 1,2,3,7.",
    )
    parser.add_argument(
        "--aggregation-policies",
        default="baseline",
        help="Comma-separated verification aggregation policies to compare. Default: baseline.",
    )
    parser.add_argument(
        "--regions",
        default=None,
        help="Optional comma-separated region filter for incremental benchmark runs.",
    )
    parser.add_argument(
        "--dates",
        default=None,
        help="Optional comma-separated YYYY-MM-DD filter for incremental benchmark runs.",
    )
    parser.add_argument(
        "--max-fresh-cases",
        type=int,
        default=None,
        help="Optional cap on uncached verification cases fetched in this run. Defaults to tier policy.",
    )
    parser.add_argument(
        "--case-cooldown-seconds",
        type=float,
        default=VERIFICATION_INCREMENTAL_CASE_COOLDOWN_SECONDS,
        help="Sleep between uncached verification cases to reduce burst rate. Default: 4.0.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def _parse_lead_days(value: str) -> tuple[int, ...]:
    lead_days = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not lead_days:
        raise ValueError("At least one lead day is required.")
    if any(day < 1 for day in lead_days):
        raise ValueError("Lead days must be positive integers.")
    return tuple(dict.fromkeys(lead_days))


def _parse_region_filter(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    regions = tuple(part.strip() for part in value.split(",") if part.strip())
    return regions or None


def _parse_date_filter(value: str | None) -> tuple[date, ...] | None:
    if value is None:
        return None
    dates = tuple(datetime.strptime(part.strip(), "%Y-%m-%d").date() for part in value.split(",") if part.strip())
    return dates or None


def main() -> None:
    args = _parse_args()
    date_override = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    lead_days = _parse_lead_days(args.lead_days)
    aggregation_policies = _parse_aggregation_policies(args.aggregation_policies)
    benchmark_tier = args.benchmark_tier
    cases = _filter_cases(
        _resolved_cases(date_override, benchmark_tier, lead_days),
        region_filter=_parse_region_filter(args.regions),
        date_filter=_parse_date_filter(args.dates),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_verification_benchmark(
        cases=cases,
        output_dir=output_dir,
        mesh_profile=args.mesh_profile,
        forecast_model=args.forecast_model,
        forecast_model_mode=args.forecast_model_mode,
        analysis_model=args.analysis_model,
        forecast_run_hour_utc=args.forecast_run_hour_utc,
        benchmark_tier=benchmark_tier,
        aggregation_policies=aggregation_policies,
        case_cache_mode=args.case_cache_mode,
        max_fresh_cases=args.max_fresh_cases,
        case_cooldown_seconds=args.case_cooldown_seconds,
    )

    lead_stem = "d" + "-".join(str(day) for day in lead_days)
    tier_stem = benchmark_tier.replace("-", "_")
    stem = (
        f"{date_override.isoformat().replace('-', '')}_{tier_stem}_{lead_stem}"
        if date_override
        else f"{tier_stem}_cases_{lead_stem}"
    )
    summary_path = output_dir / f"comfortwx_verify_benchmark_{stem}.csv"
    summary.to_csv(summary_path, index=False)
    baseline_summary = summary
    if "verification_aggregation_policy" in summary.columns and len(aggregation_policies) > 1:
        baseline_only = summary.loc[summary["verification_aggregation_policy"] == "baseline"].copy()
        if not baseline_only.empty:
            baseline_summary = baseline_only
    policy_summary, aggregation_policy_summary_csv_path = _write_aggregation_policy_summary(
        summary,
        output_dir=output_dir,
        stem=stem,
    )
    charts = _write_benchmark_charts(baseline_summary, output_dir=output_dir, stem=stem)
    component_heatmap = _write_component_heatmap(baseline_summary, output_dir=output_dir, stem=stem)
    if component_heatmap is not None:
        charts["component_heatmap"] = component_heatmap
    region_summary, region_summary_csv_path = _write_region_summary(baseline_summary, output_dir=output_dir, stem=stem)
    region_summary_chart = _write_region_summary_chart(region_summary, output_dir=output_dir, stem=stem)
    if region_summary_chart is not None:
        charts["region_summary_chart"] = region_summary_chart
    lead_summary, lead_summary_csv_path = _write_lead_summary(baseline_summary, output_dir=output_dir, stem=stem)
    lead_summary_chart = _write_lead_summary_chart(lead_summary, output_dir=output_dir, stem=stem)
    if lead_summary_chart is not None:
        charts["lead_summary_chart"] = lead_summary_chart
    region_lead_summary, region_lead_summary_csv_path = _write_region_lead_summary(baseline_summary, output_dir=output_dir, stem=stem)
    region_lead_heatmap = _write_region_lead_heatmap(region_lead_summary, output_dir=output_dir, stem=stem)
    if region_lead_heatmap is not None:
        charts["region_lead_heatmap"] = region_lead_heatmap
    component_priority_summary, component_priority_csv_path = _write_component_priority_summary(baseline_summary, output_dir=output_dir, stem=stem)
    component_priority_chart = _write_component_priority_chart(component_priority_summary, output_dir=output_dir, stem=stem)
    if component_priority_chart is not None:
        charts["component_priority_chart"] = component_priority_chart
    priority_cases, priority_cases_csv_path = _write_priority_case_summary(baseline_summary, output_dir=output_dir, stem=stem)
    priority_case_chart = _write_priority_case_chart(priority_cases, output_dir=output_dir, stem=stem)
    if priority_case_chart is not None:
        charts["priority_case_chart"] = priority_case_chart
    calibration_summary, calibration_summary_csv_path = _build_calibration_summary(baseline_summary, output_dir=output_dir, stem=stem)
    calibration_mae_chart = _write_calibration_mae_chart(calibration_summary, output_dir=output_dir, stem=stem)
    if calibration_mae_chart is not None:
        charts["calibration_mae_chart"] = calibration_mae_chart
    calibration_lead_chart = _write_calibration_lead_chart(calibration_summary, output_dir=output_dir, stem=stem)
    if calibration_lead_chart is not None:
        charts["calibration_lead_chart"] = calibration_lead_chart
    report_path = _write_benchmark_html_report(
        baseline_summary,
        charts=charts,
        benchmark_tier=benchmark_tier,
        region_summary=region_summary,
        region_summary_csv_path=region_summary_csv_path,
        lead_summary=lead_summary,
        lead_summary_csv_path=lead_summary_csv_path,
        region_lead_summary=region_lead_summary,
        region_lead_summary_csv_path=region_lead_summary_csv_path,
        component_priority_summary=component_priority_summary,
        component_priority_csv_path=component_priority_csv_path,
        priority_cases=priority_cases,
        priority_cases_csv_path=priority_cases_csv_path,
        calibration_summary=calibration_summary,
        calibration_summary_csv_path=calibration_summary_csv_path,
        output_dir=output_dir,
        stem=stem,
    )
    verification_site_index = _write_verification_site(
        summary=baseline_summary,
        summary_path=summary_path,
        charts=charts,
        report_path=report_path,
        region_summary_csv_path=region_summary_csv_path,
        lead_summary_csv_path=lead_summary_csv_path,
        region_lead_summary_csv_path=region_lead_summary_csv_path,
        component_priority_csv_path=component_priority_csv_path,
        priority_cases_csv_path=priority_cases_csv_path,
        calibration_summary_csv_path=calibration_summary_csv_path,
        output_dir=output_dir,
        stem=stem,
    )

    print(f"Saved verification benchmark summary: {summary_path}")
    if aggregation_policy_summary_csv_path is not None:
        print(f"Saved aggregation policy comparison summary: {aggregation_policy_summary_csv_path}")
    if region_summary_csv_path is not None:
        print(f"Saved regional benchmark summary: {region_summary_csv_path}")
    if lead_summary_csv_path is not None:
        print(f"Saved lead benchmark summary: {lead_summary_csv_path}")
    if region_lead_summary_csv_path is not None:
        print(f"Saved region/lead benchmark summary: {region_lead_summary_csv_path}")
    if component_priority_csv_path is not None:
        print(f"Saved component priority summary: {component_priority_csv_path}")
    if priority_cases_csv_path is not None:
        print(f"Saved ranked priority cases: {priority_cases_csv_path}")
    if calibration_summary_csv_path is not None:
        print(f"Saved held-out calibration summary: {calibration_summary_csv_path}")
    for chart_name, path in charts.items():
        print(f"Saved {chart_name}: {path}")
    print(f"Saved verification benchmark report: {report_path}")
    print(f"Saved verification site index: {verification_site_index}")
    print(format_benchmark_table(summary))


if __name__ == "__main__":
    main()
