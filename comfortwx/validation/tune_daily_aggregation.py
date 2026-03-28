"""Held-out tuning harness for daily aggregation modes."""

from __future__ import annotations

import argparse
import html
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
    OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC,
    OUTPUT_DIR,
    VERIFICATION_AGGREGATION_EXPERIMENTAL_POLICIES,
    VERIFICATION_AGGREGATION_TUNING_CANDIDATE_MODES,
    VERIFICATION_HOURLY_CACHE_VERSION,
    VERIFICATION_INCREMENTAL_CASE_COOLDOWN_SECONDS,
    resolve_verification_aggregation_mode,
)
from comfortwx.data.openmeteo_reliability import openmeteo_request_context
from comfortwx.data.openmeteo_verification import (
    OpenMeteoVerificationRegionalLoader,
    resolve_openmeteo_verification_forecast_model,
)
from comfortwx.scoring.daily import aggregate_daily_scores
from comfortwx.scoring.hourly import score_hourly_dataset
from comfortwx.validation.verify_benchmark import (
    _filter_cases,
    _max_fresh_cases_for_tier,
    _parse_date_filter,
    _parse_region_filter,
    _resolved_cases,
)
from comfortwx.validation.verify_benchmark_cases import VERIFICATION_BENCHMARK_TIER_DEFAULT
from comfortwx.validation.verify_benchmark_cases import VERIFICATION_BENCHMARK_TIERS


def _parse_lead_days(value: str) -> tuple[int, ...]:
    lead_days = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not lead_days:
        raise ValueError("At least one lead day is required.")
    if any(day < 1 for day in lead_days):
        raise ValueError("Lead days must be positive integers.")
    return tuple(dict.fromkeys(lead_days))


def _parse_candidate_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes:
        raise ValueError("At least one aggregation mode is required.")
    return tuple(dict.fromkeys(modes))


def _case_label(region_name: str, valid_date: date, forecast_lead_days: int) -> str:
    return f"{region_name} {valid_date:%Y-%m-%d} D+{forecast_lead_days}"


def _hourly_cache_paths(
    *,
    cache_dir: Path,
    region_name: str,
    valid_date: date,
    forecast_model: str,
    forecast_model_mode: str = "auto",
    analysis_model: str = OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    forecast_lead_days: int,
) -> tuple[Path, Path]:
    resolved_forecast_model = resolve_openmeteo_verification_forecast_model(
        requested_model=forecast_model,
        forecast_lead_days=forecast_lead_days,
        forecast_model_mode=forecast_model_mode,
        region_name=region_name,
    )
    normalized_analysis_model = "".join(character if character.isalnum() else "_" for character in analysis_model.strip().lower()).strip("_")
    prefix = f"comfortwx_verify_{region_name}_{resolved_forecast_model}_{normalized_analysis_model}_d{forecast_lead_days}_{VERIFICATION_HOURLY_CACHE_VERSION}"
    stem = f"{valid_date:%Y%m%d}"
    return (
        cache_dir / f"{prefix}_forecast_hourly_scored_{stem}.nc",
        cache_dir / f"{prefix}_analysis_hourly_scored_{stem}.nc",
    )


def _load_or_build_hourly_case(
    *,
    valid_date: date,
    region_name: str,
    mesh_profile: str,
    forecast_model: str,
    forecast_model_mode: str = "auto",
    analysis_model: str = OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    forecast_run_hour_utc: int,
    forecast_lead_days: int,
    cache_dir: Path,
    case_cache_mode: str,
) -> tuple[xr.Dataset, xr.Dataset, str]:
    forecast_cache_path, analysis_cache_path = _hourly_cache_paths(
        cache_dir=cache_dir,
        region_name=region_name,
        valid_date=valid_date,
        forecast_model=forecast_model,
        forecast_model_mode=forecast_model_mode,
        analysis_model=analysis_model,
        forecast_lead_days=forecast_lead_days,
    )
    if (
        case_cache_mode == "reuse"
        and forecast_cache_path.exists()
        and analysis_cache_path.exists()
    ):
        return xr.load_dataset(forecast_cache_path), xr.load_dataset(analysis_cache_path), "cache"

    loader = OpenMeteoVerificationRegionalLoader(
        region_name=region_name,
        mesh_profile=mesh_profile,
        forecast_model=forecast_model,
        forecast_model_mode=forecast_model_mode,
        analysis_model=analysis_model,
        forecast_run_hour_utc=forecast_run_hour_utc,
        forecast_lead_days=forecast_lead_days,
    )
    with openmeteo_request_context(
        workflow="verification_tuning",
        label=f"region={region_name};date={valid_date.isoformat()};lead={forecast_lead_days}",
    ):
        forecast_hourly, analysis_hourly, _ = loader.load_pair(valid_date)
    forecast_scored_hourly = score_hourly_dataset(forecast_hourly)
    analysis_scored_hourly = score_hourly_dataset(analysis_hourly)
    cache_dir.mkdir(parents=True, exist_ok=True)
    forecast_scored_hourly.to_netcdf(forecast_cache_path)
    analysis_scored_hourly.to_netcdf(analysis_cache_path)
    return forecast_scored_hourly, analysis_scored_hourly, "fresh"


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


def evaluate_daily_aggregation_modes(
    *,
    cases: list,
    output_dir: Path,
    mesh_profile: str,
    forecast_model: str,
    forecast_model_mode: str = "auto",
    analysis_model: str = OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT,
    forecast_run_hour_utc: int,
    candidate_modes: tuple[str, ...],
    case_cache_mode: str,
    benchmark_tier: str,
    max_fresh_cases: int | None,
    case_cooldown_seconds: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    cache_dir = output_dir / "verification_hourly_cache"
    fresh_case_count = 0
    effective_max_fresh_cases = _max_fresh_cases_for_tier(benchmark_tier) if max_fresh_cases is None else max(0, int(max_fresh_cases))

    for case in cases:
        attempted_fresh_case = False
        try:
            forecast_cache_path, analysis_cache_path = _hourly_cache_paths(
                cache_dir=cache_dir,
                region_name=case.region_name,
                valid_date=case.valid_date,
                forecast_model=forecast_model,
                forecast_model_mode=forecast_model_mode,
                analysis_model=analysis_model,
                forecast_lead_days=case.forecast_lead_days,
            )
            cached_case_available = (
                case_cache_mode == "reuse"
                and forecast_cache_path.exists()
                and analysis_cache_path.exists()
            )
            if not cached_case_available and effective_max_fresh_cases and fresh_case_count >= effective_max_fresh_cases:
                for aggregation_mode in candidate_modes:
                    records.append(
                        {
                            "benchmark_tier": benchmark_tier,
                            "region": case.region_name,
                            "date": case.valid_date.isoformat(),
                            "forecast_lead_days": case.forecast_lead_days,
                            "case_label": _case_label(case.region_name, case.valid_date, case.forecast_lead_days),
                            "aggregation_mode": aggregation_mode,
                            "build_source": "deferred",
                            "status": "deferred",
                            "error": f"Deferred after reaching fresh-case cap ({effective_max_fresh_cases}) for this run.",
                        }
                    )
                continue
            forecast_scored_hourly, analysis_scored_hourly, build_source = _load_or_build_hourly_case(
                valid_date=case.valid_date,
                region_name=case.region_name,
                mesh_profile=mesh_profile,
                forecast_model=forecast_model,
                forecast_model_mode=forecast_model_mode,
                analysis_model=analysis_model,
                forecast_run_hour_utc=forecast_run_hour_utc,
                forecast_lead_days=case.forecast_lead_days,
                cache_dir=cache_dir,
                case_cache_mode=case_cache_mode,
            )
            attempted_fresh_case = build_source == "fresh"
            if attempted_fresh_case:
                fresh_case_count += 1
                if case_cooldown_seconds > 0:
                    time.sleep(case_cooldown_seconds)
            for aggregation_mode in candidate_modes:
                forecast_daily = aggregate_daily_scores(forecast_scored_hourly, aggregation_mode=aggregation_mode)
                analysis_daily = aggregate_daily_scores(analysis_scored_hourly, aggregation_mode=aggregation_mode)
                metrics = _daily_score_metrics(forecast_daily, analysis_daily)
                records.append(
                    {
                        "benchmark_tier": benchmark_tier,
                        "region": case.region_name,
                        "date": case.valid_date.isoformat(),
                        "forecast_lead_days": case.forecast_lead_days,
                        "case_label": _case_label(case.region_name, case.valid_date, case.forecast_lead_days),
                        "aggregation_mode": aggregation_mode,
                        "build_source": build_source,
                        "status": "ok",
                        **metrics,
                    }
                )
        except Exception as exc:
            if attempted_fresh_case and case_cooldown_seconds > 0:
                time.sleep(case_cooldown_seconds)
            for aggregation_mode in candidate_modes:
                records.append(
                    {
                        "benchmark_tier": benchmark_tier,
                        "region": case.region_name,
                        "date": case.valid_date.isoformat(),
                        "forecast_lead_days": case.forecast_lead_days,
                        "case_label": _case_label(case.region_name, case.valid_date, case.forecast_lead_days),
                        "aggregation_mode": aggregation_mode,
                        "build_source": "error",
                        "status": "error",
                        "error": str(exc),
                    }
                )

    return pd.DataFrame.from_records(records)


def _ok_case_scores(case_scores: pd.DataFrame) -> pd.DataFrame:
    ok = case_scores.loc[case_scores["status"] == "ok"].copy()
    if ok.empty:
        return ok
    ok["date"] = pd.to_datetime(ok["date"])
    ok["forecast_lead_days"] = ok["forecast_lead_days"].astype(int)
    ok["score_mae"] = ok["score_mae"].astype(float)
    ok["score_rmse"] = ok["score_rmse"].astype(float)
    ok["score_bias_mean"] = ok["score_bias_mean"].astype(float)
    ok["exact_category_agreement_fraction"] = ok["exact_category_agreement_fraction"].astype(float)
    ok["near_category_agreement_fraction"] = ok["near_category_agreement_fraction"].astype(float)
    return ok


def summarize_candidate_modes(case_scores: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_case_scores(case_scores)
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby(["forecast_lead_days", "aggregation_mode"], dropna=False)
        .agg(
            case_count=("case_label", "count"),
            mean_score_mae=("score_mae", "mean"),
            mean_score_rmse=("score_rmse", "mean"),
            mean_abs_bias=("score_bias_mean", lambda values: float(np.mean(np.abs(values)))),
            mean_exact_category_agreement=("exact_category_agreement_fraction", "mean"),
            mean_near_category_agreement=("near_category_agreement_fraction", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["forecast_lead_days", "mean_score_mae", "mean_score_rmse", "aggregation_mode"],
        ascending=[True, True, True, True],
    )


def recommend_modes_by_lead(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    if candidate_summary.empty:
        return pd.DataFrame()

    ranked = candidate_summary.sort_values(
        [
            "forecast_lead_days",
            "mean_score_mae",
            "mean_score_rmse",
            "mean_abs_bias",
            "mean_near_category_agreement",
        ],
        ascending=[True, True, True, True, False],
    )
    recommended = ranked.groupby("forecast_lead_days", as_index=False).head(1).copy()
    return recommended.rename(columns={"aggregation_mode": "recommended_aggregation_mode"})


def build_holdout_mode_selection(case_scores: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_case_scores(case_scores)
    if ok.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for case_label in ok["case_label"].drop_duplicates():
        case_rows = ok.loc[ok["case_label"] == case_label].copy()
        if case_rows.empty:
            continue
        lead_day = int(case_rows["forecast_lead_days"].iloc[0])
        region_name = str(case_rows["region"].iloc[0])
        baseline_row = case_rows.loc[case_rows["aggregation_mode"] == "baseline"]
        if baseline_row.empty:
            continue

        training_rows = ok.loc[ok["case_label"] != case_label].copy()
        lead_training = training_rows.loc[training_rows["forecast_lead_days"] == lead_day].copy()
        region_lead_training = lead_training.loc[lead_training["region"] == region_name].copy()
        if len(region_lead_training["case_label"].drop_duplicates()) >= 2:
            selection_scope = "region+lead"
            selected_training = region_lead_training
        else:
            selection_scope = "lead"
            selected_training = lead_training
        if selected_training.empty:
            continue

        ranked_modes = (
            selected_training.groupby("aggregation_mode", dropna=False)
            .agg(
                training_case_count=("case_label", lambda values: int(pd.Series(values).nunique())),
                training_mean_score_mae=("score_mae", "mean"),
                training_mean_score_rmse=("score_rmse", "mean"),
                training_mean_abs_bias=("score_bias_mean", lambda values: float(np.mean(np.abs(values)))),
                training_mean_near_agreement=("near_category_agreement_fraction", "mean"),
            )
            .reset_index()
            .sort_values(
                [
                    "training_mean_score_mae",
                    "training_mean_score_rmse",
                    "training_mean_abs_bias",
                    "training_mean_near_agreement",
                    "aggregation_mode",
                ],
                ascending=[True, True, True, False, True],
            )
        )
        selected_mode = str(ranked_modes.iloc[0]["aggregation_mode"])
        selected_case_row = case_rows.loc[case_rows["aggregation_mode"] == selected_mode].iloc[0]
        baseline_case_row = baseline_row.iloc[0]
        records.append(
            {
                "case_label": case_label,
                "region": region_name,
                "date": case_rows["date"].iloc[0].date().isoformat(),
                "forecast_lead_days": lead_day,
                "selection_scope": selection_scope,
                "selected_aggregation_mode": selected_mode,
                "training_case_count": int(ranked_modes.iloc[0]["training_case_count"]),
                "baseline_score_mae": float(baseline_case_row["score_mae"]),
                "selected_score_mae": float(selected_case_row["score_mae"]),
                "score_mae_improvement": round(
                    float(baseline_case_row["score_mae"]) - float(selected_case_row["score_mae"]),
                    2,
                ),
                "baseline_score_rmse": float(baseline_case_row["score_rmse"]),
                "selected_score_rmse": float(selected_case_row["score_rmse"]),
                "score_rmse_improvement": round(
                    float(baseline_case_row["score_rmse"]) - float(selected_case_row["score_rmse"]),
                    2,
                ),
                "baseline_near_category_agreement_fraction": float(
                    baseline_case_row["near_category_agreement_fraction"]
                ),
                "selected_near_category_agreement_fraction": float(
                    selected_case_row["near_category_agreement_fraction"]
                ),
                "near_category_agreement_improvement": round(
                    float(selected_case_row["near_category_agreement_fraction"])
                    - float(baseline_case_row["near_category_agreement_fraction"]),
                    4,
                ),
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records).sort_values(
        ["forecast_lead_days", "score_mae_improvement", "region", "date"],
        ascending=[True, False, True, True],
    )


def _resolve_policy_mode(
    policy_definition: dict[str, object],
    *,
    policy_name: str,
    region_name: str,
    lead_day: int,
    valid_date: date,
) -> str:
    # Runtime policies should resolve through the shared config helper so
    # verification runners and tuning compare the same effective modes.
    # Test helpers and ad hoc comparisons can still inject local policy
    # definitions that are not registered in config.
    if policy_name in VERIFICATION_AGGREGATION_EXPERIMENTAL_POLICIES:
        return resolve_verification_aggregation_mode(
            policy_name=policy_name,
            region_name=region_name,
            valid_date=valid_date,
            forecast_lead_days=lead_day,
        )

    regime_definitions = policy_definition.get("calendar_regimes", {})
    selected_definition = policy_definition
    if isinstance(regime_definitions, dict):
        regime_name = "cool_season" if valid_date.month in (11, 12, 1, 2) else "warm_season"
        regime_definition = regime_definitions.get(regime_name)
        if isinstance(regime_definition, dict):
            selected_definition = {
                "default": policy_definition.get("default", {}),
                "regions": policy_definition.get("regions", {}),
                **regime_definition,
            }

    default_modes = selected_definition.get("default", {})
    region_overrides = selected_definition.get("regions", {})
    if not isinstance(default_modes, dict):
        raise ValueError("Verification aggregation policy default modes must be a mapping.")
    if not isinstance(region_overrides, dict):
        raise ValueError("Verification aggregation policy region overrides must be a mapping.")

    regional_modes = region_overrides.get(region_name, {})
    if isinstance(regional_modes, dict) and lead_day in regional_modes:
        return str(regional_modes[lead_day])
    if lead_day in default_modes:
        return str(default_modes[lead_day])
    raise ValueError(f"Policy '{policy_name}' does not define a mode for lead day {lead_day}.")


def build_policy_comparison(
    case_scores: pd.DataFrame,
    *,
    policy_definitions: dict[str, dict[str, object]] = VERIFICATION_AGGREGATION_EXPERIMENTAL_POLICIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = _ok_case_scores(case_scores)
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    policy_case_records: list[dict[str, object]] = []
    for case_label in ok["case_label"].drop_duplicates():
        case_rows = ok.loc[ok["case_label"] == case_label].copy()
        if case_rows.empty:
            continue
        lead_day = int(case_rows["forecast_lead_days"].iloc[0])
        region_name = str(case_rows["region"].iloc[0])
        for policy_name, policy_modes in policy_definitions.items():
            aggregation_mode = _resolve_policy_mode(
                policy_modes,
                policy_name=policy_name,
                region_name=region_name,
                lead_day=lead_day,
                valid_date=case_rows["date"].iloc[0].date(),
            )
            matched_row = case_rows.loc[case_rows["aggregation_mode"] == aggregation_mode]
            if matched_row.empty:
                continue
            selected_row = matched_row.iloc[0]
            policy_case_records.append(
                {
                    "policy_name": policy_name,
                    "aggregation_mode": aggregation_mode,
                    "case_label": case_label,
                    "region": str(selected_row["region"]),
                    "date": selected_row["date"].date().isoformat(),
                    "forecast_lead_days": lead_day,
                    "score_bias_mean": float(selected_row["score_bias_mean"]),
                    "score_mae": float(selected_row["score_mae"]),
                    "score_rmse": float(selected_row["score_rmse"]),
                    "exact_category_agreement_fraction": float(selected_row["exact_category_agreement_fraction"]),
                    "near_category_agreement_fraction": float(selected_row["near_category_agreement_fraction"]),
                }
            )

    if not policy_case_records:
        return pd.DataFrame(), pd.DataFrame()

    policy_case_scores = pd.DataFrame.from_records(policy_case_records)
    policy_summary = (
        policy_case_scores.groupby(["forecast_lead_days", "policy_name"], dropna=False)
        .agg(
            case_count=("case_label", "count"),
            mean_score_mae=("score_mae", "mean"),
            mean_score_rmse=("score_rmse", "mean"),
            mean_abs_bias=("score_bias_mean", lambda values: float(np.mean(np.abs(values)))),
            mean_exact_category_agreement=("exact_category_agreement_fraction", "mean"),
            mean_near_category_agreement=("near_category_agreement_fraction", "mean"),
            aggregation_modes_used=("aggregation_mode", lambda values: ",".join(sorted(set(str(value) for value in values)))),
        )
        .reset_index()
        .sort_values(["forecast_lead_days", "mean_score_mae", "mean_score_rmse", "policy_name"])
    )

    baseline = policy_summary.loc[policy_summary["policy_name"] == "baseline", [
        "forecast_lead_days",
        "mean_score_mae",
        "mean_score_rmse",
        "mean_near_category_agreement",
    ]].rename(
        columns={
            "mean_score_mae": "baseline_mean_score_mae",
            "mean_score_rmse": "baseline_mean_score_rmse",
            "mean_near_category_agreement": "baseline_mean_near_category_agreement",
        }
    )
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
    return policy_case_scores, policy_summary


def _save_chart(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_candidate_mae_chart(candidate_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if candidate_summary.empty:
        return None
    pivot = candidate_summary.pivot(
        index="forecast_lead_days",
        columns="aggregation_mode",
        values="mean_score_mae",
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for mode in pivot.columns:
        ax.plot(pivot.index, pivot[mode], marker="o", linewidth=2.0, label=mode)
    ax.set_xlabel("Forecast lead day")
    ax.set_ylabel("Mean MAE")
    ax.set_title("Daily Aggregation Candidate MAE by Forecast Lead")
    ax.set_xticks(list(pivot.index))
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, ncols=2)
    return _save_chart(fig, output_dir / f"comfortwx_tune_daily_aggregation_{stem}_candidate_mae.png")


def _write_holdout_improvement_chart(holdout_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if holdout_summary.empty:
        return None
    grouped = (
        holdout_summary.groupby("forecast_lead_days", dropna=False)
        .agg(
            mean_score_mae_improvement=("score_mae_improvement", "mean"),
            mean_score_rmse_improvement=("score_rmse_improvement", "mean"),
        )
        .reset_index()
    )
    x = np.arange(len(grouped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, grouped["mean_score_mae_improvement"], width=width, label="MAE improvement", color="#2c7fb8")
    ax.bar(
        x + width / 2,
        grouped["mean_score_rmse_improvement"],
        width=width,
        label="RMSE improvement",
        color="#7fcdbb",
    )
    ax.axhline(0.0, color="#555555", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D+{int(value)}" for value in grouped["forecast_lead_days"]])
    ax.set_ylabel("Improvement vs baseline")
    ax.set_title("Held-Out Aggregation Selection Improvement by Lead")
    ax.grid(alpha=0.2, linewidth=0.5, axis="y")
    ax.legend(frameon=False)
    return _save_chart(fig, output_dir / f"comfortwx_tune_daily_aggregation_{stem}_holdout_improvement.png")


def _write_policy_comparison_chart(policy_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if policy_summary.empty:
        return None
    pivot = policy_summary.pivot(
        index="forecast_lead_days",
        columns="policy_name",
        values="mean_score_mae",
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for policy_name in pivot.columns:
        ax.plot(pivot.index, pivot[policy_name], marker="o", linewidth=2.0, label=policy_name)
    ax.set_xlabel("Forecast lead day")
    ax.set_ylabel("Mean MAE")
    ax.set_title("Experimental Aggregation Policy Comparison")
    ax.set_xticks(list(pivot.index))
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    return _save_chart(fig, output_dir / f"comfortwx_tune_daily_aggregation_{stem}_policy_comparison.png")


def _html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No results available.</p>"
    return frame.to_html(index=False, border=0, classes="summary-table")


def _write_tuning_report(
    *,
    candidate_summary: pd.DataFrame,
    recommended_modes: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    chart_paths: dict[str, Path],
    output_dir: Path,
    stem: str,
) -> Path:
    report_path = output_dir / f"comfortwx_tune_daily_aggregation_{stem}.html"
    chart_blocks: list[str] = []
    for label, path in chart_paths.items():
        chart_blocks.append(
            "\n".join(
                [
                    "<figure class='chart-card'>",
                    f"<img src='{html.escape(path.name)}' alt='{html.escape(label)}'>",
                    f"<figcaption>{html.escape(label)}</figcaption>",
                    "</figure>",
                ]
            )
        )
    report_path.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html lang='en'>",
                "<head>",
                "<meta charset='utf-8'>",
                "<title>Comfort Index Daily Aggregation Tuning</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 24px; color: #1c2330; background: #fafbfc; }",
                ".summary-table { border-collapse: collapse; width: 100%; background: white; }",
                ".summary-table th, .summary-table td { border: 1px solid #d9dee5; padding: 8px 10px; font-size: 0.92rem; }",
                ".summary-table th { background: #eef2f6; }",
                ".chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin: 20px 0; }",
                ".chart-card { background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 12px; }",
                ".chart-card img { width: 100%; height: auto; display: block; }",
                ".chart-card figcaption { margin-top: 8px; color: #46505a; font-size: 0.9rem; }",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Comfort Index Daily Aggregation Tuning</h1>",
                "<p>Verification-only held-out comparison of candidate daily aggregation modes. This does not change operational scoring on its own.</p>",
                "<h2>Recommended Mode by Lead</h2>",
                _html_table(recommended_modes),
                "<h2>Candidate Summary</h2>",
                _html_table(candidate_summary),
                "<h2>Held-Out Selection Summary</h2>",
                _html_table(holdout_summary),
                "<h2>Experimental Policy Comparison</h2>",
                _html_table(policy_summary),
                "<h2>Charts</h2>",
                "<div class='chart-grid'>",
                *chart_blocks,
                "</div>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune verification-only daily aggregation modes on held-out benchmark cases.")
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
        "--lead-days",
        default=",".join(str(value) for value in OPENMETEO_VERIFICATION_BENCHMARK_LEAD_DAYS),
        help="Comma-separated verification forecast lead days. Default: 1,2,3,7.",
    )
    parser.add_argument(
        "--candidate-modes",
        default=",".join(VERIFICATION_AGGREGATION_TUNING_CANDIDATE_MODES),
        help="Comma-separated aggregation modes to evaluate.",
    )
    parser.add_argument(
        "--case-cache-mode",
        default="reuse",
        choices=["reuse", "refresh"],
        help="Reuse cached scored-hourly verification cases when available. Default: reuse.",
    )
    parser.add_argument(
        "--regions",
        default=None,
        help="Optional comma-separated region filter for incremental tuning runs.",
    )
    parser.add_argument(
        "--dates",
        default=None,
        help="Optional comma-separated YYYY-MM-DD filter for incremental tuning runs.",
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


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_override = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    lead_days = _parse_lead_days(args.lead_days)
    candidate_modes = _parse_candidate_modes(args.candidate_modes)
    cases = _filter_cases(
        _resolved_cases(date_override, args.benchmark_tier, lead_days),
        region_filter=_parse_region_filter(args.regions),
        date_filter=_parse_date_filter(args.dates),
    )

    case_scores = evaluate_daily_aggregation_modes(
        cases=cases,
        output_dir=output_dir,
        mesh_profile=args.mesh_profile,
        forecast_model=args.forecast_model,
        forecast_model_mode=args.forecast_model_mode,
        analysis_model=args.analysis_model,
        forecast_run_hour_utc=args.forecast_run_hour_utc,
        candidate_modes=candidate_modes,
        case_cache_mode=args.case_cache_mode,
        benchmark_tier=args.benchmark_tier,
        max_fresh_cases=args.max_fresh_cases,
        case_cooldown_seconds=args.case_cooldown_seconds,
    )

    lead_stem = "d" + "-".join(str(day) for day in lead_days)
    tier_stem = args.benchmark_tier.replace("-", "_")
    stem = (
        f"{date_override.isoformat().replace('-', '')}_{tier_stem}_{lead_stem}"
        if date_override
        else f"{tier_stem}_cases_{lead_stem}"
    )
    case_scores_path = output_dir / f"comfortwx_tune_daily_aggregation_{stem}_case_scores.csv"
    case_scores.to_csv(case_scores_path, index=False)

    candidate_summary = summarize_candidate_modes(case_scores)
    candidate_summary_path = output_dir / f"comfortwx_tune_daily_aggregation_{stem}_candidate_summary.csv"
    candidate_summary.to_csv(candidate_summary_path, index=False)

    recommended_modes = recommend_modes_by_lead(candidate_summary)
    recommended_modes_path = output_dir / f"comfortwx_tune_daily_aggregation_{stem}_recommended_modes.csv"
    recommended_modes.to_csv(recommended_modes_path, index=False)

    holdout_summary = build_holdout_mode_selection(case_scores)
    holdout_summary_path = output_dir / f"comfortwx_tune_daily_aggregation_{stem}_holdout_selection.csv"
    holdout_summary.to_csv(holdout_summary_path, index=False)

    policy_case_scores, policy_summary = build_policy_comparison(case_scores)
    policy_case_scores_path = output_dir / f"comfortwx_tune_daily_aggregation_{stem}_policy_case_scores.csv"
    policy_case_scores.to_csv(policy_case_scores_path, index=False)
    policy_summary_path = output_dir / f"comfortwx_tune_daily_aggregation_{stem}_policy_summary.csv"
    policy_summary.to_csv(policy_summary_path, index=False)

    chart_paths: dict[str, Path] = {}
    candidate_mae_chart = _write_candidate_mae_chart(candidate_summary, output_dir=output_dir, stem=stem)
    if candidate_mae_chart is not None:
        chart_paths["Candidate mean MAE by lead"] = candidate_mae_chart
    holdout_chart = _write_holdout_improvement_chart(holdout_summary, output_dir=output_dir, stem=stem)
    if holdout_chart is not None:
        chart_paths["Held-out improvement by lead"] = holdout_chart
    policy_chart = _write_policy_comparison_chart(policy_summary, output_dir=output_dir, stem=stem)
    if policy_chart is not None:
        chart_paths["Experimental policy comparison"] = policy_chart

    report_path = _write_tuning_report(
        candidate_summary=candidate_summary,
        recommended_modes=recommended_modes,
        holdout_summary=holdout_summary,
        policy_summary=policy_summary,
        chart_paths=chart_paths,
        output_dir=output_dir,
        stem=stem,
    )

    print(f"Saved case scores: {case_scores_path}")
    print(f"Saved candidate summary: {candidate_summary_path}")
    print(f"Saved recommended modes: {recommended_modes_path}")
    print(f"Saved holdout selection summary: {holdout_summary_path}")
    print(f"Saved policy case scores: {policy_case_scores_path}")
    print(f"Saved policy summary: {policy_summary_path}")
    for label, path in chart_paths.items():
        print(f"Saved {label}: {path}")
    print(f"Saved tuning report: {report_path}")
    if not recommended_modes.empty:
        print(recommended_modes.to_string(index=False))


if __name__ == "__main__":
    main()
