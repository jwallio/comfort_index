"""Benchmark harness for proxy forecast-vs-analysis verification."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from nicewx.config import (
    OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS,
    OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC,
    OUTPUT_DIR,
)
from nicewx.validation.verify_benchmark_cases import (
    DEFAULT_VERIFICATION_BENCHMARK_CASES,
    VerificationBenchmarkCase,
)
from nicewx.validation.verify_model import run_verification


def _resolved_cases(date_override: date | None) -> list[VerificationBenchmarkCase]:
    if date_override is None:
        return list(DEFAULT_VERIFICATION_BENCHMARK_CASES)
    regions_in_order: list[str] = []
    for case in DEFAULT_VERIFICATION_BENCHMARK_CASES:
        if case.region_name not in regions_in_order:
            regions_in_order.append(case.region_name)
    return [VerificationBenchmarkCase(region_name=region_name, valid_date=date_override) for region_name in regions_in_order]


def format_benchmark_table(summary: pd.DataFrame) -> str:
    columns = [
        "region",
        "date",
        "status",
        "score_bias_mean",
        "score_mae",
        "score_rmse",
        "exact_category_agreement_fraction",
        "near_category_agreement_fraction",
    ]
    present_columns = [column for column in columns if column in summary.columns]
    return summary.loc[:, present_columns].sort_values(["score_mae", "score_rmse", "region", "date"]).to_string(index=False)


def run_verification_benchmark(
    *,
    cases: list[VerificationBenchmarkCase],
    output_dir: Path,
    mesh_profile: str,
    forecast_model: str,
    forecast_run_hour_utc: int,
    forecast_lead_days: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for case in cases:
        try:
            outputs = run_verification(
                valid_date=case.valid_date,
                region_name=case.region_name,
                output_dir=output_dir,
                mesh_profile=mesh_profile,
                forecast_model=forecast_model,
                forecast_run_hour_utc=forecast_run_hour_utc,
                forecast_lead_days=forecast_lead_days,
                workflow_name="verification_benchmark",
            )
            summary_record = dict(outputs["summary_record"])
            summary_record.update(
                {
                    "region": case.region_name,
                    "date": case.valid_date.isoformat(),
                    "status": "ok",
                    "forecast_score_map_path": str(outputs["forecast_score_map"]),
                    "analysis_score_map_path": str(outputs["analysis_score_map"]),
                    "score_difference_map_path": str(outputs["score_difference_map"]),
                    "summary_csv_path": str(outputs["summary_csv"]),
                    "point_metrics_csv_path": str(outputs["point_metrics_csv"]),
                    "request_summary_csv_path": str(outputs["request_summary_csv"]),
                    "request_detail_csv_path": str(outputs["request_detail_csv"]),
                }
            )
            records.append(summary_record)
        except Exception as exc:
            records.append(
                {
                    "region": case.region_name,
                    "date": case.valid_date.isoformat(),
                    "forecast_model": forecast_model,
                    "score_bias_mean": "",
                    "score_mae": "",
                    "score_rmse": "",
                    "exact_category_agreement_fraction": "",
                    "near_category_agreement_fraction": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
    return pd.DataFrame.from_records(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small benchmark suite for proxy Comfort Index verification.")
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD override for all benchmark regions.")
    parser.add_argument("--mesh-profile", default="standard", help="Regional mesh profile. Default: standard.")
    parser.add_argument("--forecast-model", default=OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT)
    parser.add_argument("--forecast-run-hour-utc", type=int, default=OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC)
    parser.add_argument("--forecast-lead-days", type=int, default=OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    date_override = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    cases = _resolved_cases(date_override)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_verification_benchmark(
        cases=cases,
        output_dir=output_dir,
        mesh_profile=args.mesh_profile,
        forecast_model=args.forecast_model,
        forecast_run_hour_utc=args.forecast_run_hour_utc,
        forecast_lead_days=args.forecast_lead_days,
    )

    stem = date_override.isoformat().replace("-", "") if date_override else "default_cases"
    summary_path = output_dir / f"nicewx_verify_benchmark_{stem}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved verification benchmark summary: {summary_path}")
    print(format_benchmark_table(summary))


if __name__ == "__main__":
    main()
