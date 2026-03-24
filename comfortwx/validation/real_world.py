"""Real-world point forecast validation harness."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from comfortwx.config import OUTPUT_DIR
from comfortwx.data.loaders import get_loader
from comfortwx.scoring.daily import aggregate_daily_scores
from comfortwx.scoring.hourly import score_hourly_dataset
from comfortwx.validation.inspection import inspect_point
from comfortwx.validation.mismatch_audit import audit_point_mismatch
from comfortwx.validation.real_world_cases import DEFAULT_REAL_WORLD_CASES, RealWorldCase


LABEL_ORDER = {
    "poor": 0,
    "fair": 1,
    "pleasant": 2,
    "ideal": 3,
    "exceptional": 4,
    # Backward-compatible aliases for historical case lists and outputs.
    "beautiful": 2,
    "perfect": 3,
    "pristine": 4,
}


def compare_expected_label(expected_label: str | None, actual_label: str) -> str:
    """Return a simple expected-vs-actual comparison label."""

    if not expected_label:
        return ""
    expected_key = expected_label.strip().lower()
    actual_key = actual_label.strip().lower()
    if expected_key not in LABEL_ORDER or actual_key not in LABEL_ORDER:
        return "mismatch"
    distance = abs(LABEL_ORDER[expected_key] - LABEL_ORDER[actual_key])
    if distance == 0:
        return "match"
    if distance == 1:
        return "near match"
    return "mismatch"


def _resolved_cases(date_override: date | None) -> list[RealWorldCase]:
    if date_override is None:
        return list(DEFAULT_REAL_WORLD_CASES)
    return [
        RealWorldCase(
            case_name=case.case_name,
            lat=case.lat,
            lon=case.lon,
            date=date_override,
            expected_label=case.expected_label,
        )
        for case in DEFAULT_REAL_WORLD_CASES
    ]


def run_real_world_validation(cases: list[RealWorldCase]) -> pd.DataFrame:
    """Fetch, score, and summarize a list of real-world point cases."""

    records: list[dict[str, object]] = []
    for case in cases:
        try:
            loader = get_loader(loader_name="openmeteo", lat=case.lat, lon=case.lon)
            hourly = loader.load_hourly_grid(valid_date=case.date)
            scored = score_hourly_dataset(hourly)
            daily = aggregate_daily_scores(scored)
            _, summary_frame, explanation, (resolved_lat, resolved_lon) = inspect_point(
                scored_hourly=scored,
                daily=daily,
                lat=case.lat,
                lon=case.lon,
            )
            summary = summary_frame.iloc[0].to_dict()
            audit = audit_point_mismatch(scored_point=scored.isel(lat=0, lon=0), daily_point=daily.isel(lat=0, lon=0))
            actual_label = str(summary["category"]).lower()
            records.append(
                {
                    "case_name": case.case_name,
                    "lat": round(resolved_lat, 2),
                    "lon": round(resolved_lon, 2),
                    "date": case.date.isoformat(),
                    "daily_score": summary["daily_score"],
                    "category": summary["category"],
                    "best_3hr": summary["best_3hr"],
                    "best_6hr": summary["best_6hr"],
                    "daytime_weighted_mean": summary["daytime_weighted_mean"],
                    "reliability_score": summary["reliability_score"],
                    "disruption_penalty": summary["disruption_penalty"],
                    "explanation": explanation,
                    "expected_label": case.expected_label or "",
                    "actual_label": actual_label,
                    "comparison": compare_expected_label(case.expected_label, actual_label),
                    **audit,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "case_name": case.case_name,
                    "lat": case.lat,
                    "lon": case.lon,
                    "date": case.date.isoformat(),
                    "daily_score": "",
                    "category": "",
                    "best_3hr": "",
                    "best_6hr": "",
                    "daytime_weighted_mean": "",
                    "reliability_score": "",
                    "disruption_penalty": "",
                    "explanation": "",
                    "expected_label": case.expected_label or "",
                    "actual_label": "",
                    "comparison": "error",
                    "dominant_limiting_factor": "",
                    "top_reason_1": "",
                    "top_reason_2": "",
                    "top_reason_3": "",
                    "top_3_reasons": "",
                    "error": str(exc),
                }
            )
    return pd.DataFrame.from_records(records)


def format_real_world_validation_table(summary: pd.DataFrame) -> str:
    """Return a compact console table for the real-world validation run."""

    display_columns = [
        "case_name",
        "date",
        "daily_score",
        "category",
        "best_6hr",
        "reliability_score",
        "disruption_penalty",
        "expected_label",
        "comparison",
    ]
    return summary.loc[:, [column for column in display_columns if column in summary.columns]].to_string(index=False)


def format_mismatch_audit_table(summary: pd.DataFrame) -> str:
    """Return a compact mismatch-only console table."""

    mismatch_rows = summary[summary["comparison"].isin(["near match", "mismatch"])]
    if mismatch_rows.empty:
        return "No near matches or mismatches."
    display_columns = [
        "case_name",
        "expected_label",
        "actual_label",
        "dominant_limiting_factor",
        "top_3_reasons",
    ]
    return mismatch_rows.loc[:, display_columns].to_string(index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-world Open-Meteo point validation cases.")
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD override for all validation cases.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for validation CSV output.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    date_override = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    cases = _resolved_cases(date_override)
    summary = run_real_world_validation(cases)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = date_override.isoformat().replace("-", "") if date_override else "mixed_dates"
    output_path = output_dir / f"comfortwx_real_world_validation_{output_stem}.csv"
    mismatch_output_path = output_dir / f"comfortwx_real_world_validation_mismatches_{output_stem}.csv"
    summary.to_csv(output_path, index=False)
    summary[summary["comparison"].isin(["near match", "mismatch"])].to_csv(mismatch_output_path, index=False)

    print(f"Saved validation summary: {output_path}")
    print(f"Saved mismatch audit: {mismatch_output_path}")
    print(format_real_world_validation_table(summary))
    print("")
    print("Mismatch audit:")
    print(format_mismatch_audit_table(summary))


if __name__ == "__main__":
    main()
