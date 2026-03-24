"""Lightweight calibration helper for demo cases."""

from __future__ import annotations

import argparse
from datetime import date, datetime

from comfortwx.validation.demo_cases import run_demo_case_validation


def format_demo_calibration_table(valid_date: date) -> str:
    """Return a concise printable calibration table for the demo cases."""

    summary, _ = run_demo_case_validation(valid_date)
    display_columns = [
        "case_name",
        "best_3hr",
        "best_6hr",
        "daytime_weighted_mean",
        "reliability_score",
        "disruption_penalty",
        "daily_score",
        "category",
    ]
    return summary.loc[:, display_columns].to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the comfort demo calibration cases.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    args = parser.parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    print(format_demo_calibration_table(valid_date))


if __name__ == "__main__":
    main()
