"""Run the pilot adjacent-region seam comparisons and collect shared diagnostics."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from comfortwx.config import OUTPUT_DIR
from comfortwx.main import run_pipeline


DEFAULT_SEAM_PAIRS: tuple[tuple[str, str], ...] = (
    ("southeast", "northeast"),
    ("southwest", "rockies"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot adjacent-region seam comparisons.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for seam comparison outputs.")
    return parser.parse_args()


def run_seam_comparison(valid_date: date, output_dir: Path) -> Path:
    """Run the configured seam pairs and write a combined comparison CSV."""

    records: list[dict[str, object]] = []
    for region_a, region_b in DEFAULT_SEAM_PAIRS:
        outputs = run_pipeline(
            valid_date=valid_date,
            loader_name="openmeteo",
            lat_points=65,
            lon_points=115,
            output_dir=output_dir,
            mosaic_regions=[region_a, region_b],
        )
        summary_path = outputs["mosaic_summary_csv"]
        summary = pd.read_csv(summary_path).iloc[0].to_dict()
        summary["seam_pair"] = f"{region_a}+{region_b}"
        records.append(summary)

    comparison = pd.DataFrame(records)
    ordered_columns = [
        "seam_pair",
        "participating_regions",
        "target_grid_shape",
        "mean_daily_score",
        "min_daily_score",
        "max_daily_score",
        "covered_cell_count",
        "overlap_cell_count",
        "overlap_fraction_of_covered",
        "pair_overlap_cell_count",
        "pair_mean_abs_score_diff",
        "pair_max_abs_score_diff",
        "pair_overlap_mean_blended_score",
        "pair_overlap_category_agreement_fraction",
        "pair_overlap_category_near_agreement_fraction",
    ]
    comparison = comparison[[column for column in ordered_columns if column in comparison.columns]]
    output_path = output_dir / f"comfortwx_seam_comparison_{valid_date:%Y%m%d}.csv"
    comparison.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    output_path = run_seam_comparison(valid_date=valid_date, output_dir=Path(args.output_dir))
    comparison = pd.read_csv(output_path)
    print(f"Valid date: {valid_date:%Y-%m-%d}")
    print(comparison.to_string(index=False))
    print(f"Saved seam comparison: {output_path}")


if __name__ == "__main__":
    main()
