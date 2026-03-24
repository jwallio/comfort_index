"""Compare baseline and softer aggregation behavior for the western seam regime."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import xarray as xr

from comfortwx.config import OUTPUT_DIR
from comfortwx.main import run_pipeline
from comfortwx.scoring.categories import categorize_scores
from comfortwx.validation.western_seam_attribution import run_western_seam_attribution

WESTERN_SEAM_PAIR: tuple[str, str] = ("southwest", "rockies")
AGGREGATION_MODES: tuple[str, str] = ("baseline", "soft_reliability")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare western seam behavior under alternate aggregation modes.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for aggregation-sensitivity outputs.")
    parser.add_argument("--mesh-profile", default="standard", help="Open-Meteo mesh profile. Default: standard.")
    return parser.parse_args()


def compare_aggregation_mosaics(baseline_daily: xr.Dataset, tuned_daily: xr.Dataset) -> dict[str, float | int]:
    """Compare baseline and tuned western mosaics on the tuned overlap grid."""

    target_lat = tuned_daily["lat"]
    target_lon = tuned_daily["lon"]
    baseline_score = baseline_daily["daily_score"].interp(lat=target_lat, lon=target_lon, method="linear")
    tuned_score = tuned_daily["daily_score"]
    overlap_mask = tuned_daily["overlap_mask"].astype(bool)
    valid_mask = baseline_score.notnull() & tuned_score.notnull() & overlap_mask

    abs_change = abs(tuned_score - baseline_score).where(valid_mask)
    baseline_category = categorize_scores(baseline_score.fillna(0.0)).where(valid_mask)
    tuned_category = tuned_daily["category_index"].where(valid_mask)
    category_delta = abs(baseline_category - tuned_category).where(valid_mask)

    return {
        "overlap_compared_cell_count": int(valid_mask.sum().values),
        "overlap_mean_abs_daily_score_change": round(float(abs_change.mean(skipna=True).fillna(0.0).values), 3),
        "overlap_max_abs_daily_score_change": round(float(abs_change.max(skipna=True).fillna(0.0).values), 3),
        "overlap_category_agreement_fraction": round(float((category_delta == 0).mean(skipna=True).fillna(0.0).values), 4),
        "overlap_category_near_agreement_fraction": round(float((category_delta <= 1).mean(skipna=True).fillna(0.0).values), 4),
        "overlap_category_flip_count": int(((category_delta > 0) & valid_mask).sum().values),
    }


def run_western_aggregation_sensitivity(
    valid_date: date,
    output_dir: Path,
    mesh_profile: str = "standard",
) -> tuple[Path, Path]:
    """Run baseline and tuned aggregation comparisons for southwest + rockies."""

    summary_records: list[dict[str, object]] = []
    mosaic_daily_fields: dict[str, xr.Dataset] = {}

    for aggregation_mode in AGGREGATION_MODES:
        attribution_summary_path, _ = run_western_seam_attribution(
            valid_date=valid_date,
            output_dir=output_dir,
            mesh_profile=mesh_profile,
            aggregation_mode=aggregation_mode,
        )
        attribution_summary = pd.read_csv(attribution_summary_path).iloc[0].to_dict()
        attribution_summary["aggregation_mode"] = aggregation_mode
        summary_records.append(attribution_summary)

        mosaic_outputs = run_pipeline(
            valid_date=valid_date,
            loader_name="openmeteo",
            lat_points=65,
            lon_points=115,
            output_dir=output_dir,
            mosaic_regions=list(WESTERN_SEAM_PAIR),
            mesh_profile=mesh_profile,
            aggregation_mode=aggregation_mode,
        )
        mosaic_daily_fields[aggregation_mode] = xr.open_dataset(mosaic_outputs["mosaic_daily_fields"]).load()

    summary_frame = pd.DataFrame(summary_records)
    ordered_summary_columns = [
        "seam_pair",
        "mesh_profile",
        "aggregation_mode",
        "dominant_driver",
        "dominant_driver_fraction",
        "secondary_driver",
        "secondary_driver_fraction",
        "driver_group",
        "reliability_score_mean_abs_diff",
        "daily_score_mean_abs_diff",
        "pair_overlap_category_agreement_fraction",
        "pair_overlap_category_near_agreement_fraction",
        "near_threshold_fraction",
    ]
    summary_frame = summary_frame[[column for column in ordered_summary_columns if column in summary_frame.columns]]
    summary_path = output_dir / f"comfortwx_western_aggregation_sensitivity_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    summary_frame.to_csv(summary_path, index=False)

    comparison = compare_aggregation_mosaics(
        baseline_daily=mosaic_daily_fields["baseline"],
        tuned_daily=mosaic_daily_fields["soft_reliability"],
    )
    for dataset in mosaic_daily_fields.values():
        dataset.close()
    comparison_record = {
        "seam_pair": "+".join(WESTERN_SEAM_PAIR),
        "mesh_profile": mesh_profile,
        "baseline_aggregation_mode": "baseline",
        "tuned_aggregation_mode": "soft_reliability",
        "baseline_reliability_score_mean_abs_diff": float(
            summary_frame.loc[summary_frame["aggregation_mode"] == "baseline", "reliability_score_mean_abs_diff"].iloc[0]
        ),
        "tuned_reliability_score_mean_abs_diff": float(
            summary_frame.loc[summary_frame["aggregation_mode"] == "soft_reliability", "reliability_score_mean_abs_diff"].iloc[0]
        ),
        "baseline_daily_score_mean_abs_diff": float(
            summary_frame.loc[summary_frame["aggregation_mode"] == "baseline", "daily_score_mean_abs_diff"].iloc[0]
        ),
        "tuned_daily_score_mean_abs_diff": float(
            summary_frame.loc[summary_frame["aggregation_mode"] == "soft_reliability", "daily_score_mean_abs_diff"].iloc[0]
        ),
        "temp_score_remains_secondary_driver": bool(
            summary_frame.loc[summary_frame["aggregation_mode"] == "soft_reliability", "secondary_driver"].iloc[0] == "temp_score"
        ),
        **comparison,
    }
    comparison_path = output_dir / f"comfortwx_western_aggregation_sensitivity_diff_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    pd.DataFrame([comparison_record]).to_csv(comparison_path, index=False)
    return summary_path, comparison_path


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    summary_path, comparison_path = run_western_aggregation_sensitivity(
        valid_date=valid_date,
        output_dir=Path(args.output_dir),
        mesh_profile=args.mesh_profile,
    )
    print(f"Valid date: {valid_date:%Y-%m-%d}")
    print(pd.read_csv(summary_path).to_string(index=False))
    print(pd.read_csv(comparison_path).to_string(index=False))
    print(f"Saved western aggregation sensitivity summary: {summary_path}")
    print(f"Saved western aggregation sensitivity diff: {comparison_path}")


if __name__ == "__main__":
    main()
