"""Compare standard vs finer western mesh mosaics without changing score science."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import xarray as xr

from nicewx.config import OUTPUT_DIR
from nicewx.main import run_pipeline

WESTERN_SEAM_PAIR: tuple[str, str] = ("southwest", "rockies")
MESH_PROFILES: tuple[str, str] = ("standard", "fine")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a western mesh-sensitivity comparison for southwest + rockies.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for mesh-sensitivity outputs.")
    return parser.parse_args()


def compare_mosaic_mesh_fields(standard_daily: xr.Dataset, fine_daily: xr.Dataset) -> dict[str, float | int]:
    """Compare standard and fine mosaic daily score fields on the finer mesh grid."""

    target_lat = fine_daily["lat"]
    target_lon = fine_daily["lon"]
    standard_score = standard_daily["daily_score"].interp(lat=target_lat, lon=target_lon, method="linear")
    fine_score = fine_daily["daily_score"]
    valid_mask = standard_score.notnull() & fine_score.notnull()
    abs_change = abs(fine_score - standard_score).where(valid_mask)
    signed_change = (fine_score - standard_score).where(valid_mask)

    return {
        "compared_cell_count": int(valid_mask.sum().values),
        "mean_abs_daily_score_change": round(float(abs_change.mean(skipna=True).fillna(0.0).values), 2),
        "max_abs_daily_score_change": round(float(abs_change.max(skipna=True).fillna(0.0).values), 2),
        "mean_signed_daily_score_change": round(float(signed_change.mean(skipna=True).fillna(0.0).values), 2),
    }


def run_western_mesh_sensitivity(valid_date: date, output_dir: Path) -> tuple[Path, Path]:
    """Run standard and fine western seam mosaics and write comparison CSVs."""

    summary_records: list[dict[str, object]] = []
    daily_fields: dict[str, xr.Dataset] = {}

    for mesh_profile in MESH_PROFILES:
        outputs = run_pipeline(
            valid_date=valid_date,
            loader_name="openmeteo",
            lat_points=65,
            lon_points=115,
            output_dir=output_dir,
            mosaic_regions=list(WESTERN_SEAM_PAIR),
            mesh_profile=mesh_profile,
        )
        summary = pd.read_csv(outputs["mosaic_summary_csv"]).iloc[0].to_dict()
        summary["seam_pair"] = "+".join(WESTERN_SEAM_PAIR)
        summary["mesh_profile"] = mesh_profile
        summary_records.append(summary)
        daily_fields[mesh_profile] = xr.open_dataset(outputs["mosaic_daily_fields"]).load()

    profile_summary = pd.DataFrame(summary_records)
    ordered_summary_columns = [
        "seam_pair",
        "mesh_profile",
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
    profile_summary = profile_summary[[column for column in ordered_summary_columns if column in profile_summary.columns]]
    profile_summary_path = output_dir / f"nicewx_western_mesh_sensitivity_{valid_date:%Y%m%d}.csv"
    profile_summary.to_csv(profile_summary_path, index=False)

    field_comparison = compare_mosaic_mesh_fields(
        standard_daily=daily_fields["standard"],
        fine_daily=daily_fields["fine"],
    )
    for dataset in daily_fields.values():
        dataset.close()
    comparison_record = {
        "seam_pair": "+".join(WESTERN_SEAM_PAIR),
        "standard_target_grid_shape": profile_summary.loc[profile_summary["mesh_profile"] == "standard", "target_grid_shape"].iloc[0],
        "fine_target_grid_shape": profile_summary.loc[profile_summary["mesh_profile"] == "fine", "target_grid_shape"].iloc[0],
        **field_comparison,
    }
    comparison_path = output_dir / f"nicewx_western_mesh_sensitivity_diff_{valid_date:%Y%m%d}.csv"
    pd.DataFrame([comparison_record]).to_csv(comparison_path, index=False)
    return profile_summary_path, comparison_path


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    summary_path, comparison_path = run_western_mesh_sensitivity(valid_date=valid_date, output_dir=Path(args.output_dir))
    print(f"Valid date: {valid_date:%Y-%m-%d}")
    print(pd.read_csv(summary_path).to_string(index=False))
    print(pd.read_csv(comparison_path).to_string(index=False))
    print(f"Saved western mesh sensitivity summary: {summary_path}")
    print(f"Saved western mesh sensitivity diff: {comparison_path}")


if __name__ == "__main__":
    main()
