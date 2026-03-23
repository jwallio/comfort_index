"""Compare western seam diagnostics under alternate mosaic methods and target grids."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import xarray as xr

from nicewx.config import OUTPUT_DIR
from nicewx.main import run_pipeline
from nicewx.scoring.categories import categorize_scores

WESTERN_SEAM_PAIR: tuple[str, str] = ("southwest", "rockies")
BASELINE_SCENARIO: tuple[str, str] = ("taper", "adaptive")
METHOD_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("taper", "adaptive"),
    ("equal_overlap", "adaptive"),
    ("winner_take_all", "adaptive"),
    ("taper", "fixed_western"),
    ("equal_overlap", "fixed_western"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run western seam sensitivity under alternate mosaic methods.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for mosaic-method outputs.")
    parser.add_argument("--mesh-profile", default="standard", help="Open-Meteo mesh profile to use. Default: standard.")
    return parser.parse_args()


def compare_mosaic_methods(
    baseline_daily: xr.Dataset,
    candidate_daily: xr.Dataset,
) -> dict[str, float | int]:
    """Compare a candidate mosaic against the baseline on the candidate grid."""

    target_lat = candidate_daily["lat"]
    target_lon = candidate_daily["lon"]
    baseline_score = baseline_daily["daily_score"].interp(lat=target_lat, lon=target_lon, method="linear")
    candidate_score = candidate_daily["daily_score"]
    overlap_mask = candidate_daily["overlap_mask"].astype(bool)
    valid_mask = baseline_score.notnull() & candidate_score.notnull()
    overlap_valid = overlap_mask & valid_mask

    abs_change = abs(candidate_score - baseline_score).where(valid_mask)
    baseline_category = categorize_scores(baseline_score.fillna(0.0)).where(valid_mask)
    candidate_category = candidate_daily["category_index"].where(valid_mask)
    category_flip_mask = (baseline_category != candidate_category).where(overlap_valid, False)

    return {
        "compared_cell_count": int(valid_mask.sum().values),
        "mean_abs_daily_score_change": round(float(abs_change.mean(skipna=True).fillna(0.0).values), 2),
        "max_abs_daily_score_change": round(float(abs_change.max(skipna=True).fillna(0.0).values), 2),
        "overlap_category_flip_count": int(category_flip_mask.sum().values),
    }


def run_western_mosaic_method_sensitivity(
    valid_date: date,
    output_dir: Path,
    mesh_profile: str = "standard",
) -> tuple[Path, Path]:
    """Run the western seam pair across diagnostic mosaic methods and write comparison CSVs."""

    summary_records: list[dict[str, object]] = []
    daily_fields: dict[tuple[str, str], xr.Dataset] = {}

    for blend_method, target_grid_policy in METHOD_SCENARIOS:
        outputs = run_pipeline(
            valid_date=valid_date,
            loader_name="openmeteo",
            lat_points=65,
            lon_points=115,
            output_dir=output_dir,
            mosaic_regions=list(WESTERN_SEAM_PAIR),
            mesh_profile=mesh_profile,
            mosaic_blend_method=blend_method,
            mosaic_target_grid=target_grid_policy,
        )
        summary = pd.read_csv(outputs["mosaic_summary_csv"]).iloc[0].to_dict()
        summary["seam_pair"] = "+".join(WESTERN_SEAM_PAIR)
        summary["method_id"] = f"{blend_method}+{target_grid_policy}"
        summary["mesh_profile"] = mesh_profile
        summary_records.append(summary)
        daily_fields[(blend_method, target_grid_policy)] = xr.open_dataset(outputs["mosaic_daily_fields"]).load()

    summary_frame = pd.DataFrame(summary_records)
    ordered_summary_columns = [
        "seam_pair",
        "mesh_profile",
        "method_id",
        "blend_method",
        "target_grid_policy",
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
        "pair_score_diff_p90",
        "pair_overlap_mean_blended_score",
        "pair_overlap_blended_score_p10",
        "pair_overlap_blended_score_p90",
        "pair_overlap_category_agreement_fraction",
        "pair_overlap_category_near_agreement_fraction",
        "pair_overlap_near_threshold_cell_count",
    ]
    summary_frame = summary_frame[[column for column in ordered_summary_columns if column in summary_frame.columns]]
    summary_path = output_dir / f"nicewx_western_mosaic_method_sensitivity_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    summary_frame.to_csv(summary_path, index=False)

    baseline_daily = daily_fields[BASELINE_SCENARIO]
    comparison_records: list[dict[str, object]] = []
    for scenario in METHOD_SCENARIOS:
        blend_method, target_grid_policy = scenario
        candidate_daily = daily_fields[scenario]
        comparison = compare_mosaic_methods(baseline_daily=baseline_daily, candidate_daily=candidate_daily)
        comparison_records.append(
            {
                "seam_pair": "+".join(WESTERN_SEAM_PAIR),
                "mesh_profile": mesh_profile,
                "baseline_method_id": f"{BASELINE_SCENARIO[0]}+{BASELINE_SCENARIO[1]}",
                "candidate_method_id": f"{blend_method}+{target_grid_policy}",
                "blend_method": blend_method,
                "target_grid_policy": target_grid_policy,
                **comparison,
            }
        )

    for dataset in daily_fields.values():
        dataset.close()

    comparison_path = output_dir / f"nicewx_western_mosaic_method_diff_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    pd.DataFrame(comparison_records).to_csv(comparison_path, index=False)
    return summary_path, comparison_path


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    summary_path, comparison_path = run_western_mosaic_method_sensitivity(
        valid_date=valid_date,
        output_dir=Path(args.output_dir),
        mesh_profile=args.mesh_profile,
    )
    print(f"Valid date: {valid_date:%Y-%m-%d}")
    print(pd.read_csv(summary_path).to_string(index=False))
    print(pd.read_csv(comparison_path).to_string(index=False))
    print(f"Saved western mosaic-method summary: {summary_path}")
    print(f"Saved western mosaic-method diff: {comparison_path}")


if __name__ == "__main__":
    main()
