"""Diagnose threshold brittleness between baseline and soft western aggregation modes."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from nicewx.config import CATEGORIES, OUTPUT_DIR, WESTERN_THRESHOLD_MARGIN_DIAGNOSTIC, WESTERN_THRESHOLD_PROXIMITY_BINS
from nicewx.validation.western_aggregation_sensitivity import run_western_aggregation_sensitivity
from nicewx.validation.western_seam_attribution import run_western_seam_attribution

WESTERN_THRESHOLDS: tuple[float, ...] = (45.0, 60.0, 75.0, 90.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run western threshold-sensitivity diagnostics for baseline vs soft_reliability.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for threshold-sensitivity outputs.")
    parser.add_argument("--mesh-profile", default="standard", help="Open-Meteo mesh profile. Default: standard.")
    return parser.parse_args()


def nearest_threshold_distance(score: float) -> float:
    """Return the smallest distance between a score and any category threshold."""

    return min(abs(score - threshold) for threshold in WESTERN_THRESHOLDS)


def crossed_thresholds(before_score: float, after_score: float) -> list[float]:
    """Return the ordered thresholds crossed between two scores."""

    return [
        threshold
        for threshold in WESTERN_THRESHOLDS
        if min(before_score, after_score) < threshold <= max(before_score, after_score)
    ]


def category_index_from_score(score: float) -> int:
    """Return the category index implied directly by the raw daily score."""

    bounded_score = max(0.0, min(100.0, float(score)))
    category_index = 0
    for index, category in enumerate(CATEGORIES):
        if bounded_score >= category.lower:
            category_index = index
    return category_index


def category_name_from_score(score: float) -> str:
    """Return the category name implied directly by the raw daily score."""

    return CATEGORIES[category_index_from_score(score)].name


def margin_stable_category_agreement(
    before_score: float,
    after_score: float,
    before_category_index: int,
    after_category_index: int,
    margin: float = WESTERN_THRESHOLD_MARGIN_DIAGNOSTIC,
) -> bool:
    """Return whether an adjacent category flip is near enough to a shared threshold to count as stable."""

    if before_category_index == after_category_index:
        return True
    if abs(before_category_index - after_category_index) > 1:
        return False
    shared_thresholds = crossed_thresholds(before_score, after_score)
    if len(shared_thresholds) != 1:
        return False
    threshold = shared_thresholds[0]
    return abs(before_score - threshold) <= margin and abs(after_score - threshold) <= margin


def run_western_threshold_sensitivity(
    valid_date: date,
    output_dir: Path,
    mesh_profile: str = "standard",
) -> tuple[Path, Path]:
    """Write threshold-proximity and category-flip diagnostics for the western seam."""

    merged, aggregation_summary = build_western_threshold_comparison_frame(
        valid_date=valid_date,
        output_dir=output_dir,
        mesh_profile=mesh_profile,
    )
    flip_detail = merged.loc[merged["category_flipped"]].copy()

    summary: dict[str, object] = {
        "seam_pair": "southwest+rockies",
        "mesh_profile": mesh_profile,
        "overlap_cell_count": int(len(merged)),
        "category_flip_count": int(flip_detail.shape[0]),
        "baseline_daily_score_mean_abs_diff": float(
            aggregation_summary.loc[aggregation_summary["aggregation_mode"] == "baseline", "daily_score_mean_abs_diff"].iloc[0]
        ),
        "tuned_daily_score_mean_abs_diff": float(
            aggregation_summary.loc[aggregation_summary["aggregation_mode"] == "soft_reliability", "daily_score_mean_abs_diff"].iloc[0]
        ),
        "baseline_reliability_score_mean_abs_diff": float(
            aggregation_summary.loc[aggregation_summary["aggregation_mode"] == "baseline", "reliability_score_mean_abs_diff"].iloc[0]
        ),
        "tuned_reliability_score_mean_abs_diff": float(
            aggregation_summary.loc[aggregation_summary["aggregation_mode"] == "soft_reliability", "reliability_score_mean_abs_diff"].iloc[0]
        ),
        "exact_category_agreement_fraction": round(float((merged["baseline_category_index"] == merged["tuned_category_index"]).mean()), 4),
        "near_category_agreement_fraction": round(
            float((merged["baseline_category_index"] - merged["tuned_category_index"]).abs().le(1).mean()),
            4,
        ),
        "margin_stable_agreement_fraction": round(float(merged["margin_stable_agreement"].mean()), 4),
        "margin_stable_flip_fraction": round(
            float(flip_detail["margin_stable_agreement"].mean()) if not flip_detail.empty else 0.0,
            4,
        ),
        "reliability_primary_flip_fraction": round(
            float(flip_detail["reliability_primary_flip"].mean()) if not flip_detail.empty else 0.0,
            4,
        ),
    }

    for distance in WESTERN_THRESHOLD_PROXIMITY_BINS:
        summary[f"baseline_within_{int(distance)}pt_fraction"] = round(float((merged["baseline_threshold_distance"] <= distance).mean()), 4)
        summary[f"tuned_within_{int(distance)}pt_fraction"] = round(float((merged["tuned_threshold_distance"] <= distance).mean()), 4)

    threshold_counts = {
        threshold: int(
            flip_detail["crossed_thresholds"].apply(lambda value, threshold=threshold: str(int(threshold)) in value.split(",")).sum()
        )
        for threshold in WESTERN_THRESHOLDS
    }
    for threshold, count in threshold_counts.items():
        summary[f"threshold_{int(threshold)}_flip_count"] = count
    summary["most_problematic_threshold"] = max(threshold_counts, key=threshold_counts.get) if threshold_counts else None

    transition_counts = flip_detail["category_transition"].value_counts()
    if not transition_counts.empty:
        summary["top_category_transition"] = transition_counts.index[0]
        summary["top_category_transition_count"] = int(transition_counts.iloc[0])
    else:
        summary["top_category_transition"] = ""
        summary["top_category_transition_count"] = 0

    summary_path = output_dir / f"nicewx_western_threshold_sensitivity_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    detail_path = output_dir / f"nicewx_western_threshold_flip_detail_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    flip_detail.to_csv(detail_path, index=False)
    return summary_path, detail_path


def build_western_threshold_comparison_frame(
    valid_date: date,
    output_dir: Path,
    mesh_profile: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the merged western overlap comparison frame used by threshold and low-end diagnostics."""

    aggregation_summary_path, _ = run_western_aggregation_sensitivity(
        valid_date=valid_date,
        output_dir=output_dir,
        mesh_profile=mesh_profile,
    )
    aggregation_summary = pd.read_csv(aggregation_summary_path)
    _, baseline_detail_path = run_western_seam_attribution(
        valid_date=valid_date,
        output_dir=output_dir,
        mesh_profile=mesh_profile,
        aggregation_mode="baseline",
    )
    _, tuned_detail_path = run_western_seam_attribution(
        valid_date=valid_date,
        output_dir=output_dir,
        mesh_profile=mesh_profile,
        aggregation_mode="soft_reliability",
    )
    baseline_detail = pd.read_csv(baseline_detail_path)
    tuned_detail = pd.read_csv(tuned_detail_path)

    merged = baseline_detail.merge(
        tuned_detail,
        on=["lat", "lon"],
        suffixes=("_baseline", "_tuned"),
    )
    merged["baseline_threshold_distance"] = merged["blended_daily_score_baseline"].apply(nearest_threshold_distance)
    merged["tuned_threshold_distance"] = merged["blended_daily_score_tuned"].apply(nearest_threshold_distance)
    merged["baseline_category_index"] = merged["blended_daily_score_baseline"].apply(category_index_from_score)
    merged["tuned_category_index"] = merged["blended_daily_score_tuned"].apply(category_index_from_score)
    merged["baseline_category"] = merged["blended_daily_score_baseline"].apply(category_name_from_score)
    merged["tuned_category"] = merged["blended_daily_score_tuned"].apply(category_name_from_score)
    merged["category_transition"] = merged["baseline_category"] + "->" + merged["tuned_category"]
    merged["crossed_thresholds"] = merged.apply(
        lambda row: ",".join(str(int(threshold)) for threshold in crossed_thresholds(row["blended_daily_score_baseline"], row["blended_daily_score_tuned"])),
        axis=1,
    )
    merged["threshold_crossed_count"] = merged["crossed_thresholds"].apply(lambda value: 0 if value == "" else len(value.split(",")))
    merged["category_flipped"] = merged["baseline_category_index"] != merged["tuned_category_index"]
    merged["reliability_change_magnitude"] = (
        (merged["southwest_reliability_score_baseline"] - merged["southwest_reliability_score_tuned"]).abs()
        + (merged["rockies_reliability_score_baseline"] - merged["rockies_reliability_score_tuned"]).abs()
    )
    merged["disruption_change_magnitude"] = (
        (merged["southwest_disruption_penalty_baseline"] - merged["southwest_disruption_penalty_tuned"]).abs()
        + (merged["rockies_disruption_penalty_baseline"] - merged["rockies_disruption_penalty_tuned"]).abs()
    )
    merged["reliability_primary_flip"] = merged["reliability_change_magnitude"] >= merged["disruption_change_magnitude"]
    merged["margin_stable_agreement"] = merged.apply(
        lambda row: margin_stable_category_agreement(
            before_score=row["blended_daily_score_baseline"],
            after_score=row["blended_daily_score_tuned"],
            before_category_index=int(row["baseline_category_index"]),
            after_category_index=int(row["tuned_category_index"]),
        ),
        axis=1,
    )
    return merged, aggregation_summary


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    summary_path, detail_path = run_western_threshold_sensitivity(
        valid_date=valid_date,
        output_dir=Path(args.output_dir),
        mesh_profile=args.mesh_profile,
    )
    summary = pd.read_csv(summary_path).iloc[0]
    print(f"Valid date: {valid_date:%Y-%m-%d}")
    print(f"Most problematic threshold: {summary['most_problematic_threshold']}")
    print(f"Category flips: {summary['category_flip_count']}")
    print(f"Margin-stable agreement fraction: {summary['margin_stable_agreement_fraction']:.4f}")
    print(f"Reliability-primary flip fraction: {summary['reliability_primary_flip_fraction']:.4f}")
    print(f"Saved threshold sensitivity summary: {summary_path}")
    print(f"Saved threshold flip detail: {detail_path}")


if __name__ == "__main__":
    main()
