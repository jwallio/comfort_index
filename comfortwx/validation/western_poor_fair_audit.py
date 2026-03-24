"""Diagnose low-end Poor/Fair threshold brittleness in the western seam."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from comfortwx.config import (
    CATEGORIES,
    OUTPUT_DIR,
    WESTERN_POOR_FAIR_BORDERLINE_MARGIN,
    WESTERN_POOR_FAIR_DIAGNOSTIC_THRESHOLDS,
    WESTERN_POOR_FAIR_DISTRIBUTION_RANGE,
    WESTERN_POOR_FAIR_FOCUS_WINDOWS,
)
from comfortwx.validation.western_threshold_sensitivity import build_western_threshold_comparison_frame

LOW_END_THRESHOLD_LABELS: dict[float, str] = {
    43.0: "shifted_43",
    45.0: "production_45",
    47.0: "shifted_47",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run western Poor/Fair threshold diagnostics for baseline vs soft_reliability.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for Poor/Fair audit outputs.")
    parser.add_argument("--mesh-profile", default="standard", help="Open-Meteo mesh profile. Default: standard.")
    return parser.parse_args()


def category_index_with_poor_fair_threshold(score: float, poor_fair_threshold: float) -> int:
    """Return a category index with a diagnostic Poor/Fair boundary while preserving higher thresholds."""

    bounded_score = max(0.0, min(100.0, float(score)))
    if bounded_score >= CATEGORIES[4].lower:
        return 4
    if bounded_score >= CATEGORIES[3].lower:
        return 3
    if bounded_score >= CATEGORIES[2].lower:
        return 2
    if bounded_score >= poor_fair_threshold:
        return 1
    return 0


def borderline_poor_fair_agreement(
    before_score: float,
    after_score: float,
    before_index: int,
    after_index: int,
    threshold: float = 45.0,
    margin: float = WESTERN_POOR_FAIR_BORDERLINE_MARGIN,
) -> bool:
    """Treat Poor/Fair splits near the low-end threshold as comparison-only borderline agreements."""

    if before_index == after_index:
        return True
    if {before_index, after_index} != {0, 1}:
        return False
    return abs(before_score - threshold) <= margin and abs(after_score - threshold) <= margin


def summarize_poor_fair_audit(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize low-end overlap behavior for the western seam."""

    score_min, score_max = WESTERN_POOR_FAIR_DISTRIBUTION_RANGE
    distribution_frame = frame.loc[
        frame["blended_daily_score_baseline"].between(score_min, score_max)
        | frame["blended_daily_score_tuned"].between(score_min, score_max)
    ].copy()
    poor_fair_flips = frame.loc[frame["production_transition"].isin(["Poor->Fair", "Fair->Poor"])].copy()

    summary: dict[str, object] = {
        "seam_pair": "southwest+rockies",
        "overlap_cell_count": int(len(frame)),
        "distribution_cell_count_35_55": int(len(distribution_frame)),
        "poor_fair_flip_count": int(len(poor_fair_flips)),
    }

    distribution_bins = (
        (35.0, 40.0),
        (40.0, 45.0),
        (45.0, 50.0),
        (50.0, 55.0),
    )
    score_instances = pd.concat(
        [
            frame["blended_daily_score_baseline"].rename("score"),
            frame["blended_daily_score_tuned"].rename("score"),
        ],
        ignore_index=True,
    )
    for lower, upper in distribution_bins:
        label = f"{int(lower)}_{int(upper)}"
        summary[f"score_instance_count_{label}"] = int(score_instances.between(lower, upper, inclusive="left").sum())

    for lower, upper in WESTERN_POOR_FAIR_FOCUS_WINDOWS:
        label = f"{int(lower)}_{int(upper)}"
        in_band_any = frame["blended_daily_score_baseline"].between(lower, upper) | frame["blended_daily_score_tuned"].between(lower, upper)
        in_band_both = frame["blended_daily_score_baseline"].between(lower, upper) & frame["blended_daily_score_tuned"].between(lower, upper)
        summary[f"cells_in_{label}_any_fraction"] = round(float(in_band_any.mean()), 4)
        summary[f"cells_in_{label}_both_fraction"] = round(float(in_band_both.mean()), 4)
        if not poor_fair_flips.empty:
            flip_any = poor_fair_flips["blended_daily_score_baseline"].between(lower, upper) | poor_fair_flips["blended_daily_score_tuned"].between(lower, upper)
            flip_both = poor_fair_flips["blended_daily_score_baseline"].between(lower, upper) & poor_fair_flips["blended_daily_score_tuned"].between(lower, upper)
            summary[f"poor_fair_flips_in_{label}_any_fraction"] = round(float(flip_any.mean()), 4)
            summary[f"poor_fair_flips_in_{label}_both_fraction"] = round(float(flip_both.mean()), 4)
        else:
            summary[f"poor_fair_flips_in_{label}_any_fraction"] = 0.0
            summary[f"poor_fair_flips_in_{label}_both_fraction"] = 0.0

    return summary


def interpret_poor_fair_results(summary: dict[str, object], alternatives: pd.DataFrame) -> tuple[str, str]:
    """Return a compact interpretation label and recommendation."""

    poor_fair_flip_fraction_42_48 = float(summary["poor_fair_flips_in_42_48_any_fraction"])
    production_row = alternatives.loc[alternatives["mode"] == "production_45"].iloc[0]
    shifted_best = alternatives.loc[alternatives["mode"] != "production_45"].sort_values("poor_fair_flip_count").iloc[0]

    if poor_fair_flip_fraction_42_48 >= 0.65 and float(shifted_best["poor_fair_flip_reduction_fraction"]) < 0.25:
        return (
            "mostly unavoidable due to true marginal raw scores",
            "A presentation-layer borderline treatment is more justified than another scoring change.",
        )
    if float(shifted_best["poor_fair_flip_reduction_fraction"]) >= 0.35 and float(shifted_best["total_distribution_shift"]) <= 0.06:
        return (
            "mostly threshold-definition driven",
            "A category-definition adjustment is more justified than more aggregation tuning.",
        )
    if float(production_row["poor_fair_flip_count"]) > 0 and poor_fair_flip_fraction_42_48 < 0.5:
        return (
            "still suggestive of upstream aggregation issues",
            "The Poor/Fair flips are not concentrated tightly enough around 45 to prefer a presentation-only fix.",
        )
    return (
        "mostly unavoidable due to true marginal raw scores",
        "A presentation-layer borderline treatment looks more justified than another scoring change.",
    )


def run_western_poor_fair_audit(
    valid_date: date,
    output_dir: Path,
    mesh_profile: str = "standard",
) -> tuple[Path, Path, Path]:
    """Write Poor/Fair boundary diagnostics and comparison-only alternatives for the western seam."""

    comparison_frame, _ = build_western_threshold_comparison_frame(
        valid_date=valid_date,
        output_dir=output_dir,
        mesh_profile=mesh_profile,
    )
    frame = comparison_frame.copy()

    for threshold in WESTERN_POOR_FAIR_DIAGNOSTIC_THRESHOLDS:
        label = LOW_END_THRESHOLD_LABELS.get(threshold, f"threshold_{threshold:.1f}")
        frame[f"baseline_index_{label}"] = frame["blended_daily_score_baseline"].apply(
            lambda score, threshold=threshold: category_index_with_poor_fair_threshold(score, threshold)
        )
        frame[f"tuned_index_{label}"] = frame["blended_daily_score_tuned"].apply(
            lambda score, threshold=threshold: category_index_with_poor_fair_threshold(score, threshold)
        )
        frame[f"transition_{label}"] = frame.apply(
            lambda row, label=label: f"{CATEGORIES[int(row[f'baseline_index_{label}'])].name}->{CATEGORIES[int(row[f'tuned_index_{label}'])].name}",
            axis=1,
        )

    frame["production_transition"] = frame["transition_production_45"]
    frame["borderline_same_tier"] = frame.apply(
        lambda row: borderline_poor_fair_agreement(
            before_score=row["blended_daily_score_baseline"],
            after_score=row["blended_daily_score_tuned"],
            before_index=int(row["baseline_index_production_45"]),
            after_index=int(row["tuned_index_production_45"]),
        ),
        axis=1,
    )

    summary = summarize_poor_fair_audit(frame)

    category_distribution_scores = pd.concat(
        [
            frame["blended_daily_score_baseline"].rename("score"),
            frame["blended_daily_score_tuned"].rename("score"),
        ],
        ignore_index=True,
    )

    alternative_rows: list[dict[str, object]] = []
    production_distribution: dict[int, float] | None = None
    production_poor_fair_flip_count = 0
    for threshold in WESTERN_POOR_FAIR_DIAGNOSTIC_THRESHOLDS:
        label = LOW_END_THRESHOLD_LABELS.get(threshold, f"threshold_{threshold:.1f}")
        baseline_col = f"baseline_index_{label}"
        tuned_col = f"tuned_index_{label}"
        transition_col = f"transition_{label}"

        category_diffs = (frame[baseline_col] - frame[tuned_col]).abs()
        poor_fair_flips = frame[transition_col].isin(["Poor->Fair", "Fair->Poor"])

        classified_scores = category_distribution_scores.apply(lambda score, threshold=threshold: category_index_with_poor_fair_threshold(score, threshold))
        distribution = {
            index: float((classified_scores == index).mean()) for index in range(len(CATEGORIES))
        }
        if label == "production_45":
            production_distribution = distribution
            production_poor_fair_flip_count = int(poor_fair_flips.sum())

        alternative_rows.append(
            {
                "mode": label,
                "poor_fair_threshold": threshold,
                "poor_fair_flip_count": int(poor_fair_flips.sum()),
                "exact_category_agreement_fraction": round(float((frame[baseline_col] == frame[tuned_col]).mean()), 4),
                "near_category_agreement_fraction": round(float(category_diffs.le(1).mean()), 4),
                "borderline_same_tier_fraction": round(
                    float(
                        (
                            (frame[baseline_col] == frame[tuned_col])
                            | frame.apply(
                                lambda row, threshold=threshold: borderline_poor_fair_agreement(
                                    before_score=row["blended_daily_score_baseline"],
                                    after_score=row["blended_daily_score_tuned"],
                                    before_index=int(row[baseline_col]),
                                    after_index=int(row[tuned_col]),
                                    threshold=threshold,
                                ),
                                axis=1,
                            )
                        ).mean()
                    ),
                    4,
                ),
                "poor_fraction": round(distribution[0], 4),
                "fair_fraction": round(distribution[1], 4),
                "pleasant_fraction": round(distribution[2], 4),
                "ideal_fraction": round(distribution[3], 4),
                "exceptional_fraction": round(distribution[4], 4),
            }
        )

    alternatives = pd.DataFrame(alternative_rows)
    if production_distribution is None:
        raise RuntimeError("Production low-end distribution was not computed.")

    alternatives["poor_fair_flip_reduction_fraction"] = alternatives["poor_fair_flip_count"].apply(
        lambda count: round(
            (production_poor_fair_flip_count - int(count)) / production_poor_fair_flip_count,
            4,
        )
        if production_poor_fair_flip_count > 0
        else 0.0
    )
    alternatives["total_distribution_shift"] = alternatives.apply(
        lambda row: round(
            0.5
            * sum(
                abs(float(row[f"{CATEGORIES[index].name.lower()}_fraction"]) - production_distribution[index])
                for index in range(len(CATEGORIES))
            ),
            4,
        ),
        axis=1,
    )

    interpretation, recommendation = interpret_poor_fair_results(summary, alternatives)
    summary["interpretation"] = interpretation
    summary["recommendation"] = recommendation
    summary["borderline_same_tier_fraction_production"] = float(
        alternatives.loc[alternatives["mode"] == "production_45", "borderline_same_tier_fraction"].iloc[0]
    )

    poor_fair_detail = frame.loc[frame["production_transition"].isin(["Poor->Fair", "Fair->Poor"])].copy()

    summary_path = output_dir / f"comfortwx_western_poor_fair_audit_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    alternatives_path = output_dir / f"comfortwx_western_poor_fair_alternatives_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    detail_path = output_dir / f"comfortwx_western_poor_fair_detail_{mesh_profile}_{valid_date:%Y%m%d}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    alternatives.to_csv(alternatives_path, index=False)
    poor_fair_detail.to_csv(detail_path, index=False)
    return summary_path, alternatives_path, detail_path


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    summary_path, alternatives_path, detail_path = run_western_poor_fair_audit(
        valid_date=valid_date,
        output_dir=Path(args.output_dir),
        mesh_profile=args.mesh_profile,
    )
    summary = pd.read_csv(summary_path).iloc[0]
    alternatives = pd.read_csv(alternatives_path)
    best_alternative = alternatives.loc[alternatives["mode"] != "production_45"].sort_values("poor_fair_flip_count").iloc[0]
    print(f"Valid date: {valid_date:%Y-%m-%d}")
    print(f"Interpretation: {summary['interpretation']}")
    print(f"Poor/Fair flips: {summary['poor_fair_flip_count']}")
    print(f"Most useful comparison-only alternative: {best_alternative['mode']}")
    print(f"Best alternative Poor/Fair flip reduction: {best_alternative['poor_fair_flip_reduction_fraction']:.4f}")
    print(f"Saved Poor/Fair audit summary: {summary_path}")
    print(f"Saved Poor/Fair alternatives: {alternatives_path}")
    print(f"Saved Poor/Fair detail: {detail_path}")


if __name__ == "__main__":
    main()
