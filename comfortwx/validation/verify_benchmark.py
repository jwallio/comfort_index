"""Benchmark harness for proxy forecast-vs-analysis verification."""

from __future__ import annotations

import argparse
import html
import shutil
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comfortwx.config import (
    OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS,
    OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT,
    OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC,
    OUTPUT_DIR,
    VERIFICATION_BENCHMARK_THRESHOLDS,
)
from comfortwx.validation.verify_benchmark_cases import (
    DEFAULT_VERIFICATION_BENCHMARK_CASES,
    VerificationBenchmarkCase,
)
from comfortwx.validation.verify_model import run_verification


def _resolved_cases(date_override: date | None) -> list[VerificationBenchmarkCase]:
    if date_override is None:
        return list(DEFAULT_VERIFICATION_BENCHMARK_CASES)
    regions_in_order: list[str] = []
    for case in DEFAULT_VERIFICATION_BENCHMARK_CASES:
        if case.region_name not in regions_in_order:
            regions_in_order.append(case.region_name)
    return [VerificationBenchmarkCase(region_name=region_name, valid_date=date_override) for region_name in regions_in_order]


def _ok_cases(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "status" not in summary.columns:
        return summary.iloc[0:0].copy()
    ok = summary.loc[summary["status"] == "ok"].copy()
    if ok.empty:
        return ok
    ok["date"] = pd.to_datetime(ok["date"])
    ok["case_label"] = ok.apply(lambda row: f"{row['region']} {row['date']:%Y-%m-%d}", axis=1)
    ok["score_bias_mean"] = ok["score_bias_mean"].astype(float)
    ok["score_mae"] = ok["score_mae"].astype(float)
    ok["score_rmse"] = ok["score_rmse"].astype(float)
    ok["exact_category_agreement_fraction"] = ok["exact_category_agreement_fraction"].astype(float)
    ok["near_category_agreement_fraction"] = ok["near_category_agreement_fraction"].astype(float)
    if "high_comfort_precision" in ok.columns:
        ok["high_comfort_precision"] = pd.to_numeric(ok["high_comfort_precision"], errors="coerce")
    if "high_comfort_recall" in ok.columns:
        ok["high_comfort_recall"] = pd.to_numeric(ok["high_comfort_recall"], errors="coerce")
    ok["abs_score_bias_mean"] = ok["score_bias_mean"].abs()
    ok["verification_rank_score"] = (
        ok["score_mae"]
        + ok["score_rmse"] * 0.35
        + ok["abs_score_bias_mean"] * 0.75
        + (1.0 - ok["near_category_agreement_fraction"]) * 20.0
    )
    return ok


def _apply_threshold_flags(summary: pd.DataFrame) -> pd.DataFrame:
    flagged = summary.copy()
    flagged["passes_mae_threshold"] = False
    flagged["passes_near_category_threshold"] = False
    flagged["passes_bias_threshold"] = False
    flagged["passes_benchmark_thresholds"] = False
    flagged["benchmark_threshold_status"] = "error"

    ok_mask = flagged["status"] == "ok"
    if not ok_mask.any():
        return flagged

    flagged.loc[ok_mask, "passes_mae_threshold"] = (
        flagged.loc[ok_mask, "score_mae"].astype(float) <= VERIFICATION_BENCHMARK_THRESHOLDS["score_mae_max"]
    )
    flagged.loc[ok_mask, "passes_near_category_threshold"] = (
        flagged.loc[ok_mask, "near_category_agreement_fraction"].astype(float)
        >= VERIFICATION_BENCHMARK_THRESHOLDS["near_category_agreement_min"]
    )
    flagged.loc[ok_mask, "passes_bias_threshold"] = (
        flagged.loc[ok_mask, "score_bias_mean"].astype(float).abs()
        <= VERIFICATION_BENCHMARK_THRESHOLDS["abs_score_bias_mean_max"]
    )
    flagged.loc[ok_mask, "passes_benchmark_thresholds"] = (
        flagged.loc[ok_mask, "passes_mae_threshold"]
        & flagged.loc[ok_mask, "passes_near_category_threshold"]
        & flagged.loc[ok_mask, "passes_bias_threshold"]
    )
    flagged.loc[ok_mask, "benchmark_threshold_status"] = np.where(
        flagged.loc[ok_mask, "passes_benchmark_thresholds"],
        "pass",
        "warn",
    )
    return flagged


def format_benchmark_table(summary: pd.DataFrame) -> str:
    columns = [
        "region",
        "date",
        "status",
        "benchmark_threshold_status",
        "score_bias_mean",
        "score_mae",
        "score_rmse",
        "exact_category_agreement_fraction",
        "near_category_agreement_fraction",
    ]
    present_columns = [column for column in columns if column in summary.columns]
    ranking_columns = [column for column in ["score_mae", "score_rmse", "region", "date"] if column in summary.columns]
    return summary.loc[:, present_columns].sort_values(ranking_columns).to_string(index=False)


def _case_colors(frame: pd.DataFrame) -> list[str]:
    return ["#2e8b57" if bool(value) else "#d95f02" for value in frame["passes_benchmark_thresholds"]]


def _save_chart(fig, path: Path) -> Path:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_mae_bar_chart(ok_cases: pd.DataFrame, output_dir: Path, stem: str) -> Path | None:
    if ok_cases.empty:
        return None
    chart_data = ok_cases.sort_values(["score_mae", "score_rmse", "region", "date"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(chart_data["case_label"], chart_data["score_mae"], color=_case_colors(chart_data))
    ax.axhline(VERIFICATION_BENCHMARK_THRESHOLDS["score_mae_max"], color="#6b7280", linestyle="--", linewidth=1.1)
    ax.set_title("Benchmark MAE by Region and Date")
    ax.set_ylabel("Score MAE")
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_mae_bar.png")


def _write_agreement_bar_chart(ok_cases: pd.DataFrame, output_dir: Path, stem: str) -> Path | None:
    if ok_cases.empty:
        return None
    chart_data = ok_cases.sort_values(["near_category_agreement_fraction", "exact_category_agreement_fraction"], ascending=False)
    positions = np.arange(len(chart_data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(
        positions - width / 2,
        chart_data["exact_category_agreement_fraction"],
        width=width,
        color="#5b8ff9",
        label="Exact agreement",
    )
    ax.bar(
        positions + width / 2,
        chart_data["near_category_agreement_fraction"],
        width=width,
        color="#61d9a2",
        label="Near agreement",
    )
    ax.axhline(
        VERIFICATION_BENCHMARK_THRESHOLDS["near_category_agreement_min"],
        color="#6b7280",
        linestyle="--",
        linewidth=1.1,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(chart_data["case_label"], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Category Agreement by Region and Date")
    ax.set_ylabel("Agreement fraction")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_agreement_bar.png")


def _write_bias_rmse_scatter(ok_cases: pd.DataFrame, output_dir: Path, stem: str) -> Path | None:
    if ok_cases.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    regions = list(dict.fromkeys(ok_cases["region"].tolist()))
    cmap = plt.get_cmap("tab10")
    for index, region in enumerate(regions):
        subset = ok_cases.loc[ok_cases["region"] == region]
        ax.scatter(
            subset["score_bias_mean"],
            subset["score_rmse"],
            s=70,
            alpha=0.9,
            color=cmap(index % 10),
            label=region,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["date"].strftime("%m/%d"),
                (row["score_bias_mean"], row["score_rmse"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7.5,
            )
    ax.axvline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7)
    ax.axvline(VERIFICATION_BENCHMARK_THRESHOLDS["abs_score_bias_mean_max"], color="#b0b4b9", linestyle="--", linewidth=1.0)
    ax.axvline(-VERIFICATION_BENCHMARK_THRESHOLDS["abs_score_bias_mean_max"], color="#b0b4b9", linestyle="--", linewidth=1.0)
    ax.set_title("Bias vs RMSE")
    ax.set_xlabel("Mean score bias")
    ax.set_ylabel("Score RMSE")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_bias_rmse_scatter.png")


def _write_ranked_case_chart(ok_cases: pd.DataFrame, output_dir: Path, stem: str) -> Path | None:
    if ok_cases.empty:
        return None
    chart_data = ok_cases.sort_values(["verification_rank_score", "score_mae", "region", "date"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.barh(chart_data["case_label"], chart_data["verification_rank_score"], color=_case_colors(chart_data))
    ax.set_title("Ranked Verification Cases (Best to Worst)")
    ax.set_xlabel("Composite verification score (lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_ranked_cases.png")


def _write_timeseries_chart(
    ok_cases: pd.DataFrame,
    *,
    y_column: str,
    y_label: str,
    title: str,
    filename_suffix: str,
    output_dir: Path,
    stem: str,
    threshold: float | None = None,
) -> Path | None:
    if ok_cases.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for region, subset in ok_cases.sort_values("date").groupby("region"):
        ax.plot(subset["date"], subset[y_column], marker="o", linewidth=1.8, label=region)
    if threshold is not None:
        ax.axhline(threshold, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Valid date")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.autofmt_xdate(rotation=25)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_{filename_suffix}.png")


def _write_benchmark_charts(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> dict[str, Path]:
    ok_cases = _ok_cases(summary)
    charts: dict[str, Path] = {}
    candidates = {
        "mae_bar_chart": _write_mae_bar_chart(ok_cases, output_dir, stem),
        "agreement_bar_chart": _write_agreement_bar_chart(ok_cases, output_dir, stem),
        "bias_rmse_scatter": _write_bias_rmse_scatter(ok_cases, output_dir, stem),
        "ranked_case_chart": _write_ranked_case_chart(ok_cases, output_dir, stem),
        "mae_timeseries_chart": _write_timeseries_chart(
            ok_cases,
            y_column="score_mae",
            y_label="Score MAE",
            title="MAE Over Benchmark Dates",
            filename_suffix="mae_timeseries",
            output_dir=output_dir,
            stem=stem,
            threshold=VERIFICATION_BENCHMARK_THRESHOLDS["score_mae_max"],
        ),
        "agreement_timeseries_chart": _write_timeseries_chart(
            ok_cases,
            y_column="near_category_agreement_fraction",
            y_label="Near category agreement",
            title="Near Category Agreement Over Benchmark Dates",
            filename_suffix="agreement_timeseries",
            output_dir=output_dir,
            stem=stem,
            threshold=VERIFICATION_BENCHMARK_THRESHOLDS["near_category_agreement_min"],
        ),
        "bias_timeseries_chart": _write_timeseries_chart(
            ok_cases,
            y_column="score_bias_mean",
            y_label="Mean score bias",
            title="Bias Over Benchmark Dates",
            filename_suffix="bias_timeseries",
            output_dir=output_dir,
            stem=stem,
            threshold=0.0,
        ),
    }
    for key, value in candidates.items():
        if value is not None:
            charts[key] = value
    return charts


def _write_component_heatmap(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    ok_cases = _ok_cases(summary)
    component_columns = [
        ("temp_mae", "Temp"),
        ("dewpoint_mae", "Dew point"),
        ("cloud_mae", "Cloud"),
        ("precip_mae", "Precip"),
        ("reliability_score_mae", "Reliability"),
        ("disruption_penalty_mae", "Disruption"),
    ]
    present = [(column, label) for column, label in component_columns if column in ok_cases.columns]
    if ok_cases.empty or not present:
        return None

    chart_data = ok_cases.sort_values(["verification_rank_score", "region", "date"])
    matrix = np.array([[float(row[column]) for column, _ in present] for _, row in chart_data.iterrows()], dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, max(3.8, 0.5 * len(chart_data) + 1.8)))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([label for _, label in present], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(chart_data)))
    ax.set_yticklabels(chart_data["case_label"])
    ax.set_title("Component MAE Heatmap")
    cbar = fig.colorbar(image, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Mean absolute component error")
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_component_heatmap.png")


def _write_region_summary(summary: pd.DataFrame, *, output_dir: Path, stem: str) -> tuple[pd.DataFrame, Path | None]:
    ok_cases = _ok_cases(summary)
    if ok_cases.empty:
        return pd.DataFrame(), None

    aggregations: dict[str, tuple[str, str]] = {
        "case_count": ("region", "size"),
        "mean_score_bias": ("score_bias_mean", "mean"),
        "mean_score_mae": ("score_mae", "mean"),
        "mean_score_rmse": ("score_rmse", "mean"),
        "mean_exact_category_agreement": ("exact_category_agreement_fraction", "mean"),
        "mean_near_category_agreement": ("near_category_agreement_fraction", "mean"),
        "pass_rate": ("passes_benchmark_thresholds", "mean"),
    }
    if "high_comfort_precision" in ok_cases.columns:
        aggregations["mean_high_comfort_precision"] = ("high_comfort_precision", "mean")
    if "high_comfort_recall" in ok_cases.columns:
        aggregations["mean_high_comfort_recall"] = ("high_comfort_recall", "mean")

    region_summary = ok_cases.groupby("region", as_index=False).agg(**aggregations).sort_values(
        ["mean_score_mae", "mean_score_rmse", "region"]
    )
    csv_path = output_dir / f"comfortwx_verify_benchmark_{stem}_region_summary.csv"
    region_summary.to_csv(csv_path, index=False)
    return region_summary, csv_path


def _write_region_summary_chart(region_summary: pd.DataFrame, *, output_dir: Path, stem: str) -> Path | None:
    if region_summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(region_summary))
    width = 0.38
    ax.bar(
        positions - width / 2,
        region_summary["mean_score_mae"],
        width=width,
        color="#4f8ad9",
        label="Mean MAE",
    )
    ax.bar(
        positions + width / 2,
        region_summary["mean_score_rmse"],
        width=width,
        color="#c96b50",
        label="Mean RMSE",
    )
    ax.axhline(VERIFICATION_BENCHMARK_THRESHOLDS["score_mae_max"], color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(region_summary["region"], rotation=20, ha="right")
    ax.set_ylabel("Score error")
    ax.set_title("Regional Verification Summary")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    return _save_chart(fig, output_dir / f"comfortwx_verify_benchmark_{stem}_region_summary.png")


def _html_table(summary: pd.DataFrame) -> str:
    ordered_columns = [
        "region",
        "date",
        "benchmark_threshold_status",
        "score_bias_mean",
        "score_mae",
        "score_rmse",
        "exact_category_agreement_fraction",
        "near_category_agreement_fraction",
        "missed_high_comfort_cell_count",
        "false_high_comfort_cell_count",
    ]
    present_columns = [column for column in ordered_columns if column in summary.columns]
    display = summary.loc[:, present_columns].copy()
    return display.to_html(index=False, border=0, classes="summary-table")


def _write_benchmark_html_report(
    summary: pd.DataFrame,
    *,
    charts: dict[str, Path],
    region_summary: pd.DataFrame,
    region_summary_csv_path: Path | None,
    output_dir: Path,
    stem: str,
) -> Path:
    ok_cases = _ok_cases(summary)
    report_path = output_dir / f"comfortwx_verify_benchmark_{stem}.html"

    chart_titles = {
        "mae_bar_chart": "MAE by case",
        "agreement_bar_chart": "Category agreement by case",
        "bias_rmse_scatter": "Bias vs RMSE",
        "ranked_case_chart": "Ranked verification cases",
        "mae_timeseries_chart": "MAE over time",
        "agreement_timeseries_chart": "Agreement over time",
        "bias_timeseries_chart": "Bias over time",
        "component_heatmap": "Component MAE heatmap",
        "region_summary_chart": "Regional verification summary",
    }
    chart_blocks = [
        f'<figure class="chart-card"><img src="{html.escape(path.name)}" alt="{html.escape(chart_titles.get(key, key))}"><figcaption>{html.escape(chart_titles.get(key, key))}</figcaption></figure>'
        for key, path in charts.items()
    ]

    case_cards: list[str] = []
    for _, row in ok_cases.sort_values(["date", "region"]).iterrows():
        thumbnails = []
        thumb_columns = [
            ("forecast_score_map_path", "Forecast score"),
            ("analysis_score_map_path", "Analysis score"),
            ("score_difference_map_path", "Score difference"),
            ("absolute_error_map_path", "Absolute error"),
            ("category_disagreement_map_path", "Category disagreement"),
            ("missed_high_comfort_map_path", "Missed high comfort"),
            ("false_high_comfort_map_path", "False high comfort"),
        ]
        for column, label in thumb_columns:
            if column in row and pd.notna(row[column]) and str(row[column]).strip():
                rel = Path(str(row[column])).name
                thumbnails.append(
                    f'<a class="thumb" href="{html.escape(rel)}"><img src="{html.escape(rel)}" alt="{html.escape(label)}"><span>{html.escape(label)}</span></a>'
                )
        case_cards.append(
            "\n".join(
                [
                    '<section class="case-card">',
                    f"<h3>{html.escape(str(row['region']))} | {row['date']:%Y-%m-%d}</h3>",
                    (
                        f"<p>MAE {float(row['score_mae']):.2f} | RMSE {float(row['score_rmse']):.2f} | "
                        f"Near agreement {float(row['near_category_agreement_fraction']):.1%} | "
                        f"Status {html.escape(str(row['benchmark_threshold_status']))} | "
                        f"<a href=\"{html.escape(Path(str(row['component_metrics_csv_path'])).name)}\">component metrics CSV</a></p>"
                    ),
                    '<div class="thumb-grid">',
                    *thumbnails,
                    "</div>",
                    "</section>",
                ]
            )
        )

    ok_count = int((summary["status"] == "ok").sum()) if "status" in summary.columns else 0
    passing_count = int(summary.get("passes_benchmark_thresholds", pd.Series(dtype=bool)).fillna(False).sum())
    best_case = ok_cases.sort_values(["verification_rank_score", "score_mae"]).head(1)
    worst_case = ok_cases.sort_values(["verification_rank_score", "score_mae"], ascending=[False, False]).head(1)
    best_case_text = (
        f"Best case: {best_case.iloc[0]['region']} {best_case.iloc[0]['date']:%Y-%m-%d} (MAE {best_case.iloc[0]['score_mae']:.2f})"
        if not best_case.empty
        else "Best case: n/a"
    )
    worst_case_text = (
        f"Worst case: {worst_case.iloc[0]['region']} {worst_case.iloc[0]['date']:%Y-%m-%d} (MAE {worst_case.iloc[0]['score_mae']:.2f})"
        if not worst_case.empty
        else "Worst case: n/a"
    )
    region_table_html = region_summary.to_html(index=False, border=0, classes="summary-table") if not region_summary.empty else "<p>No successful regional summary available.</p>"
    region_summary_link = (
        f"<p><a href='{html.escape(region_summary_csv_path.name)}'>Download regional summary CSV</a></p>"
        if region_summary_csv_path is not None
        else ""
    )
    report_path.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html lang='en'>",
                "<head>",
                "<meta charset='utf-8'>",
                "<title>Comfort Index Verification Benchmark</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 24px; color: #1c2330; background: #fafbfc; }",
                "h1, h2, h3 { margin-bottom: 0.35rem; }",
                ".meta { color: #556070; margin-bottom: 1.2rem; }",
                ".summary-table { border-collapse: collapse; width: 100%; background: white; }",
                ".summary-table th, .summary-table td { border: 1px solid #d9dee5; padding: 8px 10px; font-size: 0.92rem; }",
                ".summary-table th { background: #eef2f6; }",
                ".chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin: 22px 0; }",
                ".chart-card, .case-card { background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 12px; }",
                ".chart-card img { width: 100%; height: auto; display: block; }",
                ".chart-card figcaption { margin-top: 8px; color: #46505a; font-size: 0.9rem; }",
                ".thumb-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }",
                ".thumb { text-decoration: none; color: inherit; font-size: 0.84rem; }",
                ".thumb img { width: 100%; height: auto; border: 1px solid #d9dee5; border-radius: 6px; display: block; margin-bottom: 6px; }",
                ".status-line { display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0 16px; }",
                ".status-pill { background: #eef2f6; border-radius: 999px; padding: 6px 12px; font-size: 0.9rem; }",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Comfort Index Verification Benchmark</h1>",
                f"<p class='meta'>Cases attempted: {len(summary)} | Successful cases: {ok_count} | Cases meeting thresholds: {passing_count}</p>",
                "<div class='status-line'>",
                f"<div class='status-pill'>MAE threshold: ≤ {VERIFICATION_BENCHMARK_THRESHOLDS['score_mae_max']:.1f}</div>",
                f"<div class='status-pill'>Near agreement threshold: ≥ {VERIFICATION_BENCHMARK_THRESHOLDS['near_category_agreement_min']:.0%}</div>",
                f"<div class='status-pill'>Absolute bias threshold: ≤ {VERIFICATION_BENCHMARK_THRESHOLDS['abs_score_bias_mean_max']:.1f}</div>",
                "</div>",
                f"<p>{html.escape(best_case_text)} | {html.escape(worst_case_text)}</p>",
                "<h2>Summary</h2>",
                _html_table(summary),
                "<h2>Regional Rollup</h2>",
                region_summary_link,
                region_table_html,
                "<h2>Benchmark Charts</h2>",
                "<div class='chart-grid'>",
                *chart_blocks,
                "</div>",
                "<h2>Case Maps</h2>",
                *case_cards,
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def _write_verification_site(
    *,
    summary: pd.DataFrame,
    summary_path: Path,
    charts: dict[str, Path],
    report_path: Path,
    region_summary_csv_path: Path | None,
    output_dir: Path,
    stem: str,
) -> Path:
    site_dir = output_dir / "verification_site" / stem
    site_dir.mkdir(parents=True, exist_ok=True)

    asset_paths: set[Path] = {summary_path, report_path}
    asset_paths.update(charts.values())
    if region_summary_csv_path is not None:
        asset_paths.add(region_summary_csv_path)

    for column in [
        "forecast_score_map_path",
        "analysis_score_map_path",
        "score_difference_map_path",
        "absolute_error_map_path",
        "category_disagreement_map_path",
        "missed_high_comfort_map_path",
        "false_high_comfort_map_path",
        "summary_csv_path",
        "point_metrics_csv_path",
        "component_metrics_csv_path",
        "request_summary_csv_path",
        "request_detail_csv_path",
    ]:
        if column in summary.columns:
            for value in summary[column].dropna():
                path = Path(str(value))
                if path.exists():
                    asset_paths.add(path)

    for asset in asset_paths:
        if asset.exists():
            shutil.copy2(asset, site_dir / asset.name)

    primary_index = site_dir / "index.html"
    shutil.copy2(report_path, primary_index)

    latest_dir = output_dir / "verification_site" / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(site_dir, latest_dir)
    return primary_index


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
                    "absolute_error_map_path": str(outputs["absolute_error_map"]),
                    "category_disagreement_map_path": str(outputs["category_disagreement_map"]),
                    "missed_high_comfort_map_path": str(outputs["missed_high_comfort_map"]),
                    "false_high_comfort_map_path": str(outputs["false_high_comfort_map"]),
                    "summary_csv_path": str(outputs["summary_csv"]),
                    "point_metrics_csv_path": str(outputs["point_metrics_csv"]),
                    "component_metrics_csv_path": str(outputs["component_metrics_csv"]),
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
                    "forecast_score_map_path": "",
                    "analysis_score_map_path": "",
                    "score_difference_map_path": "",
                    "absolute_error_map_path": "",
                    "category_disagreement_map_path": "",
                    "missed_high_comfort_map_path": "",
                    "false_high_comfort_map_path": "",
                    "summary_csv_path": "",
                    "point_metrics_csv_path": "",
                    "component_metrics_csv_path": "",
                    "request_summary_csv_path": "",
                    "request_detail_csv_path": "",
                }
            )
    return _apply_threshold_flags(pd.DataFrame.from_records(records))


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
    summary_path = output_dir / f"comfortwx_verify_benchmark_{stem}.csv"
    summary.to_csv(summary_path, index=False)
    charts = _write_benchmark_charts(summary, output_dir=output_dir, stem=stem)
    component_heatmap = _write_component_heatmap(summary, output_dir=output_dir, stem=stem)
    if component_heatmap is not None:
        charts["component_heatmap"] = component_heatmap
    region_summary, region_summary_csv_path = _write_region_summary(summary, output_dir=output_dir, stem=stem)
    region_summary_chart = _write_region_summary_chart(region_summary, output_dir=output_dir, stem=stem)
    if region_summary_chart is not None:
        charts["region_summary_chart"] = region_summary_chart
    report_path = _write_benchmark_html_report(
        summary,
        charts=charts,
        region_summary=region_summary,
        region_summary_csv_path=region_summary_csv_path,
        output_dir=output_dir,
        stem=stem,
    )
    verification_site_index = _write_verification_site(
        summary=summary,
        summary_path=summary_path,
        charts=charts,
        report_path=report_path,
        region_summary_csv_path=region_summary_csv_path,
        output_dir=output_dir,
        stem=stem,
    )

    print(f"Saved verification benchmark summary: {summary_path}")
    if region_summary_csv_path is not None:
        print(f"Saved regional benchmark summary: {region_summary_csv_path}")
    for chart_name, path in charts.items():
        print(f"Saved {chart_name}: {path}")
    print(f"Saved verification benchmark report: {report_path}")
    print(f"Saved verification site index: {verification_site_index}")
    print(format_benchmark_table(summary))


if __name__ == "__main__":
    main()
