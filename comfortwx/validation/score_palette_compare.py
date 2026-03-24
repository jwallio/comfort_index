"""Render stitched CONUS public score-map palette variants from saved daily fields."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import xarray as xr

from comfortwx.config import OUTPUT_DIR, STITCHED_CONUS_PRESENTATION
from comfortwx.mapping.plotting import plot_raw_score_map


PALETTE_VARIANTS: tuple[str, ...] = (
    "premium_muted",
    "bold_social",
    "blue_green_yellow_magenta",
)


def _default_daily_fields_path(valid_date: datetime.date) -> Path:
    return (
        OUTPUT_DIR
        / f"comfortwx_mosaic_west_coast_southwest_rockies_plains_southeast_northeast_great_lakes_openmeteo_daily_fields_{valid_date:%Y%m%d}.nc"
    )


def render_score_palette_variants(
    *,
    daily_fields_path: Path,
    valid_date: datetime.date,
    output_dir: Path,
    variants: tuple[str, ...] = PALETTE_VARIANTS,
) -> list[Path]:
    """Render all stitched CONUS score-map palette variants."""

    if not daily_fields_path.exists():
        raise FileNotFoundError(f"Daily fields file not found: {daily_fields_path}")

    with xr.open_dataset(daily_fields_path) as dataset:
        daily = dataset.load()

    suffix = f"_daily_fields_{valid_date:%Y%m%d}"
    base_prefix = daily_fields_path.stem.removesuffix(suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for variant in variants:
        output_path = output_dir / f"{base_prefix}_presentation_score_{variant}_{valid_date:%Y%m%d}.png"
        plot_raw_score_map(
            daily=daily,
            valid_date=valid_date,
            output_path=output_path,
            presentation=True,
            presentation_theme=variant,
            product_metadata={
                "product_title": str(STITCHED_CONUS_PRESENTATION["title"]),
                "product_subtitle": str(STITCHED_CONUS_PRESENTATION["product_subtitle"]),
                "subtitle_source_line": str(STITCHED_CONUS_PRESENTATION["subtitle_source_line"]),
            },
            presentation_canvas="stitched_conus",
        )
        written_paths.append(output_path)
    return written_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render stitched CONUS score-map palette variants.")
    parser.add_argument("--date", required=True, help="Valid date in YYYY-MM-DD format.")
    parser.add_argument("--daily-fields-path", default=None, help="Optional stitched daily-fields NetCDF path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for score-map variant PNGs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    valid_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    daily_fields_path = Path(args.daily_fields_path) if args.daily_fields_path else _default_daily_fields_path(valid_date)
    output_dir = Path(args.output_dir)
    written = render_score_palette_variants(
        daily_fields_path=daily_fields_path,
        valid_date=valid_date,
        output_dir=output_dir,
    )
    print(f"Rendered {len(written)} stitched score-map palette variants for {valid_date:%Y-%m-%d}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
