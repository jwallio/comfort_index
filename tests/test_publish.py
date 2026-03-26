from __future__ import annotations

from datetime import date

import pandas as pd

from comfortwx.mapping.plotting import resolve_presentation_theme
from comfortwx.publishing import (
    build_archive_run_directory,
    resolve_publish_preset,
    write_archive_index,
    write_pilot_day_index,
    write_pilot_day_status_summary,
    write_publish_bundle,
)


def test_write_publish_bundle_creates_csv_and_json(tmp_path) -> None:
    source_file = tmp_path / "example.png"
    source_file.write_text("placeholder", encoding="utf-8")

    csv_path, json_path = write_publish_bundle(
        output_dir=tmp_path,
        valid_date=date(2026, 3, 24),
        preset_name="standard",
        product_kind="region",
        product_slug="region_southwest_openmeteo",
        theme_name="shareable",
        bundle_files={"presentation_score_map": source_file},
    )

    assert csv_path.exists()
    assert json_path.exists()


def test_theme_and_publish_preset_resolution_support_named_presets() -> None:
    theme = resolve_presentation_theme("shareable")
    public_theme = resolve_presentation_theme("public")
    premium_muted_theme = resolve_presentation_theme("premium_muted")
    bold_social_theme = resolve_presentation_theme("bold_social")
    bgym_theme = resolve_presentation_theme("blue_green_yellow_magenta")
    preset = resolve_publish_preset("standard")

    assert theme["name"] == "shareable"
    assert public_theme["name"] == "public"
    assert premium_muted_theme["name"] == "premium_muted"
    assert bold_social_theme["name"] == "bold_social"
    assert bgym_theme["name"] == "blue_green_yellow_magenta"
    assert len(public_theme["raw_score_levels"]) == 11
    assert len(premium_muted_theme["raw_score_levels"]) == 11
    assert preset["name"] == "standard"


def test_write_pilot_day_index_creates_csv_json_and_html(tmp_path) -> None:
    preview_png = tmp_path / "comfortwx_region_southwest_openmeteo_presentation_score_20260324.png"
    preview_png.write_text("placeholder", encoding="utf-8")
    category_png = tmp_path / "comfortwx_region_southwest_openmeteo_presentation_category_20260324.png"
    category_png.write_text("placeholder", encoding="utf-8")
    city_rankings_csv = tmp_path / "comfortwx_city_rankings_20260324.csv"
    pd.DataFrame(
        [
            {
                "city": "San Diego, CA",
                "score": 82.5,
                "category": "Ideal",
                "sample_lat": 32.7,
                "sample_lon": -117.1,
                "distance_degrees": 0.0,
                "priority": 8,
                "ranking_group": "best",
                "ranking_position": 1,
            },
            {
                "city": "Miami, FL",
                "score": 31.2,
                "category": "Fair",
                "sample_lat": 25.8,
                "sample_lon": -80.2,
                "distance_degrees": 0.0,
                "priority": 36,
                "ranking_group": "worst",
                "ranking_position": 1,
            },
        ]
    ).to_csv(city_rankings_csv, index=False)
    status_csv_path, status_json_path = write_pilot_day_status_summary(
        output_dir=tmp_path,
        valid_date=date(2026, 3, 24),
        source_name="openmeteo",
        status_record={
            "valid_date": "2026-03-24",
            "source": "openmeteo",
            "overall_run_status": "completed",
        },
    )
    csv_path, json_path, html_path = write_pilot_day_index(
        output_dir=tmp_path,
        valid_date=date(2026, 3, 24),
        source_name="openmeteo",
        presentation_theme="shareable",
        publish_preset_name="standard",
        product_rows=[
            {
                "product_type": "region",
                "product_name": "southwest",
                "valid_date": "2026-03-24",
                "daily_fields_path": "output/example.nc",
                "presentation_score_map_path": str(preview_png),
                "presentation_category_map_path": str(category_png),
                "city_rankings_csv_path": str(city_rankings_csv),
                "status": "completed",
            }
        ],
        status_summary_csv_path=status_csv_path,
        status_summary_json_path=status_json_path,
    )

    assert csv_path.exists()
    assert json_path.exists()
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Daily Maps for March 24, 2026" in html_text
    assert "<img src=" in html_text
    assert "<table" not in html_text
    assert "Top 10 best cities" in html_text
    assert "San Diego, CA" in html_text
    assert "Top 10 toughest cities" in html_text


def test_build_archive_run_directory_supports_configured_layouts(tmp_path) -> None:
    nested = build_archive_run_directory(archive_root=tmp_path, valid_date=date(2026, 3, 24), layout="year/month/day")
    compact = build_archive_run_directory(archive_root=tmp_path, valid_date=date(2026, 3, 24), layout="yyyymmdd")

    assert nested == tmp_path / "2026" / "03" / "24"
    assert compact == tmp_path / "20260324"


def test_write_archive_index_scans_archived_pilot_day_runs(tmp_path) -> None:
    run_dir = tmp_path / "2026" / "03" / "24"
    run_dir.mkdir(parents=True)
    preview_png = run_dir / "comfortwx_mosaic_example_presentation_score_20260324.png"
    preview_png.write_text("placeholder", encoding="utf-8")
    category_png = run_dir / "comfortwx_mosaic_example_presentation_category_20260324.png"
    category_png.write_text("placeholder", encoding="utf-8")
    status_csv_path, status_json_path = write_pilot_day_status_summary(
        output_dir=run_dir,
        valid_date=date(2026, 3, 24),
        source_name="openmeteo",
        status_record={
            "valid_date": "2026-03-24",
            "source": "openmeteo",
            "overall_run_status": "completed",
        },
    )
    write_pilot_day_index(
        output_dir=run_dir,
        valid_date=date(2026, 3, 24),
        source_name="openmeteo",
        presentation_theme="shareable",
        publish_preset_name="standard",
        product_rows=[
            {
                "product_type": "mosaic",
                "product_name": "west_coast+southwest+rockies",
                "valid_date": "2026-03-24",
                "daily_fields_path": str(run_dir / "example.nc"),
                "presentation_score_map_path": str(preview_png),
                "presentation_category_map_path": str(category_png),
                "status": "completed",
            }
        ],
        status_summary_csv_path=status_csv_path,
        status_summary_json_path=status_json_path,
    )

    csv_path, json_path, html_path = write_archive_index(archive_root=tmp_path)

    assert csv_path.exists()
    assert json_path.exists()
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Daily Outdoor Comfort Maps" in html_text
    assert "<img src=" in html_text
    assert "<table" not in html_text


def test_write_archive_index_builds_day_selector_and_rewrites_run_pages(tmp_path) -> None:
    for valid_day in (24, 25):
        run_dir = tmp_path / "2026" / "03" / f"{valid_day:02d}"
        run_dir.mkdir(parents=True)
        preview_png = run_dir / f"comfortwx_mosaic_conus_presentation_score_202603{valid_day:02d}.png"
        preview_png.write_text("placeholder", encoding="utf-8")
        category_png = run_dir / f"comfortwx_mosaic_conus_presentation_category_202603{valid_day:02d}.png"
        category_png.write_text("placeholder", encoding="utf-8")
        status_csv_path, status_json_path = write_pilot_day_status_summary(
            output_dir=run_dir,
            valid_date=date(2026, 3, valid_day),
            source_name="openmeteo",
            status_record={
                "valid_date": f"2026-03-{valid_day:02d}",
                "source": "openmeteo",
                "overall_run_status": "completed",
            },
        )
        write_pilot_day_index(
            output_dir=run_dir,
            valid_date=date(2026, 3, valid_day),
            source_name="openmeteo",
            presentation_theme="shareable",
            publish_preset_name="standard",
            product_rows=[
                {
                    "product_type": "mosaic",
                    "product_name": "west_coast+southwest+rockies+plains+southeast+northeast+great_lakes",
                    "valid_date": f"2026-03-{valid_day:02d}",
                    "daily_fields_path": str(run_dir / "example.nc"),
                    "presentation_score_map_path": str(preview_png),
                    "presentation_category_map_path": str(category_png),
                    "status": "completed",
                }
            ],
            status_summary_csv_path=status_csv_path,
            status_summary_json_path=status_json_path,
        )

    _, _, archive_html_path = write_archive_index(archive_root=tmp_path)

    archive_html = archive_html_path.read_text(encoding="utf-8")
    assert "Forecast day" in archive_html
    assert "Day 1 | Tue 03/24/26" in archive_html
    assert "Day 2 | Wed 03/25/26" in archive_html

    run_html_path = tmp_path / "2026" / "03" / "25" / "comfortwx_pilot_day_openmeteo_20260325_index.html"
    run_html = run_html_path.read_text(encoding="utf-8")
    assert "Forecast day" in run_html
    assert "../24/comfortwx_pilot_day_openmeteo_20260324_index.html" in run_html
    assert "stitched CONUS score and category maps" in run_html


def test_write_archive_index_links_verification_dashboard_when_present(tmp_path) -> None:
    verification_dir = tmp_path / "verification"
    verification_dir.mkdir(parents=True)
    (verification_dir / "index.html").write_text("<html>verification</html>", encoding="utf-8")

    _, _, html_path = write_archive_index(archive_root=tmp_path)

    html_text = html_path.read_text(encoding="utf-8")
    assert "Verification dashboard" in html_text
    assert "verification/index.html" in html_text
