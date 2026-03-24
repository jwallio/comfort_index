from __future__ import annotations

from datetime import date

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
            }
        ],
        status_summary_csv_path=status_csv_path,
        status_summary_json_path=status_json_path,
    )

    assert csv_path.exists()
    assert json_path.exists()
    assert html_path.exists()


def test_build_archive_run_directory_supports_configured_layouts(tmp_path) -> None:
    nested = build_archive_run_directory(archive_root=tmp_path, valid_date=date(2026, 3, 24), layout="year/month/day")
    compact = build_archive_run_directory(archive_root=tmp_path, valid_date=date(2026, 3, 24), layout="yyyymmdd")

    assert nested == tmp_path / "2026" / "03" / "24"
    assert compact == tmp_path / "20260324"


def test_write_archive_index_scans_archived_pilot_day_runs(tmp_path) -> None:
    run_dir = tmp_path / "2026" / "03" / "24"
    run_dir.mkdir(parents=True)
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
                "product_type": "region",
                "product_name": "southwest",
                "valid_date": "2026-03-24",
                "daily_fields_path": str(run_dir / "example.nc"),
            }
        ],
        status_summary_csv_path=status_csv_path,
        status_summary_json_path=status_json_path,
    )

    csv_path, json_path, html_path = write_archive_index(archive_root=tmp_path)

    assert csv_path.exists()
    assert json_path.exists()
    assert html_path.exists()
