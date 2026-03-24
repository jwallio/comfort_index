"""Helpers for publish-style output bundles."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from comfortwx.config import ARCHIVE_SETTINGS, PRODUCT_METADATA, PUBLISH_PRESETS


def resolve_publish_preset(preset_name: str) -> dict[str, object]:
    """Return the configured publish preset."""

    normalized_name = preset_name.strip().lower()
    if normalized_name not in PUBLISH_PRESETS:
        raise ValueError(f"Unknown publish preset '{preset_name}'. Available presets: {', '.join(sorted(PUBLISH_PRESETS))}.")
    preset = dict(PUBLISH_PRESETS[normalized_name])
    preset["name"] = normalized_name
    return preset


def write_publish_bundle(
    *,
    output_dir: Path,
    valid_date: date,
    preset_name: str,
    product_kind: str,
    product_slug: str,
    theme_name: str,
    bundle_files: dict[str, Path | None],
    product_metadata: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write CSV and JSON manifests for a standardized product bundle."""

    metadata = dict(PRODUCT_METADATA)
    if product_metadata:
        metadata.update(product_metadata)

    rows: list[dict[str, object]] = []
    for role, path in bundle_files.items():
        if path is None:
            continue
        rows.append(
            {
                "role": role,
                "path": str(path),
                "filename": Path(path).name,
                "exists": Path(path).exists(),
                "product_kind": product_kind,
                "product_slug": product_slug,
                "publish_preset": preset_name,
                "presentation_theme": theme_name,
            }
        )

    bundle_frame = pd.DataFrame(rows)
    csv_path = output_dir / f"comfortwx_publish_{product_kind}_{product_slug}_{valid_date:%Y%m%d}_bundle.csv"
    json_path = output_dir / f"comfortwx_publish_{product_kind}_{product_slug}_{valid_date:%Y%m%d}_bundle.json"
    bundle_frame.to_csv(csv_path, index=False)

    payload = {
        "valid_date": valid_date.isoformat(),
        "publish_preset": preset_name,
        "product_kind": product_kind,
        "product_slug": product_slug,
        "presentation_theme": theme_name,
        "metadata": metadata,
        "files": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def write_pilot_day_index(
    *,
    output_dir: Path,
    valid_date: date,
    source_name: str,
    presentation_theme: str,
    publish_preset_name: str,
    product_rows: list[dict[str, object]],
    run_timestamp: datetime | None = None,
    status_summary_csv_path: Path | None = None,
    status_summary_json_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write CSV, JSON, and HTML master indexes for a one-date pilot product run."""

    frame = pd.DataFrame(product_rows)
    base_name = f"comfortwx_pilot_day_{source_name}_{valid_date:%Y%m%d}_index"
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"
    html_path = output_dir / f"{base_name}.html"
    frame.to_csv(csv_path, index=False)

    payload = {
        "valid_date": valid_date.isoformat(),
        "run_timestamp": (run_timestamp or datetime.now()).isoformat(timespec="seconds"),
        "source": source_name,
        "presentation_theme": presentation_theme,
        "publish_preset": publish_preset_name,
        "metadata": dict(PRODUCT_METADATA),
        "status_summary_csv_path": str(status_summary_csv_path or ""),
        "status_summary_json_path": str(status_summary_json_path or ""),
        "products": product_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html_parts = [
        "<html><head><meta charset='utf-8'><title>ComfortWX Pilot Day Index</title></head><body>",
        f"<h1>{PRODUCT_METADATA['product_title']} Pilot Day Index</h1>",
        f"<p>Valid date: {valid_date:%Y-%m-%d} | Run timestamp: {(run_timestamp or datetime.now()):%Y-%m-%d %H:%M:%S} | Source: {source_name} | Theme: {presentation_theme} | Preset: {publish_preset_name}</p>",
        (
            f"<p><a href='{Path(status_summary_csv_path).name}'>Status summary CSV</a> | "
            f"<a href='{Path(status_summary_json_path).name}'>Status summary JSON</a></p>"
            if status_summary_csv_path and status_summary_json_path
            else ""
        ),
        frame.to_html(index=False, escape=False),
        "</body></html>",
    ]
    html_path.write_text("".join(html_parts), encoding="utf-8")
    return csv_path, json_path, html_path


def write_pilot_day_status_summary(
    *,
    output_dir: Path,
    valid_date: date,
    source_name: str,
    status_record: dict[str, object],
) -> tuple[Path, Path]:
    """Write compact CSV and JSON status summaries for a pilot-day run."""

    base_name = f"comfortwx_pilot_day_{source_name}_{valid_date:%Y%m%d}_status"
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"
    pd.DataFrame([status_record]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(status_record, indent=2), encoding="utf-8")
    return csv_path, json_path


def build_archive_run_directory(
    *,
    archive_root: Path,
    valid_date: date,
    layout: str = ARCHIVE_SETTINGS["layout"],
) -> Path:
    """Return the dated archive directory for a pilot-day run."""

    normalized_layout = layout.strip().lower()
    if normalized_layout == "year/month/day":
        return archive_root / f"{valid_date:%Y}" / f"{valid_date:%m}" / f"{valid_date:%d}"
    if normalized_layout == "yyyymmdd":
        return archive_root / f"{valid_date:%Y%m%d}"
    raise ValueError("Unsupported archive layout. Available layouts: year/month/day, yyyymmdd.")


def _relative_link(path_value: str, *, from_dir: Path) -> str:
    target_path = Path(path_value)
    try:
        relative_target = target_path.relative_to(from_dir)
    except ValueError:
        try:
            relative_target = target_path.resolve().relative_to(from_dir.resolve())
        except ValueError:
            return path_value
    return relative_target.as_posix()


def write_archive_index(
    *,
    archive_root: Path,
) -> tuple[Path, Path, Path]:
    """Scan archived pilot-day runs and write archive-wide CSV, JSON, and HTML indexes."""

    archive_root.mkdir(parents=True, exist_ok=True)
    run_index_paths = sorted(archive_root.glob("**/comfortwx_pilot_day_*_index.json"))
    run_rows: list[dict[str, object]] = []
    run_payloads: list[tuple[Path, dict[str, object]]] = []
    for run_index_path in run_index_paths:
        payload = json.loads(run_index_path.read_text(encoding="utf-8"))
        run_payloads.append((run_index_path, payload))
        product_count = len(payload.get("products", []))
        run_rows.append(
            {
                "valid_date": payload.get("valid_date", ""),
                "run_timestamp": payload.get("run_timestamp", ""),
                "source": payload.get("source", ""),
                "presentation_theme": payload.get("presentation_theme", ""),
                "publish_preset": payload.get("publish_preset", ""),
                "product_count": product_count,
                "index_csv_path": _relative_link(str(run_index_path.with_suffix(".csv")), from_dir=archive_root),
                "index_json_path": _relative_link(str(run_index_path), from_dir=archive_root),
                "index_html_path": _relative_link(str(run_index_path.with_suffix(".html")), from_dir=archive_root),
                "status_summary_csv_path": _relative_link(str(payload.get("status_summary_csv_path", "")), from_dir=archive_root),
                "status_summary_json_path": _relative_link(str(payload.get("status_summary_json_path", "")), from_dir=archive_root),
            }
        )

    frame = pd.DataFrame(run_rows)
    base_name = ARCHIVE_SETTINGS["run_index_base_name"]
    csv_path = archive_root / f"{base_name}.csv"
    json_path = archive_root / f"{base_name}.json"
    html_path = archive_root / "index.html"
    frame.to_csv(csv_path, index=False)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": dict(PRODUCT_METADATA),
        "runs": run_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html_parts = [
        "<html><head><meta charset='utf-8'><title>ComfortWX Archive</title></head><body>",
        f"<h1>{PRODUCT_METADATA['product_title']} Archive</h1>",
        f"<p>{PRODUCT_METADATA['subtitle_source_line']}</p>",
        frame.to_html(index=False, escape=False),
    ]
    for run_index_path, payload in run_payloads:
        run_dir = run_index_path.parent
        html_parts.append(
            f"<h2>Run {payload.get('valid_date', '')} | {payload.get('source', '')} | {payload.get('run_timestamp', '')}</h2>"
        )
        html_parts.append("<ul>")
        html_parts.append(
            f"<li><a href='{_relative_link(str(run_index_path.with_suffix('.html')), from_dir=archive_root)}'>Pilot-day HTML index</a></li>"
        )
        html_parts.append(
            f"<li><a href='{_relative_link(str(run_index_path.with_suffix('.csv')), from_dir=archive_root)}'>Pilot-day CSV index</a></li>"
        )
        html_parts.append(
            f"<li><a href='{_relative_link(str(run_index_path), from_dir=archive_root)}'>Pilot-day JSON index</a></li>"
        )
        if payload.get("status_summary_csv_path"):
            html_parts.append(
                f"<li><a href='{_relative_link(str(payload.get('status_summary_csv_path')), from_dir=archive_root)}'>Status summary CSV</a></li>"
            )
        if payload.get("status_summary_json_path"):
            html_parts.append(
                f"<li><a href='{_relative_link(str(payload.get('status_summary_json_path')), from_dir=archive_root)}'>Status summary JSON</a></li>"
            )
        html_parts.append("</ul>")
        product_rows = payload.get("products", [])
        if product_rows:
            product_frame = pd.DataFrame(product_rows)
            for column in [
                "daily_fields_path",
                "debug_score_map_path",
                "debug_category_map_path",
                "presentation_score_map_path",
                "presentation_category_map_path",
                "summary_csv_path",
                "bundle_csv_path",
                "bundle_json_path",
                "samples_or_seam_path",
            ]:
                if column in product_frame:
                    product_frame[column] = product_frame[column].apply(
                        lambda value, run_dir=run_dir: f"<a href='{_relative_link(str(value), from_dir=archive_root)}'>{Path(str(value)).name}</a>" if value else ""
                    )
            html_parts.append(product_frame.to_html(index=False, escape=False))
    html_parts.append("</body></html>")
    html_path.write_text("".join(html_parts), encoding="utf-8")
    return csv_path, json_path, html_path
