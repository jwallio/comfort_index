"""Helpers for publish-style output bundles."""

from __future__ import annotations

import html
import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from comfortwx.config import ARCHIVE_SETTINGS, PRODUCT_METADATA, PUBLISH_PRESETS
from comfortwx.scoring.categories import category_colors, category_labels


_CITY_RANKING_CATEGORY_COLOR_MAP = dict(zip(category_labels(), category_colors(), strict=False))


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

    html_path.write_text(
        _build_pilot_day_gallery_html(
            valid_date=valid_date,
            source_name=source_name,
            presentation_theme=presentation_theme,
            publish_preset_name=publish_preset_name,
            product_rows=product_rows,
            run_timestamp=run_timestamp or datetime.now(),
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )
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
    if not path_value:
        return ""
    target_path = Path(path_value)
    try:
        relative_target = target_path.relative_to(from_dir)
    except ValueError:
        try:
            relative_target = target_path.resolve().relative_to(from_dir.resolve())
        except ValueError:
            normalized_value = path_value.replace("\\", "/")
            archive_marker = "/output/archive/"
            if archive_marker in normalized_value:
                archive_root = _find_archive_root(from_dir)
                if archive_root is not None:
                    archive_relative = normalized_value.split(archive_marker, 1)[1].lstrip("/")
                    normalized_target = archive_root.joinpath(*archive_relative.split("/"))
                    return os.path.relpath(normalized_target, start=from_dir).replace("\\", "/")
            try:
                return os.path.relpath(str(target_path), start=str(from_dir)).replace("\\", "/")
            except ValueError:
                return path_value
    return relative_target.as_posix()


def _find_archive_root(from_dir: Path) -> Path | None:
    base_name = str(ARCHIVE_SETTINGS["run_index_base_name"])
    for candidate in (from_dir, *from_dir.parents):
        if (candidate / f"{base_name}.json").exists() or (candidate / f"{base_name}.csv").exists():
            return candidate
        if candidate.name == "archive":
            return candidate
    return None


def _humanize_product_name(product_name: str) -> str:
    return product_name.replace("+", " + ").replace("_", " ").title()


def _token_count(product_name: str) -> int:
    return len([token for token in product_name.split("+") if token])


def _is_completed_product(product_row: dict[str, object]) -> bool:
    return str(product_row.get("status", "")).lower() == "completed"


def _product_image_rows(product_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in product_rows if _is_completed_product(row) and row.get("presentation_score_map_path")]


def _featured_product_row(product_rows: list[dict[str, object]]) -> dict[str, object] | None:
    completed_rows = _product_image_rows(product_rows)
    if not completed_rows:
        return None

    mosaics = [row for row in completed_rows if row.get("product_type") == "mosaic"]
    if mosaics:
        return max(mosaics, key=lambda row: _token_count(str(row.get("product_name", ""))))
    return completed_rows[0]


def _build_run_summary(product_rows: list[dict[str, object]]) -> str:
    completed = sum(1 for row in product_rows if _is_completed_product(row))
    attempted = len(product_rows)
    region_count = sum(1 for row in product_rows if row.get("product_type") == "region" and _is_completed_product(row))
    mosaic_count = sum(1 for row in product_rows if row.get("product_type") == "mosaic" and _is_completed_product(row))
    return f"{completed} of {attempted} products completed. {mosaic_count} stitched mosaics and {region_count} regions are shown below."


def _render_image_panel(*, image_href: str, label: str) -> str:
    return (
        "<figure class='image-panel'>"
        f"<img src='{html.escape(image_href, quote=True)}' alt='{html.escape(label, quote=True)}' loading='lazy'>"
        f"<figcaption>{html.escape(label)}</figcaption>"
        "</figure>"
    )


def _render_product_card(product_row: dict[str, object], *, from_dir: Path) -> str:
    score_href = _relative_link(str(product_row.get("presentation_score_map_path", "")), from_dir=from_dir)
    category_href = _relative_link(str(product_row.get("presentation_category_map_path", "")), from_dir=from_dir)
    if not score_href and not category_href:
        return ""

    product_label = _humanize_product_name(str(product_row.get("product_name", "")))
    product_type = str(product_row.get("product_type", "")).title()
    card_parts = [
        "<section class='product-card'>",
        f"<div class='product-card-header'><span class='product-kicker'>{html.escape(product_type)}</span><h3>{html.escape(product_label)}</h3></div>",
        "<div class='image-grid'>",
    ]
    if score_href:
        card_parts.append(_render_image_panel(image_href=score_href, label=f"{product_label} score map"))
    if category_href:
        card_parts.append(_render_image_panel(image_href=category_href, label=f"{product_label} category map"))
    card_parts.extend(["</div>", "</section>"])
    return "".join(card_parts)


def _resolve_existing_path(path_value: str) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.exists():
        return candidate
    return None


def _city_rankings_for_product(product_row: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranking_path = _resolve_existing_path(str(product_row.get("city_rankings_csv_path", "")))
    if ranking_path is None:
        return [], []
    frame = pd.read_csv(ranking_path)
    if frame.empty:
        return [], []
    best_rows = frame.loc[frame["ranking_group"] == "best"].sort_values("ranking_position").to_dict(orient="records")
    worst_rows = frame.loc[frame["ranking_group"] == "worst"].sort_values("ranking_position").to_dict(orient="records")
    return best_rows, worst_rows


def _render_city_ranking_list(title: str, rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    items = []
    for row in rows:
        category_name = str(row.get("category", "")).strip()
        category_color = _CITY_RANKING_CATEGORY_COLOR_MAP.get(category_name, "#6b7280")
        items.append(
            "<li>"
            f"<span class='city-rank'>{int(row['ranking_position'])}.</span>"
            f"<span class='city-name'>{html.escape(str(row['city']))}</span>"
            "<span class='city-meta'>"
            f"<span class='city-score'>{float(row['score']):.1f}</span>"
            f"<span class='city-bucket' style='--bucket-color: {html.escape(category_color, quote=True)};'>{html.escape(category_name)}</span>"
            "</span>"
            "</li>"
        )
    return (
        "<section class='city-rankings-card'>"
        f"<h3>{html.escape(title)}</h3>"
        "<ol class='city-rankings-list'>"
        f"{''.join(items)}"
        "</ol>"
        "</section>"
    )


def _render_city_rankings(product_row: dict[str, object]) -> str:
    best_rows, worst_rows = _city_rankings_for_product(product_row)
    if not best_rows and not worst_rows:
        return ""
    return (
        "<section class='city-rankings-section'>"
        "<div class='section-heading'>"
        "<h2>City Rankings</h2>"
        "<p>Daily Comfort Index rankings for a curated set of major cities across the contiguous U.S.</p>"
        "</div>"
        "<div class='city-rankings-grid'>"
        f"{_render_city_ranking_list('Most Comfortable', best_rows)}"
        f"{_render_city_ranking_list('Least Comfortable', worst_rows)}"
        "</div>"
        "</section>"
    )


def _render_section(
    *,
    title: str,
    description: str,
    product_rows: list[dict[str, object]],
    from_dir: Path,
) -> str:
    cards = [_render_product_card(row, from_dir=from_dir) for row in product_rows]
    cards = [card for card in cards if card]
    if not cards:
        return ""
    return (
        "<section class='gallery-section'>"
        f"<div class='section-heading'><h2>{html.escape(title)}</h2><p>{html.escape(description)}</p></div>"
        f"{''.join(cards)}"
        "</section>"
    )


def _parse_valid_date(value: object) -> date | None:
    try:
        if not value:
            return None
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _archive_day_entries(
    run_payloads: list[tuple[Path, dict[str, object]]],
    *,
    from_dir: Path,
) -> list[dict[str, object]]:
    dated_entries: list[tuple[date, Path]] = []
    for run_index_path, payload in run_payloads:
        valid_date = _parse_valid_date(payload.get("valid_date"))
        if valid_date is None:
            continue
        dated_entries.append((valid_date, run_index_path.with_suffix(".html")))

    latest_entries = sorted(dated_entries, key=lambda item: item[0], reverse=True)[:7]
    entries: list[dict[str, object]] = []
    for valid_date, html_path in sorted(latest_entries, key=lambda item: item[0]):
        entries.append(
            {
                "valid_date": valid_date,
                "path": _relative_link(str(html_path), from_dir=from_dir),
            }
        )
    return entries


def _render_day_selector(*, entries: list[dict[str, object]], selected_path: str) -> str:
    if not entries:
        return ""
    options: list[str] = []
    for index, entry in enumerate(entries[:7], start=1):
        valid_date = entry["valid_date"]
        path = str(entry["path"])
        selected_attr = " selected" if path == selected_path else ""
        label = f"Day {index} | {valid_date:%a %m/%d/%y}"
        options.append(f"<option value='{html.escape(path, quote=True)}'{selected_attr}>{html.escape(label)}</option>")
    return (
        "<div class='day-selector-wrap'>"
        "<label for='day-selector'>Forecast day</label>"
        "<select id='day-selector' onchange=\"if(this.value){window.location=this.value;}\">"
        f"{''.join(options)}"
        "</select>"
        "</div>"
    )


def _build_gallery_styles() -> str:
    return """
    <style>
      :root {
        color-scheme: light;
        --page-bg: #f4f0e8;
        --panel-bg: #fffdf9;
        --text: #1e262d;
        --muted: #5d676f;
        --line: #d9d2c4;
        --shadow: 0 16px 36px rgba(42, 48, 58, 0.08);
        --radius: 18px;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        background:
          radial-gradient(circle at top, rgba(255,255,255,0.92), rgba(244,240,232,0.98) 45%),
          linear-gradient(180deg, #f1ede4 0%, #f7f3eb 100%);
        color: var(--text);
      }
      main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 32px 22px 56px;
      }
      .hero, .gallery-section, .run-card {
        background: var(--panel-bg);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
      }
      .hero {
        padding: 28px 28px 24px;
        margin-bottom: 28px;
      }
      .eyebrow {
        margin: 0 0 10px;
        font: 700 0.8rem/1.2 "Segoe UI", Arial, sans-serif;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
      }
      h1, h2, h3 {
        margin: 0;
        font-weight: 700;
        line-height: 1.08;
      }
      h1 { font-size: clamp(2rem, 4.2vw, 3.5rem); }
      h2 { font-size: clamp(1.4rem, 2.6vw, 2.15rem); }
      h3 { font-size: clamp(1.15rem, 2vw, 1.45rem); }
      .subtitle, .meta, .section-heading p, .run-card p, figcaption, .archive-note {
        font-family: "Segoe UI", Arial, sans-serif;
      }
      .subtitle {
        margin: 10px 0 0;
        font-size: 1.05rem;
        color: var(--muted);
      }
      .meta {
        margin: 18px 0 0;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        color: var(--muted);
        font-size: 0.95rem;
      }
      .meta span {
        background: #f3eee4;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 8px 12px;
      }
      .featured-grid, .image-grid {
        display: grid;
        gap: 18px;
      }
      .top-bar {
        margin-top: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
      }
      .day-selector-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: "Segoe UI", Arial, sans-serif;
        color: var(--muted);
      }
      .day-selector-wrap label {
        font-size: 0.92rem;
        font-weight: 600;
      }
      .day-selector-wrap select {
        min-width: 180px;
        padding: 9px 12px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: #fff;
        color: var(--text);
        font: 500 0.95rem/1.2 "Segoe UI", Arial, sans-serif;
      }
      .featured-grid {
        margin-top: 22px;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      }
      .gallery-section {
        margin-top: 24px;
        padding: 22px;
      }
      .city-rankings-section {
        margin-top: 24px;
        background: var(--panel-bg);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 22px;
      }
      .city-rankings-grid {
        display: grid;
        gap: 18px;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        margin-top: 18px;
      }
      .city-rankings-card {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        background: #fff;
      }
      .city-rankings-card h3 {
        font-size: 1.1rem;
      }
      .city-rankings-list {
        margin: 14px 0 0;
        padding: 0;
        list-style: none;
        display: grid;
        gap: 10px;
      }
      .city-rankings-list li {
        display: grid;
        grid-template-columns: 28px minmax(0, 1fr) auto;
        gap: 10px;
        align-items: baseline;
        font-family: "Segoe UI", Arial, sans-serif;
      }
      .city-rank {
        color: var(--muted);
        font-weight: 700;
      }
      .city-name {
        font-weight: 600;
        min-width: 0;
      }
      .city-meta {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--muted);
        white-space: nowrap;
      }
      .city-score {
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        color: #18212f;
      }
      .city-bucket {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: #fff;
        background: var(--bucket-color, #6b7280);
      }
      .section-heading {
        margin-bottom: 18px;
      }
      .section-heading p, .archive-note {
        margin: 8px 0 0;
        color: var(--muted);
      }
      .product-card {
        border-top: 1px solid var(--line);
        padding-top: 18px;
        margin-top: 18px;
      }
      .product-card:first-of-type {
        border-top: 0;
        padding-top: 0;
        margin-top: 0;
      }
      .product-card-header {
        margin-bottom: 12px;
      }
      .product-kicker {
        display: inline-block;
        margin-bottom: 6px;
        font: 700 0.76rem/1.2 "Segoe UI", Arial, sans-serif;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .image-grid {
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      }
      .image-panel {
        margin: 0;
      }
      .image-panel img {
        display: block;
        width: 100%;
        height: auto;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: #edf1f4;
      }
      figcaption {
        margin-top: 8px;
        font-size: 0.92rem;
        color: var(--muted);
      }
      .runs-grid {
        display: grid;
        gap: 18px;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        margin-top: 20px;
      }
      .run-card {
        padding: 18px;
      }
      .run-card a {
        color: inherit;
        text-decoration: none;
      }
      .run-card h3 {
        margin-bottom: 8px;
      }
      .run-card .thumb {
        margin-top: 12px;
      }
      .run-card .thumb img {
        width: 100%;
        border-radius: 12px;
        border: 1px solid var(--line);
        display: block;
      }
      @media (max-width: 720px) {
        main { padding: 18px 14px 38px; }
        .hero, .gallery-section, .run-card { padding-left: 16px; padding-right: 16px; }
      }
    </style>
    """


def _build_pilot_day_gallery_html(
    *,
    valid_date: date,
    source_name: str,
    presentation_theme: str,
    publish_preset_name: str,
    product_rows: list[dict[str, object]],
    run_timestamp: datetime,
    output_dir: Path,
    archive_day_entries: list[dict[str, object]] | None = None,
) -> str:
    featured = _featured_product_row(product_rows)

    hero_images = ""
    if featured:
        featured_label = _humanize_product_name(str(featured.get("product_name", "")))
        score_href = _relative_link(str(featured.get("presentation_score_map_path", "")), from_dir=output_dir)
        category_href = _relative_link(str(featured.get("presentation_category_map_path", "")), from_dir=output_dir)
        hero_images = (
            "<div class='featured-grid'>"
            f"{_render_image_panel(image_href=score_href, label=f'{featured_label} score map') if score_href else ''}"
            f"{_render_image_panel(image_href=category_href, label=f'{featured_label} category map') if category_href else ''}"
            "</div>"
        )
    city_rankings_markup = _render_city_rankings(featured) if featured else ""

    selected_path = f"comfortwx_pilot_day_{source_name}_{valid_date:%Y%m%d}_index.html"
    day_selector = _render_day_selector(entries=archive_day_entries or [], selected_path=selected_path)
    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Comfort Index Daily Maps</title>",
        _build_gallery_styles(),
        "</head><body><main>",
        "<section class='hero'>",
        "<p class='eyebrow'>Comfort Index</p>",
        f"<h1>Daily Maps for {valid_date:%B %d, %Y}</h1>",
        "<p class='subtitle'>Public-facing stitched CONUS score and category maps for this forecast day.</p>",
        "<div class='top-bar'>",
        "<div class='meta'>"
        f"<span>Source: {html.escape(source_name)}</span>"
        f"<span>Updated: {run_timestamp:%Y-%m-%d %H:%M:%S}</span>"
        "</div>",
        day_selector,
        "</div>",
        hero_images,
        city_rankings_markup,
        "</section>",
        "</main></body></html>",
    ]
    return "".join(body)


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

    day_entries_from_root = _archive_day_entries(run_payloads, from_dir=archive_root)
    html_path.write_text(
        _build_archive_gallery_html(
            archive_root=archive_root,
            run_payloads=run_payloads,
            day_entries=day_entries_from_root,
        ),
        encoding="utf-8",
    )

    # Rewrite per-day public pages with the same day selector used by the archive landing page.
    for run_index_path, payload in run_payloads:
        valid_date = _parse_valid_date(payload.get("valid_date"))
        if valid_date is None:
            continue
        run_dir = run_index_path.parent
        run_html_path = run_index_path.with_suffix(".html")
        day_entries_from_run = _archive_day_entries(run_payloads, from_dir=run_dir)
        run_html_path.write_text(
            _build_pilot_day_gallery_html(
                valid_date=valid_date,
                source_name=str(payload.get("source", "")),
                presentation_theme=str(payload.get("presentation_theme", "")),
                publish_preset_name=str(payload.get("publish_preset", "")),
                product_rows=list(payload.get("products", [])),
                run_timestamp=datetime.fromisoformat(str(payload.get("run_timestamp", datetime.now().isoformat()))),
                output_dir=run_dir,
                archive_day_entries=day_entries_from_run,
            ),
            encoding="utf-8",
        )
    return csv_path, json_path, html_path


def _build_archive_gallery_html(
    *,
    archive_root: Path,
    run_payloads: list[tuple[Path, dict[str, object]]],
    day_entries: list[dict[str, object]],
) -> str:
    payload_by_date = {
        valid_date: payload
        for _, payload in run_payloads
        if (valid_date := _parse_valid_date(payload.get("valid_date"))) is not None
    }
    latest_payload = payload_by_date.get(day_entries[0]["valid_date"]) if day_entries else None
    selected_path = str(day_entries[0]["path"]) if day_entries else ""
    day_selector = _render_day_selector(entries=day_entries, selected_path=selected_path)

    latest_markup = ""
    if latest_payload:
        latest_featured = _featured_product_row(latest_payload.get("products", []))
        if latest_featured:
            latest_score = _relative_link(str(latest_featured.get("presentation_score_map_path", "")), from_dir=archive_root)
            latest_category = _relative_link(str(latest_featured.get("presentation_category_map_path", "")), from_dir=archive_root)
            latest_markup = (
                "<div class='featured-grid'>"
                f"{_render_image_panel(image_href=latest_score, label='Latest stitched score map') if latest_score else ''}"
                f"{_render_image_panel(image_href=latest_category, label='Latest stitched category map') if latest_category else ''}"
                "</div>"
            )
    city_rankings_markup = ""
    if latest_payload:
        latest_featured = _featured_product_row(latest_payload.get("products", []))
        if latest_featured:
            city_rankings_markup = _render_city_rankings(latest_featured)

    verification_markup = ""
    verification_index = archive_root / "verification" / "index.html"
    if verification_index.exists():
        verification_markup = (
            "<div class='meta'>"
            "<span><a href='verification/index.html' style='text-decoration:none;color:inherit;'>Verification dashboard</a></span>"
            "</div>"
            "<p class='archive-note'>Verification charts, benchmark stats, and case error maps are available in the verification dashboard.</p>"
        )

    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Comfort Index Daily Maps</title>",
        _build_gallery_styles(),
        "</head><body><main>",
        "<section class='hero'>",
        "<p class='eyebrow'>Comfort Index</p>",
        "<h1>Daily Outdoor Comfort Maps</h1>",
        "<p class='subtitle'>Select a forecast day to view the stitched CONUS score and category maps.</p>",
        "<div class='top-bar'>",
        verification_markup,
        day_selector,
        "</div>",
        latest_markup,
        city_rankings_markup,
        "<p class='archive-note'>Supporting CSV, JSON, and NetCDF files remain in the archive, but this public view focuses on the stitched CONUS products.</p>",
        "</section>",
        "</main></body></html>",
    ]
    return "".join(body)
