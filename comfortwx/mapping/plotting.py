"""Static map rendering for comfort outputs."""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch
from matplotlib.patches import PathPatch
import numpy as np
import xarray as xr

from comfortwx.config import (
    CATEGORIES,
    MAP_SUBTITLE_TEMPLATE,
    MAP_TITLE_TEMPLATE,
    PLOT_STYLE,
    PRESENTATION_CATEGORY_COLORS,
    PRESENTATION_LOW_END_BORDERLINE,
    PRESENTATION_THEME_PRESETS,
    PRESENTATION_NOTE,
    PRESENTATION_PLOT_STYLE,
    PRESENTATION_RAW_SCORE_COLORS,
    PRESENTATION_RENDERING,
    PRODUCT_METADATA,
    RAW_SCORE_COLORS,
    STITCHED_CONUS_PRESENTATION,
    STITCHED_CONUS_STATE_FILE,
)
from comfortwx.mapping.smoothing import smooth_field
from comfortwx.scoring.categories import category_colors, category_labels

try:  # pragma: no cover
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:  # pragma: no cover
    HAS_CARTOPY = False


def _continuous_score_cmap(colors: tuple[str, ...]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("comfortwx_score", colors)


def resolve_presentation_theme(theme_name: str = "default") -> dict[str, object]:
    """Return presentation colors and style for the requested theme."""

    normalized_name = theme_name.strip().lower()
    if normalized_name not in PRESENTATION_THEME_PRESETS:
        raise ValueError(
            f"Unknown presentation theme '{theme_name}'. Available themes: {', '.join(sorted(PRESENTATION_THEME_PRESETS))}."
        )
    theme = PRESENTATION_THEME_PRESETS[normalized_name]
    return {
        "name": normalized_name,
        "raw_score_colors": tuple(theme["raw_score_colors"]),
        "raw_score_levels": tuple(theme.get("raw_score_levels", ())),
        "category_colors": tuple(theme["category_colors"]),
        "plot_style": dict(theme["plot_style"]),
    }


def _product_metadata(overrides: dict[str, str] | None = None) -> dict[str, str]:
    metadata = dict(PRODUCT_METADATA)
    if overrides:
        metadata.update({key: value for key, value in overrides.items() if value})
    return metadata


def _presentation_footer_note(*, low_end_borderline: bool = False) -> str:
    note = PRESENTATION_NOTE
    if low_end_borderline and bool(PRESENTATION_LOW_END_BORDERLINE["enabled"]):
        lower_bound = float(PRESENTATION_LOW_END_BORDERLINE["lower_bound"])
        upper_bound = float(PRESENTATION_LOW_END_BORDERLINE["upper_bound"])
        note += f" Borderline {lower_bound:.0f}-{upper_bound:.0f} shading near the Poor/Fair boundary is presentation-only."
    return note


def _display_valid_date(valid_date: date) -> str:
    return valid_date.strftime("%b %d, %Y")


def _display_timezone_now() -> datetime:
    workflow_timestamp = os.environ.get("COMFORTWX_RUN_TIMESTAMP_UTC", "").strip()
    if workflow_timestamp:
        normalized = workflow_timestamp.replace("Z", "+00:00")
        try:
            run_time = datetime.fromisoformat(normalized)
        except ValueError:
            run_time = datetime.now(ZoneInfo("UTC"))
    else:
        run_time = datetime.now(ZoneInfo("UTC"))
    if run_time.tzinfo is None:
        run_time = run_time.replace(tzinfo=ZoneInfo("UTC"))
    return run_time.astimezone(ZoneInfo("America/New_York"))


def _display_run_time() -> str:
    run_time = _display_timezone_now()
    return run_time.strftime("%b %d, %Y %I:%M %p %Z")


def _display_public_run_header() -> str:
    run_time = _display_timezone_now()
    hour = run_time.hour % 12 or 12
    minute_text = f":{run_time.minute:02d}" if run_time.minute else ""
    meridiem = "am" if run_time.hour < 12 else "pm"
    return f"Run: {run_time.month}/{run_time.day}/{run_time:%y} {hour}{minute_text}{meridiem} ET"


def _display_public_valid_header(valid_date: date) -> str:
    return f"Valid {valid_date:%A} {valid_date.month}/{valid_date.day}/{valid_date:%y}"


def _add_stitched_credit(
    fig,
    ax,
    *,
    presentation_style: dict[str, str | float | int] | None = None,
    cbar=None,
) -> None:
    style = presentation_style or PRESENTATION_PLOT_STYLE
    credit_footer = str(STITCHED_CONUS_PRESENTATION.get("credit_footer", "")).strip()
    if not credit_footer:
        return
    axes_box = ax.get_position()
    if cbar is not None:
        colorbar_box = cbar.ax.get_position()
        credit_y = colorbar_box.y1 + max((axes_box.y0 - colorbar_box.y1) * 0.55, 0.012)
        fig.text(
            axes_box.x1 - 0.006,
            credit_y,
            credit_footer,
            ha="right",
            va="center",
            fontsize=float(style["source_line_size"]),
            fontweight="bold",
            color=str(style["title_color"]),
        )
        return
    credit_y = max(axes_box.y0 - 0.022, 0.018)
    fig.text(
        axes_box.x1 - 0.006,
        credit_y,
        credit_footer,
        ha="right",
        va="center",
        fontsize=float(style["source_line_size"]),
        fontweight="bold",
        color=str(style["title_color"]),
    )


def _stitched_conus_extent() -> tuple[float, float, float, float]:
    return (
        float(STITCHED_CONUS_PRESENTATION["extent_lon_min"]),
        float(STITCHED_CONUS_PRESENTATION["extent_lon_max"]),
        float(STITCHED_CONUS_PRESENTATION["extent_lat_min"]),
        float(STITCHED_CONUS_PRESENTATION["extent_lat_max"]),
    )


def _use_projected_stitched_fallback(stitched_conus: bool) -> bool:
    return stitched_conus and (not HAS_CARTOPY) and STITCHED_CONUS_STATE_FILE.exists()


def _lambert_conformal_project(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    lon_array = np.asarray(lon, dtype=float)
    lat_array = np.asarray(lat, dtype=float)
    lat_array = np.clip(lat_array, -89.999, 89.999)

    phi = np.deg2rad(lat_array)
    lam = np.deg2rad(lon_array)
    phi1 = math.radians(33.0)
    phi2 = math.radians(45.0)
    phi0 = math.radians(39.0)
    lam0 = math.radians(-96.0)

    if abs(phi1 - phi2) < 1e-9:
        n = math.sin(phi1)
    else:
        n = math.log(math.cos(phi1) / math.cos(phi2)) / math.log(
            math.tan(math.pi / 4.0 + phi2 / 2.0) / math.tan(math.pi / 4.0 + phi1 / 2.0)
        )
    f_value = math.cos(phi1) * (math.tan(math.pi / 4.0 + phi1 / 2.0) ** n) / n
    rho = f_value / (np.tan(np.pi / 4.0 + phi / 2.0) ** n)
    rho0 = f_value / (math.tan(math.pi / 4.0 + phi0 / 2.0) ** n)
    theta = n * (lam - lam0)
    x_value = rho * np.sin(theta)
    y_value = rho0 - rho * np.cos(theta)
    return x_value, y_value


def _projected_stitched_extent() -> tuple[float, float, float, float]:
    lon_min, lon_max, lat_min, lat_max = _stitched_conus_extent()
    west_lons = np.full(80, lon_min)
    east_lons = np.full(80, lon_max)
    south_lats = np.full(120, lat_min)
    north_lats = np.full(120, lat_max)
    lon_samples = np.concatenate(
        [
            np.linspace(lon_min, lon_max, 120),
            np.linspace(lon_min, lon_max, 120),
            west_lons,
            east_lons,
        ]
    )
    lat_samples = np.concatenate(
        [
            south_lats,
            north_lats,
            np.linspace(lat_min, lat_max, 80),
            np.linspace(lat_min, lat_max, 80),
        ]
    )
    x_value, y_value = _lambert_conformal_project(lon_samples, lat_samples)
    x_pad = 0.02 * float(np.nanmax(x_value) - np.nanmin(x_value))
    y_pad = 0.025 * float(np.nanmax(y_value) - np.nanmin(y_value))
    return (
        float(np.nanmin(x_value) - x_pad),
        float(np.nanmax(x_value) + x_pad),
        float(np.nanmin(y_value) - y_pad),
        float(np.nanmax(y_value) + y_pad),
    )


def _load_stitched_state_geometries() -> list[list[list[tuple[float, float]]]]:
    if not STITCHED_CONUS_STATE_FILE.exists():
        return []
    feature_collection = json.loads(STITCHED_CONUS_STATE_FILE.read_text(encoding="utf-8"))
    extent = _stitched_conus_extent()
    geometries: list[list[list[tuple[float, float]]]] = []
    for feature in feature_collection.get("features", []):
        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])
        polygons = coords if geom_type == "MultiPolygon" else [coords] if geom_type == "Polygon" else []
        for polygon in polygons:
            if not polygon:
                continue
            lon_values = [point[0] for ring in polygon for point in ring]
            lat_values = [point[1] for ring in polygon for point in ring]
            if not lon_values or not lat_values:
                continue
            if max(lon_values) < extent[0] or min(lon_values) > extent[1] or max(lat_values) < extent[2] or min(lat_values) > extent[3]:
                continue
            geometries.append(
                [
                    [(float(point[0]), float(point[1])) for point in ring]
                    for ring in polygon
                    if len(ring) >= 3
                ]
            )
    return geometries


def _polygon_path_from_rings(rings: list[list[tuple[float, float]]]) -> MplPath | None:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in rings:
        if len(ring) < 3:
            continue
        lon_values = np.array([point[0] for point in ring], dtype=float)
        lat_values = np.array([point[1] for point in ring], dtype=float)
        x_value, y_value = _lambert_conformal_project(lon_values, lat_values)
        ring_vertices = list(zip(x_value.tolist(), y_value.tolist(), strict=False))
        if ring_vertices[0] != ring_vertices[-1]:
            ring_vertices.append(ring_vertices[0])
        vertices.extend(ring_vertices)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring_vertices) - 2) + [MplPath.CLOSEPOLY])
    if not vertices:
        return None
    return MplPath(vertices, codes)


def _lonlat_polygon_path_from_rings(rings: list[list[tuple[float, float]]]) -> MplPath | None:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in rings:
        if len(ring) < 3:
            continue
        ring_vertices = [(float(point[0]), float(point[1])) for point in ring]
        if ring_vertices[0] != ring_vertices[-1]:
            ring_vertices.append(ring_vertices[0])
        vertices.extend(ring_vertices)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring_vertices) - 2) + [MplPath.CLOSEPOLY])
    if not vertices:
        return None
    return MplPath(vertices, codes)


def _draw_projected_stitched_basemap(
    ax,
    *,
    line_only: bool,
    presentation_style: dict[str, str | float | int] | None = None,
) -> None:
    geometries = _load_stitched_state_geometries()
    if not geometries:
        return
    style = presentation_style or PRESENTATION_PLOT_STYLE
    for rings in geometries:
        polygon_path = _polygon_path_from_rings(rings)
        if polygon_path is None:
            continue
        if not line_only:
            ax.add_patch(
                PathPatch(
                    polygon_path,
                    facecolor=str(STITCHED_CONUS_PRESENTATION["land_color"]),
                    edgecolor="none",
                    lw=0.0,
                    zorder=0.4,
                )
            )
        ax.add_patch(
            PathPatch(
                polygon_path,
                facecolor="none",
                edgecolor=str(STITCHED_CONUS_PRESENTATION.get("state_line_color", style["state_color"])),
                lw=float(STITCHED_CONUS_PRESENTATION["state_linewidth"]) * (1.35 if line_only else 1.0),
                alpha=0.95 if line_only else 0.85,
                zorder=5.5 if line_only else 1.1,
            )
        )


def _stitched_land_clip_patch(ax) -> PathPatch | None:
    geometries = _load_stitched_state_geometries()
    compound_path: MplPath | None = None
    use_cartopy_transform = HAS_CARTOPY and hasattr(ax, "projection")
    for rings in geometries:
        polygon_path = _lonlat_polygon_path_from_rings(rings) if use_cartopy_transform else _polygon_path_from_rings(rings)
        if polygon_path is None:
            continue
        compound_path = polygon_path if compound_path is None else MplPath.make_compound_path(compound_path, polygon_path)
    if compound_path is None:
        return None
    transform = ccrs.PlateCarree()._as_mpl_transform(ax) if use_cartopy_transform else ax.transData
    return PathPatch(compound_path, transform=transform)


def _apply_stitched_land_clip(ax, artist) -> None:
    clip_patch = _stitched_land_clip_patch(ax)
    if clip_patch is None:
        return
    if hasattr(artist, "collections"):
        for collection in artist.collections:
            collection.set_clip_path(clip_patch)
        return
    artist.set_clip_path(clip_patch)


def _add_score_key(cbar, *, presentation_style: dict[str, str | float | int] | None = None) -> None:
    style = presentation_style or PRESENTATION_PLOT_STYLE
    label_color = str(style.get("subtitle_color", style.get("border_color", "#505860")))
    fontsize = float(style.get("stitched_score_key_font_size", 8.4))
    cbar.ax.text(
        0.0,
        float(style.get("stitched_score_key_y", -1.7)),
        str(STITCHED_CONUS_PRESENTATION.get("score_key_left_label", "Lower comfort")),
        transform=cbar.ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=label_color,
    )
    cbar.ax.text(
        0.5,
        float(style.get("stitched_score_key_y", -1.7)),
        str(STITCHED_CONUS_PRESENTATION.get("score_key_center_label", "Mixed conditions")),
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=label_color,
    )
    cbar.ax.text(
        1.0,
        float(style.get("stitched_score_key_y", -1.7)),
        str(STITCHED_CONUS_PRESENTATION.get("score_key_right_label", "Higher comfort")),
        transform=cbar.ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=label_color,
    )


def _build_category_legend_handles(
    *,
    category_palette: tuple[str, ...] | list[str],
    presentation: bool,
    stitched_conus: bool,
    presentation_style: dict[str, str | float | int] | None = None,
    include_no_coverage: bool = False,
    include_borderline: bool = False,
) -> list[Patch]:
    style = presentation_style or PRESENTATION_PLOT_STYLE
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor=str(style["legend_edgecolor"]) if presentation and stitched_conus else "none",
            linewidth=0.55 if presentation and stitched_conus else 0.0,
            label=label,
        )
        for label, color in zip(category_labels(), category_palette, strict=False)
    ]
    if include_no_coverage and presentation and stitched_conus:
        legend_handles.append(
            Patch(
                facecolor=str(STITCHED_CONUS_PRESENTATION["coverage_fill_color"]),
                edgecolor=str(STITCHED_CONUS_PRESENTATION["coverage_outline_color"]),
                linewidth=0.8,
                label="No coverage",
                alpha=float(STITCHED_CONUS_PRESENTATION["coverage_fill_alpha"]),
            )
        )
    if include_borderline and presentation:
        lower_bound = float(PRESENTATION_LOW_END_BORDERLINE["lower_bound"])
        upper_bound = float(PRESENTATION_LOW_END_BORDERLINE["upper_bound"])
        legend_handles.append(
            Patch(
                facecolor="#f4ead8",
                edgecolor=category_palette[1],
                linestyle="--",
                linewidth=1.0,
                label=f"Borderline {lower_bound:.0f}-{upper_bound:.0f} (display only)",
                alpha=0.9,
            )
        )
    return legend_handles


def _project_field_coords(field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lon_values, lat_values = np.meshgrid(field["lon"].values, field["lat"].values)
    return _lambert_conformal_project(lon_values, lat_values)


def _projected_pcolormesh(ax, field: xr.DataArray, **kwargs):
    x_value, y_value = _project_field_coords(field)
    return ax.pcolormesh(x_value, y_value, field.values, **kwargs)


def _projected_contour(ax, field: xr.DataArray, **kwargs):
    x_value, y_value = _project_field_coords(field)
    return ax.contour(x_value, y_value, field.values, **kwargs)


def _projected_contourf(ax, field: xr.DataArray, **kwargs):
    x_value, y_value = _project_field_coords(field)
    return ax.contourf(x_value, y_value, field.values, **kwargs)


def _setup_axes(
    *,
    presentation: bool = False,
    presentation_style: dict[str, str | float | int] | None = None,
    stitched_conus: bool = False,
):
    if HAS_CARTOPY:
        projection = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=38.0) if presentation else ccrs.PlateCarree()
        style = presentation_style or PRESENTATION_PLOT_STYLE
        width = float(style["figure_width"]) if presentation else 12.0
        height = float(style["figure_height"]) if presentation else 7.0
        fig, ax = plt.subplots(
            figsize=(width, height),
            subplot_kw={"projection": projection},
            constrained_layout=not (presentation and stitched_conus),
        )
    else:
        style = presentation_style or PRESENTATION_PLOT_STYLE
        width = float(style["figure_width"]) if presentation else 12.0
        height = float(style["figure_height"]) if presentation else 7.0
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=not (presentation and stitched_conus))
        if _use_projected_stitched_fallback(stitched_conus and presentation):
            ax.set_aspect("equal", adjustable="box")
    if presentation and stitched_conus:
        fig.subplots_adjust(
            top=float(style.get("stitched_margin_top", 0.9)),
            bottom=float(style.get("stitched_margin_bottom", 0.055)),
            left=float(style.get("stitched_margin_left", 0.02)),
            right=float(style.get("stitched_margin_right", 0.985)),
        )
    return fig, ax


def _decorate_map(
    ax,
    *,
    extent: tuple[float, float, float, float] | None = None,
    presentation: bool = False,
    presentation_style: dict[str, str | float | int] | None = None,
    stitched_conus: bool = False,
) -> None:
    final_extent = _stitched_conus_extent() if stitched_conus else extent or (-125.0, -66.5, 24.0, 50.0)
    style = presentation_style or PRESENTATION_PLOT_STYLE
    if HAS_CARTOPY:
        if presentation:
            if stitched_conus:
                ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=str(STITCHED_CONUS_PRESENTATION["ocean_color"]), edgecolor="none", zorder=0)
                ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=str(STITCHED_CONUS_PRESENTATION["land_color"]), edgecolor="none", zorder=0)
            ax.coastlines(
                resolution="50m",
                color=str(style["coastline_color"]),
                linewidth=float(STITCHED_CONUS_PRESENTATION["coastline_linewidth"]) if stitched_conus else 0.9,
            )
            ax.add_feature(
                cfeature.BORDERS,
                edgecolor=str(style["border_color"]),
                linewidth=float(STITCHED_CONUS_PRESENTATION["border_linewidth"]) if stitched_conus else 0.5,
            )
            ax.add_feature(
                cfeature.STATES.with_scale("50m"),
                edgecolor=str(STITCHED_CONUS_PRESENTATION.get("state_line_color", style["state_color"])) if stitched_conus else str(style["state_color"]),
                linewidth=float(STITCHED_CONUS_PRESENTATION["state_linewidth"]) if stitched_conus else 0.35,
            )
            ax.add_feature(
                cfeature.LAKES.with_scale("50m"),
                edgecolor="none",
                facecolor=str(STITCHED_CONUS_PRESENTATION["lake_color"]) if stitched_conus else "#eef2f1",
                alpha=0.65 if stitched_conus else 0.55,
            )
            # Cartopy's visible frame API differs across versions. Prefer the
            # older outline_patch when available, otherwise fall back to spines.
            if hasattr(ax, "outline_patch") and ax.outline_patch is not None:
                ax.outline_patch.set_edgecolor(str(style["border_color"]))
                ax.outline_patch.set_linewidth(0.7)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor(str(style["border_color"]))
                    spine.set_linewidth(0.7)
        else:
            ax.coastlines(resolution="50m", color=str(PLOT_STYLE["coastline_color"]), linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, edgecolor=str(PLOT_STYLE["border_color"]), linewidth=0.5)
            ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=str(PLOT_STYLE["border_color"]), linewidth=0.35)
        ax.set_extent(final_extent, crs=ccrs.PlateCarree())
    elif _use_projected_stitched_fallback(stitched_conus and presentation):
        projected_extent = _projected_stitched_extent()
        ax.set_xlim(projected_extent[0], projected_extent[1])
        ax.set_ylim(projected_extent[2], projected_extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(str(style["border_color"]))
            spine.set_linewidth(0.8)
    else:
        ax.set_xlim(final_extent[0], final_extent[1])
        ax.set_ylim(final_extent[2], final_extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        grid_color = style["grid_color"] if presentation else PLOT_STYLE["grid_color"]
        ax.grid(color=str(grid_color), linewidth=0.4, alpha=0.7)


def _add_titles(
    fig,
    ax,
    plot_title: str,
    valid_date: date,
    *,
    map_label: str | None = None,
    presentation: bool = False,
    footer_note: str | None = None,
    presentation_style: dict[str, str | float | int] | None = None,
    product_metadata: dict[str, str] | None = None,
    stitched_conus: bool = False,
) -> None:
    metadata = _product_metadata(product_metadata)
    title_root = metadata["product_title"] or MAP_TITLE_TEMPLATE
    subtitle_source_line = metadata["subtitle_source_line"] or MAP_SUBTITLE_TEMPLATE
    title_prefix = title_root if map_label is None else f"{title_root}: {map_label}"
    style = presentation_style or PRESENTATION_PLOT_STYLE
    if presentation:
        ax.set_title("")
        if stitched_conus:
            axes_box = ax.get_position()
            header_x = axes_box.x0
            header_right_x = axes_box.x1
            header_title_y = axes_box.y1 + float(style.get("stitched_header_title_gap", 0.04))
            header_meta_y = axes_box.y1 + float(style.get("stitched_header_meta_gap", 0.015))
            fig.text(
                header_x,
                header_title_y,
                f"{title_root}",
                ha="left",
                va="center",
                fontsize=float(style["title_size"]),
                fontweight="semibold",
                color=str(style["title_color"]),
            )
            product_subtitle = metadata.get("product_subtitle", "").strip() or str(
                STITCHED_CONUS_PRESENTATION["product_subtitle"]
            )
            fig.text(
                header_x,
                header_meta_y,
                product_subtitle,
                ha="left",
                va="center",
                fontsize=float(style["subtitle_top_size"]),
                color=str(style["subtitle_color"]),
            )
            fig.text(
                header_right_x,
                header_meta_y,
                f"{_display_public_run_header()}  |  {_display_public_valid_header(valid_date)}",
                ha="right",
                va="center",
                fontsize=float(style["source_line_size"]),
                color=str(style["subtitle_color"]),
            )
            if plot_title.strip():
                ax.set_title(
                    f"{plot_title}",
                    fontsize=float(style.get("stitched_axes_title_size", style["subtitle_size"])),
                    color=str(style["subtitle_color"]),
                    pad=float(style.get("stitched_axes_title_pad", 14)),
                    loc="left",
                    fontweight="semibold",
                )
        else:
            fig.suptitle(
                f"{title_prefix}",
                y=0.98,
                fontsize=float(style["title_size"]),
                fontweight="semibold",
                color=str(style["title_color"]),
            )
            ax.set_title(
                f"{plot_title} | {subtitle_source_line} | Valid {_display_valid_date(valid_date)}",
                fontsize=float(style["subtitle_size"]),
                color=str(style["subtitle_color"]),
                pad=10,
            )
        footer_bits = [metadata.get("credit_line", "").strip(), metadata.get("branding_footer", "").strip(), (footer_note or PRESENTATION_NOTE).strip()]
        if not stitched_conus:
            footer_text = str(style.get("footer_separator", " | ")).join([bit for bit in footer_bits if bit])
            fig.text(
                0.01,
                0.012,
                footer_text,
                ha="left",
                va="bottom",
                fontsize=float(style["footer_size"]),
                color=str(style["footer_color"]),
            )
        return

    ax.set_title(
        f"{title_prefix} - {plot_title}\n{MAP_SUBTITLE_TEMPLATE} | Valid {valid_date:%Y-%m-%d}",
        fontsize=15,
        color=str(PLOT_STYLE["title_color"]),
        pad=14,
    )


def _display_field(field: xr.DataArray, *, presentation: bool, smooth_sigma: float | None = None) -> xr.DataArray:
    if not presentation:
        return smooth_field(field, sigma=smooth_sigma or 0.75)

    resample_factor = int(max(1, round(float(PRESENTATION_RENDERING["resample_factor"]))))
    display_field = field
    if resample_factor > 1 and field.sizes.get("lat", 0) > 1 and field.sizes.get("lon", 0) > 1:
        target_lat = np.linspace(float(field["lat"].min().values), float(field["lat"].max().values), field.sizes["lat"] * resample_factor)
        target_lon = np.linspace(float(field["lon"].min().values), float(field["lon"].max().values), field.sizes["lon"] * resample_factor)
        display_field = field.interp(lat=target_lat, lon=target_lon, method="linear")
    if smooth_sigma is not None and smooth_sigma > 0:
        display_field = smooth_field(display_field, sigma=smooth_sigma)
    return display_field


def _apply_coverage_outline(
    ax,
    coverage_field: xr.DataArray,
    *,
    presentation_style: dict[str, str | float | int] | None = None,
    projected_stitched_fallback: bool = False,
) -> None:
    if np.all(np.isnan(coverage_field.values)):
        return
    try:
        if HAS_CARTOPY:
            contour = ax.contour(
                coverage_field["lon"],
                coverage_field["lat"],
                coverage_field,
                levels=[0.5],
                colors=[str(STITCHED_CONUS_PRESENTATION["coverage_outline_color"])],
                linewidths=float(STITCHED_CONUS_PRESENTATION["coverage_outline_linewidth"]),
                alpha=float(STITCHED_CONUS_PRESENTATION["coverage_outline_alpha"]),
                transform=ccrs.PlateCarree(),
            )
            _apply_stitched_land_clip(ax, contour)
        elif projected_stitched_fallback:
            contour = _projected_contour(
                ax,
                coverage_field,
                levels=[0.5],
                colors=[str(STITCHED_CONUS_PRESENTATION["coverage_outline_color"])],
                linewidths=float(STITCHED_CONUS_PRESENTATION["coverage_outline_linewidth"]),
                alpha=float(STITCHED_CONUS_PRESENTATION["coverage_outline_alpha"]),
                zorder=5.0,
            )
            _apply_stitched_land_clip(ax, contour)
    except ValueError:
        return


def _apply_coverage_fade(
    ax,
    coverage_field: xr.DataArray,
    *,
    projected_stitched_fallback: bool = False,
) -> None:
    if np.all(np.isnan(coverage_field.values)):
        return
    try:
        if HAS_CARTOPY:
            contourf = ax.contourf(
                coverage_field["lon"],
                coverage_field["lat"],
                coverage_field,
                levels=[-0.01, 0.45],
                colors=[str(STITCHED_CONUS_PRESENTATION["coverage_fill_color"])],
                alpha=float(STITCHED_CONUS_PRESENTATION["coverage_fill_alpha"]),
                transform=ccrs.PlateCarree(),
                zorder=1.4,
            )
            _apply_stitched_land_clip(ax, contourf)
        elif projected_stitched_fallback:
            contourf = _projected_contourf(
                ax,
                coverage_field,
                levels=[-0.01, 0.45],
                colors=[str(STITCHED_CONUS_PRESENTATION["coverage_fill_color"])],
                alpha=float(STITCHED_CONUS_PRESENTATION["coverage_fill_alpha"]),
                zorder=1.4,
            )
            _apply_stitched_land_clip(ax, contourf)
    except ValueError:
        return


def _low_end_borderline_style(*, for_category: bool) -> dict[str, float]:
    mode = str(PRESENTATION_LOW_END_BORDERLINE["mode"]).strip().lower()
    band_alpha = float(PRESENTATION_LOW_END_BORDERLINE["band_alpha"])
    raw_overlay_alpha = float(PRESENTATION_LOW_END_BORDERLINE["raw_overlay_alpha"])
    category_overlay_alpha = float(PRESENTATION_LOW_END_BORDERLINE["category_overlay_alpha"])

    if mode == "category_emphasis":
        return {
            "gradient_alpha": category_overlay_alpha * (0.55 if for_category else 0.0),
            "band_alpha": band_alpha * 1.15,
        }
    if mode == "balanced":
        return {
            "gradient_alpha": category_overlay_alpha if for_category else raw_overlay_alpha,
            "band_alpha": band_alpha,
        }
    return {
        "gradient_alpha": category_overlay_alpha * (1.0 if for_category else 0.65),
        "band_alpha": band_alpha * (0.9 if for_category else 0.7),
    }


def _apply_low_end_borderline_overlay(
    ax,
    raw_display: xr.DataArray,
    *,
    for_category: bool,
    presentation_style: dict[str, str | float | int] | None = None,
    threshold_color: str | None = None,
    projected_stitched_fallback: bool = False,
) -> bool:
    if not bool(PRESENTATION_LOW_END_BORDERLINE["enabled"]):
        return False

    lower_bound = float(PRESENTATION_LOW_END_BORDERLINE["lower_bound"])
    upper_bound = float(PRESENTATION_LOW_END_BORDERLINE["upper_bound"])
    threshold = float(PRESENTATION_LOW_END_BORDERLINE["threshold"])
    style_map = presentation_style or PRESENTATION_PLOT_STYLE
    if float(raw_display.max().values) < lower_bound or float(raw_display.min().values) > upper_bound:
        return False

    style = _low_end_borderline_style(for_category=for_category)
    borderline_field = raw_display.where((raw_display >= lower_bound) & (raw_display <= upper_bound))
    if np.all(np.isnan(borderline_field.values)):
        return False

    gradient_cmap = _continuous_score_cmap((PRESENTATION_CATEGORY_COLORS[0], PRESENTATION_CATEGORY_COLORS[1]))
    kwargs = {"cmap": gradient_cmap, "vmin": lower_bound, "vmax": upper_bound, "shading": "auto", "alpha": style["gradient_alpha"]}
    if HAS_CARTOPY:
        mesh = ax.pcolormesh(
            borderline_field["lon"],
            borderline_field["lat"],
            borderline_field,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
        _apply_stitched_land_clip(ax, mesh)
    elif projected_stitched_fallback:
        mesh = _projected_pcolormesh(ax, borderline_field, **kwargs)
        _apply_stitched_land_clip(ax, mesh)
    else:
        ax.pcolormesh(borderline_field["lon"], borderline_field["lat"], borderline_field, **kwargs)

    if style["band_alpha"] > 0:
        neutral_band = xr.where((raw_display >= lower_bound) & (raw_display <= upper_bound), 1.0, np.nan)
        band_kwargs = {
            "cmap": ListedColormap(["#f4ead8"]),
            "vmin": 0.0,
            "vmax": 1.0,
            "shading": "auto",
            "alpha": style["band_alpha"],
        }
        if HAS_CARTOPY:
            mesh = ax.pcolormesh(
                neutral_band["lon"],
                neutral_band["lat"],
                neutral_band,
                transform=ccrs.PlateCarree(),
                **band_kwargs,
            )
            _apply_stitched_land_clip(ax, mesh)
        elif projected_stitched_fallback:
            mesh = _projected_pcolormesh(ax, neutral_band, **band_kwargs)
            _apply_stitched_land_clip(ax, mesh)
        else:
            ax.pcolormesh(neutral_band["lon"], neutral_band["lat"], neutral_band, **band_kwargs)

    band_levels = [lower_bound, upper_bound]
    contour_kwargs = {
        "levels": band_levels,
        "colors": [str(style_map["border_color"])],
        "linewidths": float(PRESENTATION_LOW_END_BORDERLINE["edge_linewidth"]),
        "linestyles": "solid",
        "alpha": float(PRESENTATION_LOW_END_BORDERLINE["edge_alpha"]),
    }
    threshold_kwargs = {
        "levels": [threshold],
        "colors": [threshold_color or PRESENTATION_CATEGORY_COLORS[1]],
        "linewidths": float(PRESENTATION_LOW_END_BORDERLINE["threshold_linewidth"]),
        "linestyles": "--",
        "alpha": float(PRESENTATION_LOW_END_BORDERLINE["threshold_alpha"]),
    }
    if HAS_CARTOPY:
        contour_kwargs["transform"] = ccrs.PlateCarree()
        threshold_kwargs["transform"] = ccrs.PlateCarree()

    try:
        if HAS_CARTOPY:
            contour = ax.contour(raw_display["lon"], raw_display["lat"], raw_display, **contour_kwargs)
            _apply_stitched_land_clip(ax, contour)
        elif projected_stitched_fallback:
            contour = _projected_contour(ax, raw_display, **contour_kwargs)
            _apply_stitched_land_clip(ax, contour)
        else:
            ax.contour(raw_display["lon"], raw_display["lat"], raw_display, **contour_kwargs)
    except ValueError:
        pass
    try:
        if HAS_CARTOPY:
            contour = ax.contour(raw_display["lon"], raw_display["lat"], raw_display, **threshold_kwargs)
            _apply_stitched_land_clip(ax, contour)
        elif projected_stitched_fallback:
            contour = _projected_contour(ax, raw_display, **threshold_kwargs)
            _apply_stitched_land_clip(ax, contour)
        else:
            ax.contour(raw_display["lon"], raw_display["lat"], raw_display, **threshold_kwargs)
    except ValueError:
        pass
    return True


def plot_raw_score_map(
    daily: xr.Dataset,
    valid_date: date,
    output_path: Path,
    smooth_sigma: float = 0.75,
    extent: tuple[float, float, float, float] | None = None,
    map_label: str | None = None,
    *,
    presentation: bool = False,
    presentation_theme: str = "default",
    product_metadata: dict[str, str] | None = None,
    presentation_canvas: str | None = None,
) -> Path:
    """Render the continuous daily score field."""

    theme = resolve_presentation_theme(presentation_theme) if presentation else None
    presentation_style = theme["plot_style"] if theme else PRESENTATION_PLOT_STYLE
    stitched_conus = presentation and presentation_canvas == "stitched_conus"
    projected_stitched_fallback = _use_projected_stitched_fallback(stitched_conus)
    fig, ax = _setup_axes(presentation=presentation, presentation_style=presentation_style, stitched_conus=stitched_conus)
    figure_face = presentation_style["figure_facecolor"] if presentation else PLOT_STYLE["figure_facecolor"]
    axes_face = (
        STITCHED_CONUS_PRESENTATION["ocean_color"]
        if stitched_conus
        else presentation_style["axes_facecolor"] if presentation else PLOT_STYLE["axes_facecolor"]
    )
    fig.patch.set_facecolor(str(figure_face))
    ax.set_facecolor(str(axes_face))
    if projected_stitched_fallback:
        _draw_projected_stitched_basemap(ax, line_only=False, presentation_style=presentation_style)

    effective_sigma = float(PRESENTATION_RENDERING["raw_score_sigma"]) if presentation else smooth_sigma
    raw_field = _display_field(daily["daily_score"], presentation=presentation, smooth_sigma=effective_sigma)
    color_scale = theme["raw_score_colors"] if theme else PRESENTATION_RAW_SCORE_COLORS if presentation else RAW_SCORE_COLORS
    score_levels = tuple(theme.get("raw_score_levels", ())) if theme else ()
    if presentation and score_levels:
        cmap = ListedColormap(color_scale)
        kwargs = {
            "cmap": cmap,
            "norm": BoundaryNorm(score_levels, cmap.N, clip=True),
            "shading": "auto",
        }
    else:
        kwargs = {"cmap": _continuous_score_cmap(color_scale), "vmin": 0.0, "vmax": 100.0, "shading": "auto"}
    if HAS_CARTOPY:
        mesh = ax.pcolormesh(
            raw_field["lon"],
            raw_field["lat"],
            raw_field,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
        if stitched_conus:
            _apply_stitched_land_clip(ax, mesh)
    elif projected_stitched_fallback:
        mesh = _projected_pcolormesh(ax, raw_field, **kwargs)
        _apply_stitched_land_clip(ax, mesh)
    else:
        mesh = ax.pcolormesh(raw_field["lon"], raw_field["lat"], raw_field, **kwargs)

    borderline_applied = False
    if presentation:
        borderline_applied = _apply_low_end_borderline_overlay(
            ax,
            raw_field,
            for_category=False,
            presentation_style=presentation_style,
            threshold_color=str(theme["category_colors"][1]) if theme else None,
            projected_stitched_fallback=projected_stitched_fallback,
        )
        if stitched_conus:
            coverage_field = _display_field(
                daily["daily_score"].notnull().astype(float),
                presentation=True,
                smooth_sigma=float(STITCHED_CONUS_PRESENTATION["coverage_mask_sigma"]),
            )
            _apply_coverage_fade(ax, coverage_field, projected_stitched_fallback=projected_stitched_fallback)
            _apply_coverage_outline(ax, coverage_field, presentation_style=presentation_style, projected_stitched_fallback=projected_stitched_fallback)
            if projected_stitched_fallback:
                _draw_projected_stitched_basemap(ax, line_only=True, presentation_style=presentation_style)

    _decorate_map(ax, extent=extent, presentation=presentation, presentation_style=presentation_style, stitched_conus=stitched_conus)
    footer_note = (
        _presentation_footer_note(low_end_borderline=borderline_applied)
        + (" " + str(STITCHED_CONUS_PRESENTATION["footer_suffix"]) if stitched_conus else "")
    )
    if presentation and stitched_conus:
        colorbar_ticks = list(score_levels) if score_levels else list(np.arange(0, 101, 20))
        cbar = fig.colorbar(
            mesh,
            ax=ax,
            orientation="horizontal",
            pad=float(presentation_style.get("stitched_colorbar_pad", 0.035)),
            shrink=float(presentation_style.get("stitched_colorbar_shrink", 0.78)),
            fraction=float(presentation_style.get("stitched_colorbar_fraction", 0.05)),
            aspect=float(presentation_style.get("stitched_colorbar_aspect", 40)),
            ticks=colorbar_ticks,
            boundaries=score_levels if score_levels else None,
            spacing="proportional" if score_levels else "uniform",
        )
        cbar.set_label(str(presentation_style.get("stitched_colorbar_label", "Comfort Index score")))
        _add_score_key(cbar, presentation_style=presentation_style)
        score_legend = ax.legend(
            handles=_build_category_legend_handles(
                category_palette=theme["category_colors"] if theme else PRESENTATION_CATEGORY_COLORS,
                presentation=presentation,
                stitched_conus=stitched_conus,
                presentation_style=presentation_style,
                include_no_coverage=True,
            ),
            title=str(presentation_style.get("stitched_legend_title", "Categories")),
            loc="lower left",
            bbox_to_anchor=(
                float(presentation_style.get("stitched_legend_bbox_x", 0.012)),
                float(presentation_style.get("stitched_legend_bbox_y", 0.018)),
            ),
            ncol=int(presentation_style.get("stitched_legend_ncol", 2)),
            labelspacing=float(presentation_style["legend_labelspacing"]),
            borderpad=float(presentation_style["legend_borderpad"]),
            handlelength=float(presentation_style["legend_handlelength"]),
            handleheight=float(presentation_style.get("legend_handleheight", 0.9)),
            handletextpad=float(presentation_style["legend_handletextpad"]),
            columnspacing=float(presentation_style["legend_columnspacing"]),
            fontsize=float(presentation_style["legend_font_size"]),
            borderaxespad=0.0,
        )
        score_legend.get_frame().set_facecolor(str(presentation_style["legend_facecolor"]))
        score_legend.get_frame().set_edgecolor(str(presentation_style["legend_edgecolor"]))
        score_legend.get_frame().set_alpha(float(presentation_style["legend_frame_alpha"]))
        score_legend.get_title().set_fontsize(float(presentation_style["legend_title_size"]))
        score_legend.get_title().set_fontweight("semibold")
        for text in score_legend.get_texts():
            text.set_color(str(presentation_style["title_color"]))
        _add_titles(
            fig,
            ax,
            "",
            valid_date,
            map_label=map_label,
            presentation=presentation,
            footer_note=footer_note,
            presentation_style=presentation_style,
            product_metadata=product_metadata,
            stitched_conus=stitched_conus,
        )
        _add_stitched_credit(fig, ax, presentation_style=presentation_style, cbar=cbar)
    else:
        cbar = fig.colorbar(
            mesh,
            ax=ax,
            pad=0.02,
            shrink=0.86,
            boundaries=score_levels if score_levels else None,
            spacing="proportional" if score_levels else "uniform",
        )
        cbar.set_label("Score (0-100)")
        _add_titles(
            fig,
            ax,
            "",
            valid_date,
            map_label=map_label,
            presentation=presentation,
            footer_note=footer_note,
            presentation_style=presentation_style,
            product_metadata=product_metadata,
            stitched_conus=stitched_conus,
        )
    if presentation:
        cbar.outline.set_edgecolor(str(presentation_style["border_color"]))
        cbar.ax.tick_params(labelsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs = {
        "dpi": int(presentation_style["dpi"]) if presentation else 180,
        "bbox_inches": "tight",
    }
    if presentation:
        savefig_kwargs["metadata"] = {"Description": PRESENTATION_NOTE}
    fig.savefig(output_path, **savefig_kwargs)
    plt.close(fig)
    return output_path


def plot_category_map(
    daily: xr.Dataset,
    valid_date: date,
    output_path: Path,
    extent: tuple[float, float, float, float] | None = None,
    map_label: str | None = None,
    *,
    presentation: bool = False,
    presentation_theme: str = "default",
    product_metadata: dict[str, str] | None = None,
    presentation_canvas: str | None = None,
) -> Path:
    """Render the discrete daily category field."""

    theme = resolve_presentation_theme(presentation_theme) if presentation else None
    presentation_style = theme["plot_style"] if theme else PRESENTATION_PLOT_STYLE
    stitched_conus = presentation and presentation_canvas == "stitched_conus"
    projected_stitched_fallback = _use_projected_stitched_fallback(stitched_conus)
    fig, ax = _setup_axes(presentation=presentation, presentation_style=presentation_style, stitched_conus=stitched_conus)
    figure_face = presentation_style["figure_facecolor"] if presentation else PLOT_STYLE["figure_facecolor"]
    axes_face = (
        STITCHED_CONUS_PRESENTATION["ocean_color"]
        if stitched_conus
        else presentation_style["axes_facecolor"] if presentation else PLOT_STYLE["axes_facecolor"]
    )
    fig.patch.set_facecolor(str(figure_face))
    ax.set_facecolor(str(axes_face))
    if projected_stitched_fallback:
        _draw_projected_stitched_basemap(ax, line_only=False, presentation_style=presentation_style)

    category_palette = theme["category_colors"] if theme else PRESENTATION_CATEGORY_COLORS if presentation else tuple(category_colors())
    cmap = ListedColormap(category_palette)
    boundaries = np.arange(len(CATEGORIES) + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)

    if presentation and bool(PRESENTATION_RENDERING["category_use_smoothed_score"]):
        raw_display = _display_field(daily["daily_score"], presentation=True, smooth_sigma=None)
        category_display = xr.full_like(raw_display, fill_value=np.nan, dtype=float)
        for index, category in enumerate(CATEGORIES):
            mask = (raw_display >= category.lower) & (raw_display <= category.upper)
            category_display = xr.where(mask, float(index), category_display)
    else:
        category_display = daily["category_index"]
        raw_display = daily["daily_score"]

    kwargs = {"cmap": cmap, "norm": norm, "shading": "auto"}
    if HAS_CARTOPY:
        mesh = ax.pcolormesh(
            category_display["lon"],
            category_display["lat"],
            category_display,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
        if stitched_conus:
            _apply_stitched_land_clip(ax, mesh)
    elif projected_stitched_fallback:
        mesh = _projected_pcolormesh(ax, category_display, **kwargs)
        _apply_stitched_land_clip(ax, mesh)
    else:
        mesh = ax.pcolormesh(category_display["lon"], category_display["lat"], category_display, **kwargs)

    borderline_applied = False
    if presentation:
        borderline_applied = _apply_low_end_borderline_overlay(
            ax,
            raw_display,
            for_category=True,
            presentation_style=presentation_style,
            threshold_color=str(category_palette[1]),
            projected_stitched_fallback=projected_stitched_fallback,
        )
        if stitched_conus:
            coverage_field = _display_field(
                daily["daily_score"].notnull().astype(float),
                presentation=True,
                smooth_sigma=float(STITCHED_CONUS_PRESENTATION["coverage_mask_sigma"]),
            )
            _apply_coverage_fade(ax, coverage_field, projected_stitched_fallback=projected_stitched_fallback)
            _apply_coverage_outline(ax, coverage_field, presentation_style=presentation_style, projected_stitched_fallback=projected_stitched_fallback)
            if projected_stitched_fallback:
                _draw_projected_stitched_basemap(ax, line_only=True, presentation_style=presentation_style)

    _decorate_map(ax, extent=extent, presentation=presentation, presentation_style=presentation_style, stitched_conus=stitched_conus)
    _add_titles(
        fig,
        ax,
        "",
        valid_date,
        map_label=map_label,
        presentation=presentation,
        footer_note=(
            _presentation_footer_note(low_end_borderline=borderline_applied)
            + (" " + str(STITCHED_CONUS_PRESENTATION["footer_suffix"]) if stitched_conus else "")
        ),
        presentation_style=presentation_style,
        product_metadata=product_metadata,
        stitched_conus=stitched_conus,
    )

    legend_handles = _build_category_legend_handles(
        category_palette=category_palette,
        presentation=presentation,
        stitched_conus=stitched_conus,
        presentation_style=presentation_style,
        include_no_coverage=True,
        include_borderline=borderline_applied,
    )
    legend = ax.legend(
        handles=legend_handles,
        title=str(presentation_style.get("stitched_legend_title", "Categories")) if presentation and stitched_conus else "Categories",
        loc="lower left" if presentation and stitched_conus else "lower left",
        bbox_to_anchor=(
            float(presentation_style.get("stitched_legend_bbox_x", 1.0)),
            float(presentation_style.get("stitched_legend_bbox_y", 0.0)),
        )
        if presentation and stitched_conus
        else None,
        frameon=True,
        ncol=int(presentation_style.get("stitched_legend_ncol", 2)) if presentation and stitched_conus else 1,
        labelspacing=float(presentation_style["legend_labelspacing"]) if presentation else 0.5,
        borderpad=float(presentation_style["legend_borderpad"]) if presentation else 0.4,
        handlelength=float(presentation_style["legend_handlelength"]) if presentation else 1.3,
        handleheight=float(presentation_style.get("legend_handleheight", 0.9)) if presentation else 0.9,
        handletextpad=float(presentation_style["legend_handletextpad"]) if presentation else 0.5,
        columnspacing=float(presentation_style["legend_columnspacing"]) if presentation else 0.8,
        fontsize=float(presentation_style["legend_font_size"]) if presentation else None,
        borderaxespad=0.0 if presentation and stitched_conus else 0.5,
    )
    if presentation:
        legend.get_frame().set_facecolor(str(presentation_style["legend_facecolor"]))
        legend.get_frame().set_edgecolor(str(presentation_style["legend_edgecolor"]))
        legend.get_frame().set_alpha(float(presentation_style["legend_frame_alpha"]) if stitched_conus else 0.92)
        legend.get_title().set_fontsize(float(presentation_style["legend_title_size"]))
        legend.get_title().set_fontweight("semibold")
        for text in legend.get_texts():
            text.set_color(str(presentation_style["title_color"] if stitched_conus else presentation_style["subtitle_color"]))
        if stitched_conus:
            _add_stitched_credit(fig, ax, presentation_style=presentation_style)
    if not (presentation and stitched_conus):
        fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.86, ticks=np.arange(len(CATEGORIES)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs = {
        "dpi": int(presentation_style["dpi"]) if presentation else 180,
        "bbox_inches": "tight",
    }
    if presentation:
        savefig_kwargs["metadata"] = {"Description": PRESENTATION_NOTE}
    fig.savefig(output_path, **savefig_kwargs)
    plt.close(fig)
    return output_path


def render_daily_maps(
    daily: xr.Dataset,
    valid_date: date,
    output_dir: Path,
    file_prefix: str = "comfortwx_daily",
    extent: tuple[float, float, float, float] | None = None,
    map_label: str | None = None,
    *,
    include_presentation: bool = False,
    presentation_theme: str = "default",
    product_metadata: dict[str, str] | None = None,
    presentation_canvas: str | None = None,
) -> dict[str, Path]:
    """Render debug maps and optional presentation maps."""

    raw_path = output_dir / f"{file_prefix}_score_{valid_date:%Y%m%d}.png"
    category_path = output_dir / f"{file_prefix}_category_{valid_date:%Y%m%d}.png"
    outputs = {
        "raw_map": plot_raw_score_map(daily=daily, valid_date=valid_date, output_path=raw_path, extent=extent, map_label=map_label),
        "category_map": plot_category_map(daily=daily, valid_date=valid_date, output_path=category_path, extent=extent, map_label=map_label),
    }
    if include_presentation and bool(PRESENTATION_RENDERING["enabled"]):
        presentation_raw_path = output_dir / f"{file_prefix}_presentation_score_{valid_date:%Y%m%d}.png"
        presentation_category_path = output_dir / f"{file_prefix}_presentation_category_{valid_date:%Y%m%d}.png"
        outputs["presentation_raw_map"] = plot_raw_score_map(
            daily=daily,
            valid_date=valid_date,
            output_path=presentation_raw_path,
            extent=extent,
            map_label=map_label,
            presentation=True,
            presentation_theme=presentation_theme,
            product_metadata=product_metadata,
            presentation_canvas=presentation_canvas,
        )
        outputs["presentation_category_map"] = plot_category_map(
            daily=daily,
            valid_date=valid_date,
            output_path=presentation_category_path,
            extent=extent,
            map_label=map_label,
            presentation=True,
            presentation_theme=presentation_theme,
            product_metadata=product_metadata,
            presentation_canvas=presentation_canvas,
        )
    return outputs
