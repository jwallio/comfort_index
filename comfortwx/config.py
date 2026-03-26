"""Shared configuration for the comfort scoring engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "output"
STITCHED_CONUS_STATE_FILE: Final[Path] = PROJECT_ROOT / "comfortwx" / "mapping" / "data" / "us_states.geojson"


@dataclass(frozen=True)
class CategoryDefinition:
    """A score category and its display styling."""

    name: str
    lower: float
    upper: float
    color: str


@dataclass(frozen=True)
class CloudRegimeDefinition:
    """Configuration for temperature-aware cloud comfort scoring."""

    name: str
    temp_min: float | None
    temp_max: float | None
    ideal_min: float
    ideal_max: float
    max_score: float
    low_penalty_per_10pct: float
    high_penalty_per_10pct: float
    overcast_threshold: float
    overcast_penalty: float
    elite_cap_threshold: float
    elite_cap_score: float
    hot_sun_threshold: float | None = None
    hot_sun_penalty: float = 0.0


HOURLY_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "temp_f",
    "dewpoint_f",
    "wind_mph",
    "gust_mph",
    "cloud_pct",
    "pop_pct",
    "qpf_in",
)

OPTIONAL_FIELDS: Final[tuple[str, ...]] = (
    "thunder",
    "aqi",
    "pm25",
    "smoke",
    "visibility_mi",
    "cape",
    "weather_code",
    "precip_type",
)

DEFAULT_LAT_POINTS: Final[int] = 65
DEFAULT_LON_POINTS: Final[int] = 115
DEFAULT_DOMAIN: Final[dict[str, float]] = {
    "lat_min": 24.0,
    "lat_max": 50.0,
    "lon_min": -125.0,
    "lon_max": -66.5,
}

LOCAL_DAY_HOURS: Final[tuple[int, int]] = (8, 20)
PRIME_DAY_HOURS: Final[tuple[int, int]] = (11, 18)

DAYTIME_HOUR_WEIGHTS: Final[dict[int, float]] = {
    8: 0.45,
    9: 0.65,
    10: 0.9,
    11: 1.05,
    12: 1.15,
    13: 1.2,
    14: 1.2,
    15: 1.15,
    16: 1.05,
    17: 0.95,
    18: 0.8,
    19: 0.65,
    20: 0.45,
}

ROLLING_WINDOWS_HOURS: Final[tuple[int, int]] = (3, 6)
RELIABILITY_THRESHOLD: Final[float] = 65.0
RELIABILITY_HIGH_THRESHOLD: Final[float] = 75.0

RELIABILITY_COMPONENT_WEIGHTS: Final[dict[str, float]] = {
    "usable_hours": 0.5,
    "strong_hours": 0.25,
    "prime_clean_hours": 0.25,
}

DAILY_SCORE_WEIGHTS: Final[dict[str, float]] = {
    "best_3hr": 0.22,
    "best_6hr": 0.28,
    "daytime_weighted_mean": 0.32,
    "reliability_score": 0.18,
}

DAILY_AGGREGATION_DEFAULT_MODE: Final[str] = "baseline"
DAILY_AGGREGATION_MODES: Final[dict[str, dict[str, object]]] = {
    "baseline": {
        "graded_reliability": False,
        "usable_score_min": 65.0,
        "usable_score_full": 65.0,
        "strong_score_min": 75.0,
        "strong_score_full": 75.0,
        "prime_clean_penalty_weights": {"rain": 0.45, "thunder": 0.35, "gust": 0.2},
        "soft_rain_signal": False,
        "soft_gust_signal": False,
        "soft_score_crash_signal": False,
        "soft_score_drop_signal": False,
        "measurable_rain_pop_min": 55.0,
        "measurable_rain_pop_full": 55.0,
        "measurable_rain_qpf_min": 0.01,
        "measurable_rain_qpf_full": 0.01,
        "gust_soft_min": 26.0,
        "gust_soft_full": 26.0,
        "score_crash_floor": 45.0,
        "score_crash_ceiling": 45.0,
        "score_drop_min": 17.0,
        "score_drop_full": 17.0,
        "disruption_weights": {
            "measurable_rain": 14.0,
            "heavy_precip": 8.0,
            "thunder": 26.0,
            "strong_gusts": 10.0,
            "score_crash": 8.0,
            "score_drop": 12.0,
        },
    },
    "soft_reliability": {
        "graded_reliability": True,
        "usable_score_min": 58.0,
        "usable_score_full": 72.0,
        "strong_score_min": 70.0,
        "strong_score_full": 84.0,
        "prime_clean_penalty_weights": {"rain": 0.35, "thunder": 0.35, "gust": 0.15},
        "soft_rain_signal": True,
        "soft_gust_signal": True,
        "soft_score_crash_signal": True,
        "soft_score_drop_signal": True,
        "measurable_rain_pop_min": 45.0,
        "measurable_rain_pop_full": 75.0,
        "measurable_rain_qpf_min": 0.005,
        "measurable_rain_qpf_full": 0.035,
        "gust_soft_min": 24.0,
        "gust_soft_full": 38.0,
        "score_crash_floor": 32.0,
        "score_crash_ceiling": 48.0,
        "score_drop_min": 12.0,
        "score_drop_full": 24.0,
        "disruption_weights": {
            "measurable_rain": 11.0,
            "heavy_precip": 7.0,
            "thunder": 24.0,
            "strong_gusts": 7.0,
            "score_crash": 5.5,
            "score_drop": 8.0,
        },
    },
    "balanced_soft": {
        "graded_reliability": True,
        "usable_score_min": 60.0,
        "usable_score_full": 74.0,
        "strong_score_min": 71.0,
        "strong_score_full": 85.0,
        "prime_clean_penalty_weights": {"rain": 0.32, "thunder": 0.34, "gust": 0.14},
        "soft_rain_signal": True,
        "soft_gust_signal": True,
        "soft_score_crash_signal": True,
        "soft_score_drop_signal": True,
        "measurable_rain_pop_min": 42.0,
        "measurable_rain_pop_full": 72.0,
        "measurable_rain_qpf_min": 0.005,
        "measurable_rain_qpf_full": 0.03,
        "gust_soft_min": 24.0,
        "gust_soft_full": 36.0,
        "score_crash_floor": 34.0,
        "score_crash_ceiling": 49.0,
        "score_drop_min": 11.0,
        "score_drop_full": 22.0,
        "disruption_weights": {
            "measurable_rain": 10.0,
            "heavy_precip": 6.5,
            "thunder": 23.0,
            "strong_gusts": 6.5,
            "score_crash": 5.0,
            "score_drop": 7.0,
        },
    },
    "long_lead_soft": {
        "graded_reliability": True,
        "usable_score_min": 57.0,
        "usable_score_full": 76.0,
        "strong_score_min": 68.0,
        "strong_score_full": 86.0,
        "prime_clean_penalty_weights": {"rain": 0.28, "thunder": 0.33, "gust": 0.1},
        "soft_rain_signal": True,
        "soft_gust_signal": True,
        "soft_score_crash_signal": True,
        "soft_score_drop_signal": True,
        "measurable_rain_pop_min": 38.0,
        "measurable_rain_pop_full": 68.0,
        "measurable_rain_qpf_min": 0.004,
        "measurable_rain_qpf_full": 0.028,
        "gust_soft_min": 22.0,
        "gust_soft_full": 35.0,
        "score_crash_floor": 30.0,
        "score_crash_ceiling": 50.0,
        "score_drop_min": 9.0,
        "score_drop_full": 20.0,
        "disruption_weights": {
            "measurable_rain": 8.5,
            "heavy_precip": 6.0,
            "thunder": 21.0,
            "strong_gusts": 5.5,
            "score_crash": 4.0,
            "score_drop": 5.5,
        },
    },
}

DAILY_DISRUPTION_WEIGHTS: Final[dict[str, float]] = {
    "measurable_rain": 14.0,
    "heavy_precip": 8.0,
    "thunder": 26.0,
    "strong_gusts": 10.0,
    "score_crash": 8.0,
    "score_drop": 12.0,
}

MAX_DAILY_DISRUPTION_PENALTY: Final[float] = 40.0

PRIME_INTERRUPTION_THRESHOLDS: Final[dict[str, float]] = {
    "qpf_in": 0.01,
    "pop_pct": 55.0,
    "heavy_qpf_in": 0.05,
    "gust_mph": 26.0,
    "score_crash": 45.0,
    "score_drop": 17.0,
}

PRIME_RECOVERY_SETTINGS: Final[dict[str, float]] = {
    "tail_hours": 3.0,
    "tail_clean_credit": 4.0,
}

TEMP_SCORE_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (68.0, 78.0, 35.0),
    (62.0, 67.0, 30.0),
    (79.0, 82.0, 30.0),
    (56.0, 61.0, 22.0),
    (83.0, 86.0, 22.0),
    (50.0, 55.0, 12.0),
    (87.0, 90.0, 12.0),
    (44.0, 49.0, 5.0),
    (91.0, 94.0, 5.0),
)

DEWPOINT_SCORE_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (None, 34.0, 18.0),
    (35.0, 41.0, 19.0),
    (42.0, 55.0, 20.0),
    (56.0, 60.0, 16.0),
    (61.0, 64.0, 11.0),
    (65.0, 67.0, 6.0),
    (68.0, 70.0, 3.0),
    (71.0, 74.0, 1.0),
)

TEMP_DEWPOINT_PENALTIES: Final[tuple[tuple[float, float, float], ...]] = (
    (78.0, 65.0, -4.0),
    (82.0, 67.0, -7.0),
    (86.0, 70.0, -12.0),
    (90.0, 72.0, -18.0),
    (94.0, 74.0, -24.0),
)

TEMP_WIND_ADJUSTMENTS: Final[
    tuple[tuple[float | None, float | None, float | None, float | None, float | None, float], ...]
] = (
    (84.0, None, None, 6.0, 14.0, 3.0),
    (88.0, None, 58.0, 8.0, 16.0, 4.0),
    (None, 52.0, None, 15.0, None, -8.0),
    (None, 45.0, None, 12.0, None, -10.0),
)

TEMP_CLOUD_ADJUSTMENTS: Final[tuple[tuple[float | None, float | None, float | None, float | None, float], ...]] = (
    (None, 58.0, 80.0, None, -5.0),
    (70.0, 82.0, 15.0, 45.0, 2.0),
    (88.0, None, 20.0, 50.0, 3.0),
    (90.0, None, None, 10.0, -4.0),
)

WIND_SCORE_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (3.0, 8.0, 10.0),
    (0.0, 2.0, 8.0),
    (9.0, 12.0, 8.0),
    (13.0, 16.0, 6.0),
    (17.0, 20.0, 3.0),
    (21.0, 25.0, 1.0),
)

CLOUD_SCORE_COOL_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (0.0, 20.0, 10.0),
    (21.0, 40.0, 8.0),
    (41.0, 70.0, 5.0),
    (71.0, None, 2.0),
)

CLOUD_SCORE_MILD_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (10.0, 40.0, 10.0),
    (0.0, 9.0, 9.0),
    (41.0, 60.0, 8.0),
    (61.0, 80.0, 5.0),
    (81.0, None, 2.0),
)

CLOUD_SCORE_WARM_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (20.0, 50.0, 10.0),
    (0.0, 19.0, 7.0),
    (51.0, 70.0, 8.0),
    (71.0, None, 4.0),
)

CLOUD_REGIMES: Final[tuple[CloudRegimeDefinition, ...]] = (
    CloudRegimeDefinition(
        name="cool",
        temp_min=None,
        temp_max=59.0,
        ideal_min=5.0,
        ideal_max=25.0,
        max_score=10.0,
        low_penalty_per_10pct=0.55,
        high_penalty_per_10pct=0.95,
        overcast_threshold=75.0,
        overcast_penalty=1.5,
        elite_cap_threshold=85.0,
        elite_cap_score=6.0,
    ),
    CloudRegimeDefinition(
        name="mild",
        temp_min=60.0,
        temp_max=82.0,
        ideal_min=15.0,
        ideal_max=42.0,
        max_score=10.0,
        low_penalty_per_10pct=0.35,
        high_penalty_per_10pct=0.7,
        overcast_threshold=82.0,
        overcast_penalty=1.2,
        elite_cap_threshold=88.0,
        elite_cap_score=6.8,
    ),
    CloudRegimeDefinition(
        name="warm_hot",
        temp_min=83.0,
        temp_max=None,
        ideal_min=22.0,
        ideal_max=52.0,
        max_score=10.0,
        low_penalty_per_10pct=0.75,
        high_penalty_per_10pct=0.5,
        overcast_threshold=78.0,
        overcast_penalty=0.8,
        elite_cap_threshold=88.0,
        elite_cap_score=7.5,
        hot_sun_threshold=12.0,
        hot_sun_penalty=2.2,
    ),
)

POP_SCORE_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (0.0, 14.999, 15.0),
    (15.0, 30.0, 12.0),
    (31.0, 50.0, 8.0),
    (51.0, 70.0, 4.0),
)

THUNDER_HOURLY_CAP: Final[float] = 60.0

AQI_PENALTY_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (None, 50.0, 0.0),
    (51.0, 80.0, 2.0),
    (81.0, 100.0, 5.0),
    (101.0, 150.0, 10.0),
    (151.0, None, 18.0),
)

AQI_CAP_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (None, 80.0, 100.0),
    (81.0, 100.0, 92.0),
    (101.0, 150.0, 80.0),
    (151.0, None, 65.0),
)

PM25_PENALTY_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (None, 12.0, 0.0),
    (12.1, 25.0, 2.0),
    (25.1, 35.0, 5.0),
    (35.1, 55.0, 10.0),
    (55.1, None, 18.0),
)

PM25_CAP_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (None, 25.0, 100.0),
    (25.1, 35.0, 92.0),
    (35.1, 55.0, 80.0),
    (55.1, None, 65.0),
)

SMOKE_PENALTY_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (None, 0.1, 0.0),
    (0.11, 0.35, 3.0),
    (0.36, 0.6, 7.0),
    (0.61, None, 14.0),
)

SMOKE_CAP_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (None, 0.35, 100.0),
    (0.36, 0.6, 88.0),
    (0.61, None, 72.0),
)

VISIBILITY_PENALTY_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (10.0, None, 0.0),
    (7.0, 9.99, 2.0),
    (5.0, 6.99, 5.0),
    (3.0, 4.99, 10.0),
    (None, 2.99, 18.0),
)

VISIBILITY_CAP_BINS: Final[tuple[tuple[float | None, float | None, float], ...]] = (
    (10.0, None, 100.0),
    (7.0, 9.99, 95.0),
    (5.0, 6.99, 88.0),
    (3.0, 4.99, 75.0),
    (None, 2.99, 60.0),
)

PRISTINE_GATE_THRESHOLDS: Final[dict[str, float]] = {
    "raw_score_min": 90.0,
    "best_6hr_min": 90.0,
    "daytime_weighted_mean_min": 84.0,
    "reliability_score_min": 85.0,
    "daytime_mean_dewpoint_max": 60.0,
    "daytime_mean_gust_max": 18.0,
    "best_6hr_daytime_gap_max": 2.5,
    "prime_score_drop_fraction_max": 0.05,
}


def get_daily_aggregation_mode_config(mode: str = DAILY_AGGREGATION_DEFAULT_MODE) -> dict[str, object]:
    """Return the configured daily aggregation mode settings."""

    normalized_mode = mode.strip().lower()
    if normalized_mode not in DAILY_AGGREGATION_MODES:
        raise ValueError(
            f"Unknown daily aggregation mode '{mode}'. Available modes: {', '.join(sorted(DAILY_AGGREGATION_MODES))}."
        )
    return DAILY_AGGREGATION_MODES[normalized_mode]

CATEGORIES: Final[tuple[CategoryDefinition, ...]] = (
    CategoryDefinition("Poor", 0.0, 44.0, "#8A817C"),
    CategoryDefinition("Fair", 45.0, 59.0, "#D8B365"),
    CategoryDefinition("Pleasant", 60.0, 74.0, "#7FBF7B"),
    CategoryDefinition("Ideal", 75.0, 89.0, "#4DB6AC"),
    CategoryDefinition("Exceptional", 90.0, 100.0, "#4F86C6"),
)

RAW_SCORE_COLORS: Final[tuple[str, ...]] = (
    "#7f8b94",
    "#94c37a",
    "#59b2a8",
    "#f0c14c",
    "#ef7f52",
)

PLOT_STYLE: Final[dict[str, str | float]] = {
    "figure_facecolor": "#f5f5ef",
    "axes_facecolor": "#fbfaf5",
    "grid_color": "#d6d4cb",
    "title_color": "#222222",
    "coastline_color": "#4a4a45",
    "border_color": "#7f7d73",
}

MAP_TITLE_TEMPLATE: Final[str] = "Comfort Index"
MAP_SUBTITLE_TEMPLATE: Final[str] = "Daily outdoor pleasantness score and category"
PRODUCT_METADATA: Final[dict[str, str]] = {
    "product_title": "Comfort Index",
    "subtitle_source_line": "Open-Meteo regional blend",
    "credit_line": "Weather Projects Lab",
    "branding_footer": "Comfort Index Map Generator",
}

PRESENTATION_RAW_SCORE_COLORS: Final[tuple[str, ...]] = (
    "#6f7b84",
    "#8fb874",
    "#59aa9e",
    "#d9be5c",
    "#da7c52",
)

PRESENTATION_CATEGORY_COLORS: Final[tuple[str, ...]] = (
    "#8A817C",
    "#D8B365",
    "#7FBF7B",
    "#4DB6AC",
    "#4F86C6",
)

PRESENTATION_PLOT_STYLE: Final[dict[str, str | float | int]] = {
    "figure_facecolor": "#f7f3eb",
    "axes_facecolor": "#f9f6ef",
    "title_color": "#1f2327",
    "subtitle_color": "#4d545b",
    "border_color": "#6f726c",
    "coastline_color": "#3f423f",
    "state_color": "#8a8d88",
    "grid_color": "#d9d2c7",
    "legend_facecolor": "#f5efe4",
    "legend_edgecolor": "#c9c0b1",
    "footer_color": "#666666",
    "title_size": 18,
    "subtitle_top_size": 12,
    "subtitle_size": 11,
    "source_line_size": 10,
    "footer_size": 8,
    "legend_title_size": 10,
    "legend_font_size": 9,
    "legend_labelspacing": 0.55,
    "legend_borderpad": 0.55,
    "legend_handlelength": 1.35,
    "legend_handleheight": 0.9,
    "legend_handletextpad": 0.6,
    "legend_columnspacing": 1.0,
    "legend_frame_alpha": 0.95,
    "footer_separator": " | ",
    "stitched_title_x": 0.012,
    "stitched_title_y": 0.985,
    "stitched_subtitle_y": 0.953,
    "stitched_source_y": 0.928,
    "stitched_credit_x": 0.955,
    "stitched_credit_y": 0.02,
    "stitched_margin_top": 0.865,
    "stitched_margin_bottom": 0.055,
    "stitched_margin_left": 0.02,
    "stitched_margin_right": 0.985,
    "stitched_axes_title_pad": 14,
    "stitched_axes_title_size": 11,
    "stitched_legend_title": "Comfort categories",
    "stitched_legend_bbox_x": 0.012,
    "stitched_legend_bbox_y": 0.018,
    "stitched_legend_ncol": 2,
    "stitched_colorbar_label": "Comfort Index (0-100)",
    "stitched_colorbar_pad": 0.032,
    "stitched_colorbar_shrink": 0.76,
    "stitched_colorbar_fraction": 0.05,
    "stitched_colorbar_aspect": 42,
    "dpi": 220,
    "figure_width": 13.5,
    "figure_height": 8.2,
}

PRESENTATION_THEME_PRESETS: Final[dict[str, dict[str, object]]] = {
    "default": {
        "raw_score_colors": PRESENTATION_RAW_SCORE_COLORS,
        "category_colors": PRESENTATION_CATEGORY_COLORS,
        "plot_style": PRESENTATION_PLOT_STYLE,
    },
    "shareable": {
        "raw_score_colors": (
            "#857C88",
            "#977A88",
            "#BE8B7C",
            "#D99E62",
            "#E4C35B",
            "#AAD15E",
            "#67C97E",
            "#34C29D",
            "#20ACBF",
            "#267FD4",
            "#5A5EE0",
        ),
        "raw_score_levels": (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        "category_colors": PRESENTATION_CATEGORY_COLORS,
        "plot_style": {
            **PRESENTATION_PLOT_STYLE,
            "figure_facecolor": "#ffffff",
            "axes_facecolor": "#ffffff",
            "title_color": "#161a1e",
            "subtitle_color": "#46505a",
            "border_color": "#656b70",
            "coastline_color": "#30353a",
            "state_color": "#8a908f",
            "legend_facecolor": "#ffffff",
            "legend_edgecolor": "#d3d5d8",
            "title_size": 18.8,
            "subtitle_top_size": 10.7,
            "source_line_size": 8.8,
            "stitched_title_y": 0.968,
            "stitched_subtitle_y": 0.944,
            "stitched_source_y": 0.944,
            "stitched_margin_top": 0.938,
            "stitched_margin_bottom": 0.065,
            "stitched_colorbar_pad": 0.026,
        },
    },
    "public": {
        "raw_score_colors": (
            "#6F6863",
            "#8E7E71",
            "#B19A76",
            "#CDB06D",
            "#DCCB73",
            "#B7CC72",
            "#86C37A",
            "#62BE90",
            "#48B6A8",
            "#43A6C6",
            "#4F86C6",
        ),
        "raw_score_levels": (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        "category_colors": PRESENTATION_CATEGORY_COLORS,
        "plot_style": {
            **PRESENTATION_PLOT_STYLE,
            "figure_facecolor": "#f6f2ea",
            "axes_facecolor": "#fbf8f2",
            "title_color": "#15191d",
            "subtitle_color": "#4f5861",
            "border_color": "#71767b",
            "coastline_color": "#4a5761",
            "state_color": "#111111",
            "legend_facecolor": "#f8f4ec",
            "legend_edgecolor": "#c8bfb2",
            "footer_color": "#70757a",
            "title_size": 18.5,
            "subtitle_top_size": 10.8,
            "subtitle_size": 10.25,
            "source_line_size": 8.9,
            "footer_size": 7.2,
            "legend_title_size": 9.5,
            "legend_font_size": 9.2,
            "legend_labelspacing": 0.5,
            "legend_borderpad": 0.7,
            "legend_handlelength": 1.5,
            "legend_handleheight": 1.0,
            "legend_handletextpad": 0.62,
            "legend_columnspacing": 1.2,
            "legend_frame_alpha": 0.97,
            "stitched_margin_top": 0.918,
            "stitched_margin_bottom": 0.065,
            "stitched_margin_left": 0.02,
            "stitched_margin_right": 0.985,
            "stitched_header_title_gap": 0.044,
            "stitched_header_meta_gap": 0.02,
            "stitched_axes_title_pad": 12,
            "stitched_axes_title_size": 10.0,
            "stitched_legend_bbox_x": 0.012,
            "stitched_legend_bbox_y": 0.018,
            "stitched_colorbar_pad": 0.028,
            "stitched_colorbar_shrink": 0.74,
            "stitched_colorbar_aspect": 46,
        },
    },
    "premium_muted": {
        "raw_score_colors": (
            "#726B67",
            "#8E8177",
            "#AE9877",
            "#C7AD74",
            "#D9C873",
            "#B8CA79",
            "#8FC381",
            "#67B894",
            "#4FAFB0",
            "#4A97C4",
            "#4C7DBF",
        ),
        "raw_score_levels": (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        "category_colors": PRESENTATION_CATEGORY_COLORS,
        "plot_style": {
            **PRESENTATION_PLOT_STYLE,
            "figure_facecolor": "#f6f2ea",
            "axes_facecolor": "#fbf8f2",
            "title_color": "#15191d",
            "subtitle_color": "#4f5861",
            "border_color": "#71767b",
            "coastline_color": "#4a5761",
            "state_color": "#111111",
            "legend_facecolor": "#f8f4ec",
            "legend_edgecolor": "#c8bfb2",
            "footer_color": "#70757a",
            "title_size": 20,
            "subtitle_top_size": 11.5,
            "subtitle_size": 10.25,
            "source_line_size": 9.5,
            "footer_size": 7.2,
            "legend_title_size": 9.5,
            "legend_font_size": 9.2,
            "legend_labelspacing": 0.5,
            "legend_borderpad": 0.7,
            "legend_handlelength": 1.5,
            "legend_handleheight": 1.0,
            "legend_handletextpad": 0.62,
            "legend_columnspacing": 1.2,
            "legend_frame_alpha": 0.97,
            "stitched_title_y": 0.978,
            "stitched_subtitle_y": 0.944,
            "stitched_source_y": 0.944,
            "stitched_margin_top": 0.885,
            "stitched_margin_bottom": 0.065,
            "stitched_margin_left": 0.02,
            "stitched_margin_right": 0.985,
            "stitched_axes_title_pad": 12,
            "stitched_axes_title_size": 10.0,
            "stitched_legend_bbox_x": 0.012,
            "stitched_legend_bbox_y": 0.018,
            "stitched_colorbar_pad": 0.028,
            "stitched_colorbar_shrink": 0.74,
            "stitched_colorbar_aspect": 46,
        },
    },
    "bold_social": {
        "raw_score_colors": (
            "#635C69",
            "#8A6A7B",
            "#B07C72",
            "#D59A5D",
            "#E2C45C",
            "#A8D05C",
            "#63C77A",
            "#2FC09E",
            "#1EACBF",
            "#2B8CCF",
            "#5D63D6",
        ),
        "raw_score_levels": (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        "category_colors": PRESENTATION_CATEGORY_COLORS,
        "plot_style": {
            **PRESENTATION_PLOT_STYLE,
            "figure_facecolor": "#f6f2ea",
            "axes_facecolor": "#fbf8f2",
            "title_color": "#15191d",
            "subtitle_color": "#4f5861",
            "border_color": "#71767b",
            "coastline_color": "#4a5761",
            "state_color": "#111111",
            "legend_facecolor": "#f8f4ec",
            "legend_edgecolor": "#c8bfb2",
            "footer_color": "#70757a",
            "title_size": 20,
            "subtitle_top_size": 11.5,
            "subtitle_size": 10.25,
            "source_line_size": 9.5,
            "footer_size": 7.2,
            "legend_title_size": 9.5,
            "legend_font_size": 9.2,
            "legend_labelspacing": 0.5,
            "legend_borderpad": 0.7,
            "legend_handlelength": 1.5,
            "legend_handleheight": 1.0,
            "legend_handletextpad": 0.62,
            "legend_columnspacing": 1.2,
            "legend_frame_alpha": 0.97,
            "stitched_title_y": 0.978,
            "stitched_subtitle_y": 0.944,
            "stitched_source_y": 0.944,
            "stitched_margin_top": 0.885,
            "stitched_margin_bottom": 0.065,
            "stitched_margin_left": 0.02,
            "stitched_margin_right": 0.985,
            "stitched_axes_title_pad": 12,
            "stitched_axes_title_size": 10.0,
            "stitched_legend_bbox_x": 0.012,
            "stitched_legend_bbox_y": 0.018,
            "stitched_colorbar_pad": 0.028,
            "stitched_colorbar_shrink": 0.74,
            "stitched_colorbar_aspect": 46,
        },
    },
    "blue_green_yellow_magenta": {
        "raw_score_colors": (
            "#5F6B7A",
            "#5D7F93",
            "#5B98A5",
            "#59B2A7",
            "#74C08B",
            "#A0CA74",
            "#D0CB6E",
            "#E5B869",
            "#D98F8D",
            "#C46CB3",
            "#9A59C8",
        ),
        "raw_score_levels": (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        "category_colors": PRESENTATION_CATEGORY_COLORS,
        "plot_style": {
            **PRESENTATION_PLOT_STYLE,
            "figure_facecolor": "#f6f2ea",
            "axes_facecolor": "#fbf8f2",
            "title_color": "#15191d",
            "subtitle_color": "#4f5861",
            "border_color": "#71767b",
            "coastline_color": "#4a5761",
            "state_color": "#111111",
            "legend_facecolor": "#f8f4ec",
            "legend_edgecolor": "#c8bfb2",
            "footer_color": "#70757a",
            "title_size": 20,
            "subtitle_top_size": 11.5,
            "subtitle_size": 10.25,
            "source_line_size": 9.5,
            "footer_size": 7.2,
            "legend_title_size": 9.5,
            "legend_font_size": 9.2,
            "legend_labelspacing": 0.5,
            "legend_borderpad": 0.7,
            "legend_handlelength": 1.5,
            "legend_handleheight": 1.0,
            "legend_handletextpad": 0.62,
            "legend_columnspacing": 1.2,
            "legend_frame_alpha": 0.97,
            "stitched_title_y": 0.978,
            "stitched_subtitle_y": 0.944,
            "stitched_source_y": 0.944,
            "stitched_margin_top": 0.885,
            "stitched_margin_bottom": 0.065,
            "stitched_margin_left": 0.02,
            "stitched_margin_right": 0.985,
            "stitched_axes_title_pad": 12,
            "stitched_axes_title_size": 10.0,
            "stitched_legend_bbox_x": 0.012,
            "stitched_legend_bbox_y": 0.018,
            "stitched_colorbar_pad": 0.028,
            "stitched_colorbar_shrink": 0.74,
            "stitched_colorbar_aspect": 46,
        },
    },
}

PUBLISH_PRESETS: Final[dict[str, dict[str, object]]] = {
    "standard": {
        "include_presentation": True,
        "write_bundle_manifest": True,
    }
}
ARCHIVE_SETTINGS: Final[dict[str, str]] = {
    "root_name": "archive",
    "layout": "year/month/day",
    "run_index_base_name": "comfortwx_archive_index",
}
PILOT_DAY_REGIONS: Final[tuple[str, ...]] = (
    "west_coast",
    "southwest",
    "rockies",
    "plains",
    "southeast",
    "northeast",
    "great_lakes",
)
PILOT_DAY_MOSAICS: Final[tuple[tuple[str, ...], ...]] = (
    ("west_coast", "southwest", "rockies"),
    ("southwest", "rockies"),
    ("plains", "great_lakes", "northeast"),
    ("west_coast", "southwest", "rockies", "plains", "southeast", "northeast", "great_lakes"),
)
PILOT_DAY_CACHE_DEFAULT_MODE: Final[str] = "reuse"
PILOT_DAY_CACHE_MODES: Final[tuple[str, ...]] = ("reuse", "refresh")

PRESENTATION_RENDERING: Final[dict[str, float | bool]] = {
    "enabled": True,
    "resample_factor": 3.0,
    "raw_score_sigma": 0.6,
    "category_use_smoothed_score": True,
}

PRESENTATION_LOW_END_BORDERLINE: Final[dict[str, float | bool | str]] = {
    "enabled": True,
    "lower_bound": 40.0,
    "upper_bound": 50.0,
    "threshold": 45.0,
    "mode": "raw_emphasis",
    "band_alpha": 0.12,
    "raw_overlay_alpha": 0.22,
    "category_overlay_alpha": 0.34,
    "edge_alpha": 0.32,
    "threshold_alpha": 0.7,
    "edge_linewidth": 0.7,
    "threshold_linewidth": 1.15,
}

STITCHED_CONUS_PRESENTATION: Final[dict[str, float | bool | str]] = {
    "enabled": True,
    "extent_lon_min": -126.0,
    "extent_lon_max": -66.0,
    "extent_lat_min": 23.0,
    "extent_lat_max": 50.5,
    "land_color": "#fcfcfa",
    "ocean_color": "#d8e4e9",
    "lake_color": "#e8f0f2",
    "coverage_fill_color": "#ffffff",
    "coverage_fill_alpha": 0.18,
    "coverage_outline_color": "#4d565a",
    "coverage_outline_alpha": 0.42,
    "coverage_outline_linewidth": 1.0,
    "coverage_mask_sigma": 0.6,
    "state_linewidth": 0.6,
    "state_line_color": "#111111",
    "border_linewidth": 0.45,
    "coastline_linewidth": 1.0,
    "title": "Comfort Index",
    "product_subtitle": "Daily Outdoor Comfort Across the Contiguous U.S.",
    "subtitle_source_line": "Open-Meteo regional blend",
    "score_key_left_label": "Less comfortable",
    "score_key_center_label": "Transition zone",
    "score_key_right_label": "Most comfortable",
    "credit_footer": "Map by: Jonathan Wall @_jwall on X",
    "footer_suffix": "Display smoothing, coverage fade, and stitched-footprint outline are presentation-only.",
}

PUBLIC_CITY_RANKING_LOCATIONS: Final[tuple[dict[str, float | int | str], ...]] = (
    {"name": "New York, NY", "lat": 40.7128, "lon": -74.0060, "priority": 1},
    {"name": "Los Angeles, CA", "lat": 34.0522, "lon": -118.2437, "priority": 2},
    {"name": "Chicago, IL", "lat": 41.8781, "lon": -87.6298, "priority": 3},
    {"name": "Houston, TX", "lat": 29.7604, "lon": -95.3698, "priority": 4},
    {"name": "Phoenix, AZ", "lat": 33.4484, "lon": -112.0740, "priority": 5},
    {"name": "Philadelphia, PA", "lat": 39.9526, "lon": -75.1652, "priority": 6},
    {"name": "San Antonio, TX", "lat": 29.4241, "lon": -98.4936, "priority": 7},
    {"name": "San Diego, CA", "lat": 32.7157, "lon": -117.1611, "priority": 8},
    {"name": "Dallas, TX", "lat": 32.7767, "lon": -96.7970, "priority": 9},
    {"name": "Austin, TX", "lat": 30.2672, "lon": -97.7431, "priority": 10},
    {"name": "Jacksonville, FL", "lat": 30.3322, "lon": -81.6557, "priority": 11},
    {"name": "Columbus, OH", "lat": 39.9612, "lon": -82.9988, "priority": 12},
    {"name": "Charlotte, NC", "lat": 35.2271, "lon": -80.8431, "priority": 13},
    {"name": "Indianapolis, IN", "lat": 39.7684, "lon": -86.1581, "priority": 14},
    {"name": "Seattle, WA", "lat": 47.6062, "lon": -122.3321, "priority": 15},
    {"name": "Denver, CO", "lat": 39.7392, "lon": -104.9903, "priority": 16},
    {"name": "Washington, DC", "lat": 38.9072, "lon": -77.0369, "priority": 17},
    {"name": "Boston, MA", "lat": 42.3601, "lon": -71.0589, "priority": 18},
    {"name": "Nashville, TN", "lat": 36.1627, "lon": -86.7816, "priority": 19},
    {"name": "Detroit, MI", "lat": 42.3314, "lon": -83.0458, "priority": 20},
    {"name": "Oklahoma City, OK", "lat": 35.4676, "lon": -97.5164, "priority": 21},
    {"name": "Portland, OR", "lat": 45.5152, "lon": -122.6784, "priority": 22},
    {"name": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398, "priority": 23},
    {"name": "Memphis, TN", "lat": 35.1495, "lon": -90.0490, "priority": 24},
    {"name": "Louisville, KY", "lat": 38.2527, "lon": -85.7585, "priority": 25},
    {"name": "Baltimore, MD", "lat": 39.2904, "lon": -76.6122, "priority": 26},
    {"name": "Milwaukee, WI", "lat": 43.0389, "lon": -87.9065, "priority": 27},
    {"name": "Albuquerque, NM", "lat": 35.0844, "lon": -106.6504, "priority": 28},
    {"name": "Tucson, AZ", "lat": 32.2226, "lon": -110.9747, "priority": 29},
    {"name": "Fresno, CA", "lat": 36.7378, "lon": -119.7871, "priority": 30},
    {"name": "Sacramento, CA", "lat": 38.5816, "lon": -121.4944, "priority": 31},
    {"name": "Kansas City, MO", "lat": 39.0997, "lon": -94.5786, "priority": 32},
    {"name": "Atlanta, GA", "lat": 33.7490, "lon": -84.3880, "priority": 33},
    {"name": "Omaha, NE", "lat": 41.2565, "lon": -95.9345, "priority": 34},
    {"name": "Raleigh, NC", "lat": 35.7796, "lon": -78.6382, "priority": 35},
    {"name": "Miami, FL", "lat": 25.7617, "lon": -80.1918, "priority": 36},
    {"name": "New Orleans, LA", "lat": 29.9511, "lon": -90.0715, "priority": 37},
    {"name": "Minneapolis, MN", "lat": 44.9778, "lon": -93.2650, "priority": 38},
    {"name": "Cleveland, OH", "lat": 41.4993, "lon": -81.6944, "priority": 39},
    {"name": "Tampa, FL", "lat": 27.9506, "lon": -82.4572, "priority": 40},
    {"name": "Pittsburgh, PA", "lat": 40.4406, "lon": -79.9959, "priority": 41},
    {"name": "Cincinnati, OH", "lat": 39.1031, "lon": -84.5120, "priority": 42},
    {"name": "St. Louis, MO", "lat": 38.6270, "lon": -90.1994, "priority": 43},
    {"name": "Salt Lake City, UT", "lat": 40.7608, "lon": -111.8910, "priority": 44},
    {"name": "Boise, ID", "lat": 43.6150, "lon": -116.2023, "priority": 45},
    {"name": "Spokane, WA", "lat": 47.6588, "lon": -117.4260, "priority": 46},
    {"name": "El Paso, TX", "lat": 31.7619, "lon": -106.4850, "priority": 47},
    {"name": "Richmond, VA", "lat": 37.5407, "lon": -77.4360, "priority": 48},
    {"name": "Birmingham, AL", "lat": 33.5186, "lon": -86.8104, "priority": 49},
    {"name": "Charleston, SC", "lat": 32.7765, "lon": -79.9311, "priority": 50},
)

PRESENTATION_NOTE: Final[str] = "Presentation map uses display-only interpolation/smoothing; science rasters and diagnostics are unchanged."

MOSAIC_DEFAULT_BLEND_METHOD: Final[str] = "taper"
MOSAIC_DEFAULT_TARGET_GRID_POLICY: Final[str] = "adaptive"
MOSAIC_CATEGORY_THRESHOLD_BUFFER: Final[float] = 2.5
WESTERN_THRESHOLD_PROXIMITY_BINS: Final[tuple[float, ...]] = (1.0, 2.0, 3.0, 5.0)
WESTERN_THRESHOLD_MARGIN_DIAGNOSTIC: Final[float] = 1.5
WESTERN_POOR_FAIR_DISTRIBUTION_RANGE: Final[tuple[float, float]] = (35.0, 55.0)
WESTERN_POOR_FAIR_FOCUS_WINDOWS: Final[tuple[tuple[float, float], ...]] = (
    (40.0, 50.0),
    (42.0, 48.0),
)
WESTERN_POOR_FAIR_BORDERLINE_MARGIN: Final[float] = 2.0
WESTERN_POOR_FAIR_DIAGNOSTIC_THRESHOLDS: Final[tuple[float, ...]] = (
    43.0,
    45.0,
    47.0,
)
WESTERN_MOSAIC_FIXED_TARGET_GRID: Final[dict[str, float]] = {
    "lat_min": 28.5,
    "lat_max": 50.5,
    "lon_min": -120.5,
    "lon_max": -100.0,
    "lat_step": 0.5,
    "lon_step": 0.5,
}

OPENMETEO_FORECAST_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_AIR_QUALITY_URL: Final[str] = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENMETEO_SINGLE_RUN_URL: Final[str] = "https://single-runs-api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_HOURLY_VARS: Final[tuple[str, ...]] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation_probability",
    "precipitation",
    "weather_code",
    "visibility",
    "cape",
)
OPENMETEO_AIR_QUALITY_VARS: Final[tuple[str, ...]] = (
    "us_aqi",
    "pm2_5",
)
OPENMETEO_THUNDER_WEATHER_CODES: Final[tuple[int, ...]] = (95, 96, 99)
OPENMETEO_CAPE_THUNDER_THRESHOLD: Final[float] = 1200.0
OPENMETEO_POP_THUNDER_THRESHOLD: Final[float] = 60.0
OPENMETEO_REGIONAL_BATCH_SIZE: Final[int] = 12
OPENMETEO_DEFAULT_MESH_PROFILE: Final[str] = "standard"
OPENMETEO_REQUEST_TIMEOUT_SECONDS: Final[float] = 45.0
OPENMETEO_REQUEST_MAX_RETRIES: Final[int] = 2
OPENMETEO_REQUEST_RETRY_BACKOFF_INITIAL_SECONDS: Final[float] = 1.5
OPENMETEO_REQUEST_RETRY_BACKOFF_MULTIPLIER: Final[float] = 2.0
OPENMETEO_REQUEST_RETRY_BACKOFF_MAX_SECONDS: Final[float] = 8.0
OPENMETEO_REQUEST_THROTTLE_SECONDS: Final[float] = 0.35
OPENMETEO_REQUEST_RETRYABLE_STATUS_CODES: Final[tuple[int, ...]] = (408, 429, 500, 502, 503, 504)
OPENMETEO_REGIONAL_MESH_SETTINGS: Final[dict[str, dict[str, float | bool]]] = {
    "southeast": {
        # Coarse pilot mesh tuned for humid subtropical and Gulf/Atlantic gradients.
        "lat_step": 3.5,
        "lon_step": 3.5,
        "include_air_quality": False,
        "timezone": "America/New_York",
    },
    "west_coast": {
        # Pacific-side pilot mesh spanning the West Coast corridor and nearby marine influence zones.
        # TODO: refine spacing once native grids can better resolve coastal gradients and terrain breaks.
        "lat_step": 2.5,
        "lon_step": 2.5,
        "include_air_quality": False,
        "timezone": "America/Los_Angeles",
    },
    "southwest": {
        # Coarse pilot mesh tuned for broad desert / high-terrain contrasts.
        # TODO: refine spacing or split terrain-focused subregions once native grids are added.
        "lat_step": 2.75,
        "lon_step": 3.0,
        "include_air_quality": False,
        "timezone": "America/Phoenix",
    },
    "rockies": {
        # Coarse pilot mesh for the terrain/arid seam test with the Southwest.
        # TODO: refine spacing around major elevation transitions once native grids are added.
        "lat_step": 2.5,
        "lon_step": 2.5,
        "include_air_quality": False,
        "timezone": "America/Denver",
    },
    "plains": {
        # Bridge-region pilot mesh spanning the central Plains between the western and eastern pilots.
        # TODO: revisit spacing once native gridded ingestion can better resolve dryline and frontal contrasts.
        "lat_step": 2.5,
        "lon_step": 2.5,
        "include_air_quality": False,
        "timezone": "America/Chicago",
    },
    "great_lakes": {
        # Northern bridge-region pilot mesh linking the Plains and Northeast across the Great Lakes corridor.
        # TODO: refine spacing around lake-breeze and shoreline gradients once native grids are available.
        "lat_step": 2.25,
        "lon_step": 2.25,
        "include_air_quality": False,
        "timezone": "America/Chicago",
    },
    "northeast": {
        # Coarse pilot mesh for the first adjacent seam test with the Southeast.
        # TODO: refine spacing around the Appalachians and coastal corridor once native grids are added.
        "lat_step": 2.5,
        "lon_step": 2.5,
        "include_air_quality": False,
        "timezone": "America/New_York",
    },
}

OPENMETEO_REGIONAL_MESH_PROFILES: Final[dict[str, dict[str, dict[str, float | bool]]]] = {
    "standard": OPENMETEO_REGIONAL_MESH_SETTINGS,
    "fine": {
        "southwest": {
            # Higher-fidelity western pilot mesh to test whether point density reduces seam roughness.
            "lat_step": 1.75,
            "lon_step": 2.0,
            "include_air_quality": False,
            "timezone": "America/Phoenix",
        },
        "rockies": {
            # Higher-fidelity western pilot mesh across major terrain transitions.
            "lat_step": 1.75,
            "lon_step": 1.75,
            "include_air_quality": False,
            "timezone": "America/Denver",
        },
    },
}

OPENMETEO_VERIFICATION_DEFAULT_REGION: Final[str] = "southeast"
OPENMETEO_VERIFICATION_FORECAST_MODEL_DEFAULT: Final[str] = "gfs_seamless"
OPENMETEO_VERIFICATION_FORECAST_SHORT_LEAD_MODEL: Final[str] = "hrrr"
OPENMETEO_VERIFICATION_ANALYSIS_MODEL_DEFAULT: Final[str] = "best_match"
OPENMETEO_VERIFICATION_FORECAST_RUN_HOUR_UTC: Final[int] = 12
OPENMETEO_VERIFICATION_FORECAST_LEAD_DAYS: Final[int] = 1
OPENMETEO_VERIFICATION_BENCHMARK_LEAD_DAYS: Final[tuple[int, ...]] = (1, 2, 3, 7)
OPENMETEO_VERIFICATION_FORECAST_DAYS: Final[int] = 2
OPENMETEO_VERIFICATION_ANALYSIS_POP_PROXY_QPF_FULL_IN: Final[float] = 0.05
OPENMETEO_VERIFICATION_FORECAST_HOURLY_VARS: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "precipitation_probability",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
    "visibility",
)
OPENMETEO_VERIFICATION_ANALYSIS_HOURLY_VARS: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "precipitation_probability",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
    "visibility",
)
OPENMETEO_VERIFICATION_SAMPLE_POINT_NAMES: Final[tuple[str, ...]] = (
    "center",
    "northwest",
    "northeast",
    "southwest",
    "southeast",
)

VERIFICATION_HIGH_COMFORT_CATEGORY_MIN_INDEX: Final[int] = 3
VERIFICATION_BENCHMARK_THRESHOLDS: Final[dict[str, float]] = {
    "score_mae_max": 8.0,
    "near_category_agreement_min": 0.9,
    "abs_score_bias_mean_max": 5.0,
}
VERIFICATION_BENCHMARK_LEAD_THRESHOLDS: Final[dict[int, dict[str, float]]] = {
    1: {
        "score_mae_max": 8.0,
        "near_category_agreement_min": 0.9,
        "abs_score_bias_mean_max": 5.0,
    },
    2: {
        "score_mae_max": 8.75,
        "near_category_agreement_min": 0.88,
        "abs_score_bias_mean_max": 5.5,
    },
    3: {
        "score_mae_max": 9.5,
        "near_category_agreement_min": 0.86,
        "abs_score_bias_mean_max": 6.0,
    },
    7: {
        "score_mae_max": 11.5,
        "near_category_agreement_min": 0.83,
        "abs_score_bias_mean_max": 7.0,
    },
}
VERIFICATION_CALIBRATION_MIN_CASES: Final[int] = 1
VERIFICATION_CALIBRATION_MIN_POINTS: Final[int] = 40
VERIFICATION_CALIBRATION_LINEAR_MIN_CASES: Final[int] = 3
VERIFICATION_CALIBRATION_LINEAR_MIN_POINTS: Final[int] = 200
VERIFICATION_CALIBRATION_SLOPE_RANGE: Final[tuple[float, float]] = (0.75, 1.25)
VERIFICATION_CALIBRATION_INTERCEPT_RANGE: Final[tuple[float, float]] = (-15.0, 15.0)
VERIFICATION_CALIBRATION_BIAS_SHRINKAGE_OFFSET: Final[float] = 2.0
VERIFICATION_AGGREGATION_TUNING_CANDIDATE_MODES: Final[tuple[str, ...]] = (
    "baseline",
    "soft_reliability",
    "balanced_soft",
    "long_lead_soft",
)


def get_openmeteo_mesh_settings(region_name: str, mesh_profile: str = OPENMETEO_DEFAULT_MESH_PROFILE) -> dict[str, float | bool]:
    """Return regional Open-Meteo mesh settings for the requested profile."""

    normalized_profile = mesh_profile.strip().lower()
    if normalized_profile not in OPENMETEO_REGIONAL_MESH_PROFILES:
        raise ValueError(
            f"Unknown Open-Meteo mesh profile '{mesh_profile}'. Available profiles: {', '.join(sorted(OPENMETEO_REGIONAL_MESH_PROFILES))}."
        )

    profile_settings = OPENMETEO_REGIONAL_MESH_PROFILES[normalized_profile]
    if region_name not in profile_settings:
        if normalized_profile == OPENMETEO_DEFAULT_MESH_PROFILE and region_name in OPENMETEO_REGIONAL_MESH_SETTINGS:
            return OPENMETEO_REGIONAL_MESH_SETTINGS[region_name]
        raise ValueError(
            f"Open-Meteo mesh profile '{normalized_profile}' does not support region '{region_name}'."
        )
    return profile_settings[region_name]
