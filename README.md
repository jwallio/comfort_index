# Nice Weather Map Generator

Nice Weather Map Generator is a production-style V1 Python project for scoring hourly outdoor-weather pleasantness and turning that into a daily CONUS map. It is built to be meteorologically interpretable, tunable, and extendable rather than a toy weighted-average script.

The first version ships with a deterministic synthetic forecast loader so the full pipeline runs end to end today. The scoring engine is kept model-agnostic so real forecast sources can replace the mock loader later.

## What It Does

Given an hourly gridded forecast, the project:

- scores each hour and grid cell from 0 to 100
- uses separate sub-scores for temperature, dew point, wind, clouds, and precipitation reliability
- applies interaction adjustments and hazard penalties and caps
- aggregates hourly values to a daily score with daytime weighting, best-window logic, reliability, and disruption penalties
- assigns daily categories: `Poor`, `Fair`, `Pleasant`, `Ideal`, `Exceptional`
- writes a daily output field and two static map images

## Scoring Philosophy

This V1 intentionally uses piecewise meteorological rules instead of a single linear normalized blend.

- Dew point matters strongly in warm weather.
- Rain and thunder during prime daytime hours matter more than overnight precipitation.
- Wind can help on hot dry days and hurt badly on cool/raw days.
- Cloud preferences depend on temperature regime.
- Hazard logic is separate from baseline comfort logic so future severe-weather, smoke, winter precip, or visibility rules can be added cleanly.
- Daily aggregation rewards both a good best window and a reliable daytime stretch.

## Project Layout

```text
nice-weather-map-generator/
  nicewx/
    config.py
    main.py
    data/
      loaders.py
      mock_data.py
    scoring/
      temperature.py
      humidity.py
      wind.py
      clouds.py
      precip.py
      interactions.py
      hazards.py
      hourly.py
      daily.py
      categories.py
    mapping/
      smoothing.py
      plotting.py
    validation/
      demo_cases.py
  tests/
  output/
  README.md
  requirements.txt
```

## Core Method

### Hourly score

The hourly score is composed as:

```text
temp_score
+ dewpoint_score
+ wind_score
+ cloud_score
+ precip_score
+ interaction_adjustments
- hazard_penalties
```

Then it is clamped to `0..100`, with hazard caps applied afterward. Thunder currently imposes both a penalty and an hourly cap near 60.

### Daily score

The daily score uses fixed local proxy daytime hours for V1 (`08-20 local proxy time`):

```text
daily_score =
    0.28 * best_3hr
  + 0.32 * best_6hr
  + 0.25 * daytime_weighted_mean
  + 0.15 * reliability_score
  - disruption_penalty
```

Where:

- `best_3hr` and `best_6hr` reward the best usable outdoor window
- `daytime_weighted_mean` emphasizes midday and afternoon
- `reliability_score` is the weighted daytime fraction of hours with score `>= 65`
- `disruption_penalty` explicitly punishes prime-hour rain, thunder, strong gusts, and badly degraded score hours

## Install

Create an environment and install the core dependencies:

```powershell
pip install -r requirements.txt
```

Optional:

- `cartopy` for coastlines, borders, and state lines on the maps
- if `cartopy` is absent, plotting falls back to a clean lat/lon heatmap

## Run

From the project root:

```powershell
python -m nicewx.main
```

Useful options:

```powershell
python -m nicewx.main --date 2026-05-18 --lat-points 65 --lon-points 115
python -m nicewx.main --output-dir .\output
python -m nicewx.main --inspect-lat 35.8 --inspect-lon -78.6
python -m nicewx.main --region southeast
python -m nicewx.main --source openmeteo --region southwest --mesh-profile fine --date 2026-03-24
python -m nicewx.main --source openmeteo --region southwest --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

### Real Point Mode

The first real-data path uses Open-Meteo point forecasts and writes point diagnostics instead of a map:

```powershell
python -m nicewx.main --source openmeteo --lat 35.78 --lon -78.64 --date 2026-03-24
```

This fetches hourly point data, scores it, writes hourly and daily CSV diagnostics to `output/`, and prints a short explanation string.

For a quick calibration pass on the fixed demo cases:

```powershell
python -m nicewx.validation.calibration --date 2026-05-18
```

For a multi-city real-world validation run using the Open-Meteo point path:

```powershell
python -m nicewx.validation.real_world --date 2026-03-24
```

## Example Outputs

The default run writes files like:

- `output/nicewx_daily_fields_20260518.nc`
- `output/nicewx_daily_score_20260518.png`
- `output/nicewx_daily_category_20260518.png`
- `output/nicewx_demo_cases_20260518.csv`
- `output/nicewx_demo_case_hourly_20260518.csv`
- `output/nicewx_point_<lat>_<lon>_<date>_hourly.csv`
- `output/nicewx_point_<lat>_<lon>_<date>_summary.csv`

Console output includes a grid summary and demo-case sanity checks.

## Debug Vs Presentation Maps

The project now keeps two map families for regional and mosaic products:

- debug/science maps: the original direct-grid PNGs used for inspection and calibration
- presentation maps: a second, more polished rendering path for publication-style visuals

This separation is intentional:

- the science outputs keep the original `daily_score` and `category_index` rasters visible with minimal styling changes
- the presentation outputs improve projection, typography, borders, legend treatment, and color styling
- display-only interpolation and light smoothing can be applied for presentation, but only in the rendering stage
- NetCDF fields, summary CSVs, seam diagnostics, and the underlying score science are unchanged
- presentation maps can also apply a display-only low-end borderline treatment around the Poor/Fair boundary so genuinely marginal `40-50` score zones read as continuous raw-score gradients rather than abrupt low-end blocks
- stitched multi-region mosaics can now use a true CONUS presentation canvas with a national Lambert-style map extent, state outlines, lakes, and a stitched-footprint outline so the finished product reads like a national weather graphic rather than detached lat/lon tiles

Presentation filenames are written alongside the original outputs, for example:

- `*_presentation_score_YYYYMMDD.png`
- `*_presentation_category_YYYYMMDD.png`

Display smoothing/resampling only affects how the PNG looks:

- raw-score presentation maps may be interpolated to a finer display grid and lightly smoothed
- category presentation maps are derived from the display score field rather than blurring `category_index` directly
- in the low-end borderline band, presentation maps may add a subtle raw-score overlay and a thin Poor/Fair threshold guide; this is display-only and does not alter the saved science fields
- stitched CONUS presentation maps use a cartographic upgrade only; NetCDF fields, debug/science maps, and diagnostics remain the authoritative diagnostic products
- large stitched mosaics now use a cleaner national presentation block with the public-facing `Comfort Index` title, the `Daily Outdoor Comfort Score Across CONUS` subtitle, state/lake/cartographic styling, a softened coverage fade outside the stitched footprint, and a cleaner legend/colorbar layout
- the final stitched national pilot also separates the source/valid-date line from the main subtitle and adds a `No coverage` legend swatch so uncovered space is visually distinct from low-scoring covered areas
- the stitched category legend is also tuned for the public-facing `Poor / Fair / Pleasant / Ideal / Exceptional` label set and palette so the final national graphic reads cleanly at publishing size
- if Cartopy is unavailable, the stitched CONUS presentation path now falls back to a projected national basemap built from local U.S. state polygons so the final PNG still renders as a real CONUS map rather than a plain lat/lon frame
- seam artifacts are still meant to remain inspectable; the presentation path is not intended to hide them
- the recommended public-facing bundle is the stitched CONUS presentation score PNG plus the stitched CONUS presentation category PNG; debug/science PNGs, NetCDFs, and diagnostics remain the internal inspection products
- the final public stitched CONUS presentation is clipped to the CONUS land footprint, uses black state outlines, keeps the legend in the lower-left, and omits the old explanatory footer text so the map itself remains the focus

## Publish Preset Workflow

The project now includes a publish/product preset layer for repeatable regional or mosaic product generation.

Example regional publish run:

```powershell
python -m nicewx.main --source openmeteo --region southwest --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

Example mosaic publish run:

```powershell
python -m nicewx.main --source openmeteo --mosaic southwest rockies --date 2026-03-24 --publish-preset standard
```

Example 4-region mini-CONUS stitched publish run:

```powershell
python -m nicewx.main --source openmeteo --mosaic southwest rockies southeast northeast --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

Example bridge-region stitched runs with plains:

```powershell
python -m nicewx.main --source openmeteo --region plains --date 2026-03-24
python -m nicewx.main --source openmeteo --mosaic southwest rockies plains --date 2026-03-24 --publish-preset standard
python -m nicewx.main --source openmeteo --mosaic plains southeast --date 2026-03-24 --publish-preset standard
python -m nicewx.main --source openmeteo --mosaic southwest rockies plains southeast northeast --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

Example northern-coverage bridge runs with great_lakes:

```powershell
python -m nicewx.main --source openmeteo --region great_lakes --date 2026-03-24
python -m nicewx.main --source openmeteo --mosaic plains great_lakes northeast --date 2026-03-24 --publish-preset standard
python -m nicewx.main --source openmeteo --mosaic southwest rockies plains southeast northeast great_lakes --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

Example Pacific-side coverage runs with west_coast:

```powershell
python -m nicewx.main --source openmeteo --region west_coast --date 2026-03-24
python -m nicewx.main --source openmeteo --mosaic west_coast southwest rockies --date 2026-03-24 --publish-preset standard
python -m nicewx.main --source openmeteo --mosaic west_coast southwest rockies plains southeast northeast great_lakes --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

Example one-date pilot-day run across all currently supported real regions and seam mosaics:

```powershell
python -m nicewx.main --pilot-day --source openmeteo --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

The operational pilot-day workflow is cache-aware by default. In `--pilot-cache-mode reuse`, each regional Open-Meteo build is fetched once, persisted as a regional daily NetCDF, and then reused to build later stitched mosaics and publish bundles without refetching the same regional source data. This reduces rate-limit risk and makes reruns/resumes practical after a partial failure. Use `--pilot-cache-mode refresh` to force a full regional refetch.

Open-Meteo requests now also have lightweight reliability hardening:

- bounded retry/backoff for transient timeouts, `429`, and server-side HTTP errors
- a small configurable inter-request throttle to avoid bursty free-tier request patterns
- per-run request diagnostics CSVs that summarize total requests, retries, timeouts, `429`s, average latency, and the slowest workflow/endpoint

For direct real-data runs, verification runs, and pilot-day runs, the request diagnostics are written alongside the normal outputs using `*_openmeteo_request_summary.csv` and `*_openmeteo_request_detail.csv` naming.

Pilot-day and pilot-day-archive runs now also write a compact operational status artifact, `nicewx_pilot_day_<source>_<date>_status.csv` plus JSON, summarizing:

- attempted vs completed product count
- regions fetched vs reused from cache
- mosaics built from cached regional fields
- Open-Meteo request totals, retries, timeouts, and `429`s
- overall run status

The pilot-day HTML index and archive landing page link to these status summaries for easier operational monitoring.

Example archived operational run:

```powershell
python -m nicewx.main --pilot-day-archive --source openmeteo --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

For a date-less local daily-style run that uses the current default date handling:

```powershell
python -m nicewx.main --pilot-day-archive --source openmeteo --publish-preset standard --presentation-theme shareable --pilot-cache-mode reuse
```

The `standard` publish preset writes a bundle manifest alongside the normal outputs. The bundle includes:

- daily fields NetCDF
- debug/science score PNG
- debug/science category PNG
- presentation score PNG
- presentation category PNG
- summary CSV
- region sample CSV or mosaic seam-summary CSV where applicable

Bundle manifests are written as both CSV and JSON so downstream publishing workflows can pick them up directly.

Pilot-day mode is a packaging/orchestration workflow, not a new science path. It currently runs the full near-CONUS pilot inventory:

- regions: `west_coast`, `southwest`, `rockies`, `plains`, `southeast`, `northeast`, `great_lakes`
- stitched mosaics: `west_coast + southwest + rockies`, `southwest + rockies`, `plains + great_lakes + northeast`, `west_coast + southwest + rockies + plains + southeast + northeast + great_lakes`

It then writes a master index in CSV, JSON, and HTML form so the full one-date pilot product set is easy to browse. This is the current operational bridge between individual pilot products and the stitched near-CONUS pilot workflow.

The current stitched mini-CONUS publish workflow uses the validated real pilot regions:

- `southwest`
- `rockies`
- `plains`
- `great_lakes`
- `southeast`
- `northeast`

It reuses the existing mosaic blend and target-grid defaults, writes the same science/debug and presentation outputs as the pairwise seam mosaics, and extends the summary CSV with pairwise overlap diagnostics across the participating regional pairs. This is still a pilot stitched product, not a full national assembly.

`plains` is the next bridge region because it begins to connect the western stitched domain (`southwest + rockies`) toward the eastern stitched domain (`southeast + northeast`) without changing the underlying scoring or blending defaults. It improves continuity in the center of the pilot footprint, but it is still not a full national product.

`great_lakes` is the next northern coverage region because it starts to connect the central Plains bridge toward the Northeast through the Great Lakes corridor. That improves stitched northern continuity and expands the pilot footprint without changing the scoring science or the blend defaults.

`west_coast` is the next Pacific-side coverage region because it attaches the stitched pilot to the coastal West and adds a cleaner ocean-to-interior transition into the Southwest. That improves western coverage breadth and makes the stitched pilot feel more like a true CONUS-facing product without changing the scoring science or mosaic defaults.

Archived operational runs are organized under a dated archive tree, by default:

```text
output/
  archive/
    YYYY/
      MM/
        DD/
          nicewx_pilot_day_<source>_<date>_index.csv
          nicewx_pilot_day_<source>_<date>_index.json
          nicewx_pilot_day_<source>_<date>_index.html
          ...pilot-day product files...
```

The archive root also gets a landing page and archive-wide indexes:

- `output/archive/index.html`
- `output/archive/nicewx_archive_index.csv`
- `output/archive/nicewx_archive_index.json`

The landing page lists archived pilot-day runs and links to:

- each run's master index CSV/JSON/HTML
- pilot-day status summary CSV/JSON
- regional bundle manifests
- mosaic bundle manifests
- presentation maps
- summary files

## GitHub Actions Daily Operation

This repo includes a GitHub Actions workflow at `.github/workflows/pilot-day.yml`.

It:

- supports both manual trigger and daily scheduled trigger
- sets up Python 3.12
- installs the project dependencies
- runs the archived pilot-day workflow
- uploads `output/archive/` as a workflow artifact
- deploys the staged archive landing page to GitHub Pages on `main`

The workflow command is:

```bash
python -m nicewx.main --pilot-day-archive --source openmeteo --publish-preset standard --presentation-theme shareable --pilot-cache-mode reuse
```

It also stages a `site/` copy of `output/archive/` as a second artifact so the archive landing page can later be connected to GitHub Pages without changing the product build itself.

To view runs on the web, enable GitHub Pages for the repository and set the build source to **GitHub Actions**. After the workflow completes on `main`, the archived landing page and dated run folders are deployed through GitHub Pages.

## First Rollout Checklist

Before trusting the daily schedule:

1. Create a new GitHub repository and push this project.
2. Confirm the required checked-in rendering asset exists in the repo:
   - `nicewx/mapping/data/us_states.geojson`
3. Open the `Daily Pilot Day` workflow in GitHub Actions and run it manually once with `workflow_dispatch`.
4. Inspect the uploaded `nicewx-archive` artifact and confirm the archived landing page, pilot-day index, stitched CONUS presentation maps, and pilot-day status summary look correct.
5. Confirm the GitHub Pages deployment succeeded and the published archive landing page loads correctly.
6. Only after that manual check passes should you rely on the scheduled daily run.

Branding and theme knobs live in `nicewx/config.py`:

- `PRODUCT_METADATA` for product title, source/subtitle line, credit line, and branding footer
- `PRESENTATION_THEME_PRESETS` for presentation-only themes such as `default`, `shareable`, and the production-facing `public`
- `PUBLISH_PRESETS` for repeatable output-bundle presets

## Tuning

Most thresholds live in `nicewx/config.py`.

Key tuning surfaces:

- piecewise score bins for temperature, dew point, wind, clouds, and PoP
- interaction adjustments
- hazard penalties and caps
- daytime weights
- daily aggregation weights
- category thresholds and colors

## Replacing Mock Data Later

The loader boundary is `nicewx/data/loaders.py`.

To swap in real forecast data later:

1. implement a new `ForecastLoader` subclass
2. return an `xarray.Dataset` with `time`, `lat`, and `lon` dimensions
3. include at least these fields:
   `temp_f`, `dewpoint_f`, `wind_mph`, `gust_mph`, `cloud_pct`, `pop_pct`, `qpf_in`
4. optionally include `thunder`, `aqi`, `visibility_mi`, `precip_type`, or other future hooks

The scoring and plotting layers do not need to know where the data came from.

## Open-Meteo Real Data Path

The Open-Meteo point loader lives in `nicewx/data/openmeteo.py`.

It currently requests hourly forecast variables from the weather API:

- `temperature_2m`
- `relative_humidity_2m`
- `dew_point_2m`
- `wind_speed_10m`
- `wind_gusts_10m`
- `cloud_cover`
- `precipitation_probability`
- `precipitation`
- `weather_code`
- `visibility`
- `cape`

It also tries to merge optional hourly air-quality fields from the air-quality API:

- `us_aqi`
- `pm2_5`

Normalized project fields:

- `temp_f`
- `dewpoint_f`
- `wind_mph`
- `gust_mph`
- `cloud_pct`
- `pop_pct`
- `qpf_in`
- optional `weather_code`, `thunder`, `cape`, `visibility_mi`, `aqi`, `pm25`

Normalization notes:

- temperature is requested directly in Fahrenheit
- wind is requested directly in mph
- precipitation is requested directly in inches
- visibility is returned in meters and converted to miles
- dew point is used directly when available; if absent, it is derived from temperature and relative humidity
- thunder is approximated from thunderstorm weather codes and a CAPE plus precipitation-probability proxy

Current limitations:

- this is a point-only real-data path, not a regional/full-grid ingestion system
- it depends on forecast availability from Open-Meteo, so dates outside the provider forecast window may fail
- air-quality fields are optional and merged only if that secondary request succeeds

Next logical step for real data would be a multi-point or tiled regional ingestion workflow that populates the same normalized schema used by scoring today.

## Real-World Validation Harness

The real-world validation runner lives in `nicewx/validation/real_world.py`.

Default cases currently include:

- Raleigh
- Miami
- Denver
- Seattle
- Phoenix
- San Diego

Each case includes:

- `case_name`
- `lat`
- `lon`
- `date`
- optional `expected_label`

Output CSV columns include:

- `case_name`
- `lat`
- `lon`
- `date`
- `daily_score`
- `category`
- `best_3hr`
- `best_6hr`
- `daytime_weighted_mean`
- `reliability_score`
- `disruption_penalty`
- `explanation`
- `expected_label`
- `actual_label`
- `comparison`
- `dominant_limiting_factor`
- `top_reason_1`
- `top_reason_2`
- `top_reason_3`
- contribution summary columns such as mean component scores and total hazard/disruption effect

Interpretation:

- `match`: expected and actual category agree
- `near match`: off by one category tier
- `mismatch`: farther apart than one tier

Each validation run also writes a second CSV containing only `near match` and `mismatch` rows so calibration work can focus on the cases that need attention first.

This is meant for manual score tuning and calibration, not automated skill verification.

## Forecast-vs-Analysis Verification

The first-pass verification runner lives in `nicewx/validation/verify_model.py`.

It compares:

- archived forecast fields from the Open-Meteo Single Runs API using `gfs_seamless`
- archive-analysis / historical weather fields from the Open-Meteo Archive API using `best_match`

Both sides are normalized into the existing hourly Comfort Index schema, then scored with the exact same hourly and daily scoring logic used elsewhere in the project.

Run it with:

```bash
python -m nicewx.validation.verify_model --date 2026-03-20 --region southeast
```

First-pass outputs include:

- forecast Comfort Index daily fields NetCDF
- analysis Comfort Index daily fields NetCDF
- forecast score map
- analysis score map
- forecast-minus-analysis score difference map
- verification summary CSV with bias / MAE / RMSE and category-agreement fractions
- sample-point CSV for regional spot checks

Current limitations:

- this is intentionally a limited regional verification workflow, not full-CONUS verification architecture
- the forecast side currently uses archived `gfs_seamless` runs as the practical first model source
- the truth side currently uses Open-Meteo historical/archive analysis fields, not direct station observations
- archive analysis does not provide probabilistic precipitation cleanly, so the verification path derives a simple `pop_pct` proxy from analyzed hourly precipitation

For a small multi-date benchmark over representative regions, use:

```bash
python -m nicewx.validation.verify_benchmark
```

The benchmark runner currently includes configurable default cases for:

- `southeast`
- `southwest`
- `plains`
- `northeast`

across multiple representative dates. It reuses the same per-run verification workflow, keeps the detailed forecast/analysis/difference maps and CSVs for each case, and writes a combined benchmark summary CSV so you can compare regional bias, MAE, RMSE, and category agreement in one place.

## Future Extensions

This layout is designed to support:

- real NBM, HRRR, or Open-Meteo loaders
- multiple forecast days
- solar-aware local daytime windows
- AQI and smoke penalties
- winter weather and freezing-precip logic
- activity-specific scoring modes
- climatology adjustments
- web and API export

## Regional Processing Framework

The project now includes an overlap-aware regional domain layer in `nicewx/mapping/regions.py`.

Current regions:

- `west_coast`: core `(-125.0, -116.0, 31.0, 49.0)`
- `southwest`: core `(-118.0, -107.0, 31.0, 41.5)`
- `rockies`: core `(-113.5, -102.0, 37.0, 49.0)`
- `plains`: core `(-104.5, -92.0, 31.0, 49.0)`
- `great_lakes`: core `(-94.0, -80.0, 40.0, 49.5)`
- `southeast`: core `(-92.0, -75.0, 24.0, 37.5)`
- `northeast`: core `(-82.0, -66.5, 37.0, 47.5)`

Each region has an overlap buffer so adjacent regional products can later blend without hard seams. The current helper layer includes:

- region subsetting with overlap-aware expanded bounds
- overlap masks
- tapering blend weights
- lightweight target-grid alignment metadata
- a basic weighted-overlap mosaic helper for compatible mock regional rasters

Current regional mode:

```powershell
python -m nicewx.main --region southeast
```

Pilot real regional mode:

```powershell
python -m nicewx.main --source openmeteo --region southeast --date 2026-03-24
python -m nicewx.main --source openmeteo --region southwest --date 2026-03-24
python -m nicewx.main --source openmeteo --region southwest --mesh-profile fine --date 2026-03-24
python -m nicewx.main --source openmeteo --region rockies --date 2026-03-24
python -m nicewx.main --source openmeteo --region rockies --mesh-profile fine --date 2026-03-24
python -m nicewx.main --source openmeteo --region northeast --date 2026-03-24
```

Pilot 2-region real mosaic test:

```powershell
python -m nicewx.main --source openmeteo --mosaic southeast southwest --date 2026-03-24
python -m nicewx.main --source openmeteo --mosaic southeast northeast --date 2026-03-24
python -m nicewx.main --source openmeteo --mosaic southwest rockies --date 2026-03-24
python -m nicewx.validation.seam_compare --date 2026-03-24
python -m nicewx.validation.western_mesh_sensitivity --date 2026-03-24
python -m nicewx.validation.western_mosaic_method_sensitivity --date 2026-03-24
python -m nicewx.validation.western_seam_attribution --date 2026-03-24
python -m nicewx.validation.western_aggregation_sensitivity --date 2026-03-24
```

This currently works with the mock gridded path and writes:

- a regional daily fields NetCDF with `daily_score`, `category_index`, `blend_weight`, and `overlap_mask`
- regional raw/category PNGs
- a regional summary CSV

The pilot real regional path uses Open-Meteo as a coarse regional mesh source for `southeast`, `southwest`, `rockies`, and `northeast`. It is not a native full raster download yet. Instead, it builds an overlap-aware regional grid from repeated batched point forecasts and assembles them into the same normalized `time/lat/lon` schema used elsewhere in the project.

Current pilot limitations:

- only `southeast`, `southwest`, `rockies`, and `northeast` are implemented for real regional ingestion
- resolution is intentionally coarse and controlled by `OPENMETEO_REGIONAL_MESH_SETTINGS` in `config.py`
- `southwest` and `rockies` now also have optional finer pilot mesh profiles for seam-sensitivity checks
- each pilot mesh currently uses a fixed regional timezone for aligned hourly arrays
- this is a stepping stone toward future true-grid or tiled multi-region ingestion, not the final CONUS production path

Pilot real regional outputs include:

- regional NetCDF
- regional raw/category PNGs
- regional presentation raw/category PNGs
- regional summary CSV with min/mean/max scores, category counts, overlap fraction, and sample-point coordinates
- regional sample-point summary CSV

Supporting both `southeast` and `southwest` is useful for calibration because they stress the same scoring engine with very different regimes:

- humid convective warmth and cloud/rain disruption in the Southeast
- dry heat, terrain-driven gradients, and cleaner skies in the Southwest

Adding `northeast` creates the first true adjacent-region seam test with the Southeast:

- both regions share an overlap corridor near the Mid-Atlantic and Appalachians
- this lets the mosaic helper report real overlap-cell counts and pre-blend score disagreement statistics
- seam diagnostics help show whether neighboring regional meshes are producing reasonable continuity before broader multi-region expansion

Adding `rockies` creates a second adjacent seam regime with the Southwest:

- this pair stresses terrain and arid-climate transitions instead of humid eastern gradients
- it helps expose whether neighboring western meshes stay coherent across dry-heat and elevation contrasts
- comparing the two seam regimes gives a better read on whether the blending approach is robust beyond one part of the country

Western mesh-sensitivity testing:

- `southwest` and `rockies` can be run with `--mesh-profile fine` to increase Open-Meteo point density without changing the score science
- this is meant to isolate whether western seam roughness is primarily a mesh-density issue
- the helper `python -m nicewx.validation.western_mesh_sensitivity --date 2026-03-24` runs both `standard` and `fine` western mosaics and writes comparison CSVs
- comparison metrics include seam diagnostics for both profiles plus grid-to-grid daily score change statistics

Western mosaic-method sensitivity:

- the mesh-sensitivity pass showed that higher western mesh density helps somewhat, but does not fully remove seam roughness
- this project now includes diagnostic overlap blend modes for western seam testing:
  - `taper` (default production-style blend)
  - `equal_overlap`
  - `winner_take_all`
- the mosaic layer can also be tested on:
  - the current `adaptive` target grid policy
  - an explicit `fixed_western` target grid defined in `config.py`
- these options are diagnostic only; they do not change the underlying score science
- use `python -m nicewx.validation.western_mosaic_method_sensitivity --date 2026-03-24` to compare methods
- key outputs include overlap diagnostics, pre-blend score-difference distributions, post-blend overlap score distributions, near-threshold overlap counts, and category flips relative to the baseline taper/adaptive mosaic

Western seam attribution:

- the attribution pass diagnoses what the southwest and rockies fields are disagreeing about inside the overlap zone
- it summarizes per-region overlap means and pairwise differences for:
  - `temp_score`
  - `dewpoint_score`
  - `wind_score`
  - `cloud_score`
  - `precip_score`
  - `hazard_penalty`
  - `interaction_adjustment`
  - `daily_score`
  - `category_index`
- it also includes overlap-zone daily attribution fields like `reliability_score` and `disruption_penalty`
- use `python -m nicewx.validation.western_seam_attribution --date 2026-03-24` to write:
  - one summary CSV
  - one overlap-cell detail CSV
- this is intended to show whether the western seam is mostly thermodynamic, cloud/precip, or aggregation/reliability driven before changing any score formulas

Western aggregation sensitivity:

- seam attribution showed that the western overlap roughness is driven mostly by `reliability_score`, with `temp_score` secondary
- this project now includes a softer diagnostic daily aggregation mode, `soft_reliability`, alongside the unchanged `baseline` mode
- the softer mode keeps the same daily philosophy but reduces brittleness by:
  - grading usable/strong reliability contributions instead of relying only on hard hourly cutoffs
  - softening some rain/gust/score-drop propagation in the disruption layer
- the default production aggregation remains `baseline`
- use `python -m nicewx.validation.western_aggregation_sensitivity --date 2026-03-24` to compare:
  - overlap `reliability_score` roughness
  - overlap `daily_score` roughness
  - overlap category agreement / near agreement
  - overlap category flips between baseline and tuned mosaics
  - whether `temp_score` remains the secondary seam driver

Western threshold sensitivity:

- the softer aggregation mode improved western raw-score seam metrics, but exact category agreement still got worse because more overlap cells ended up near or across category thresholds
- this project now includes a threshold-brittleness runner for `southwest + rockies` that compares `baseline` vs `soft_reliability`
- use `python -m nicewx.validation.western_threshold_sensitivity --date 2026-03-24` to write:
  - one summary CSV with threshold-proximity fractions, threshold-crossing counts, category-flip totals, and a small margin-stable comparison metric
  - one overlap-cell detail CSV for the category flips
- this pass is diagnostic only; it does not change the production thresholds or add spatial smoothing to categories

Western Poor/Fair boundary audit:

- the threshold-sensitivity pass showed that the western seam flips cluster most strongly at the `45` Poor/Fair boundary
- this project now includes a low-end audit for `southwest + rockies` that focuses on whether those flips are mostly unavoidable marginal-score behavior or mostly an artifact of the hard category split
- use `python -m nicewx.validation.western_fair_nice_audit --date 2026-03-24` to write:
  - one summary CSV with low-end score-band fractions and interpretation
  - one alternatives CSV comparing small diagnostic-only Poor/Fair threshold variants
  - one detail CSV for production Poor/Fair flips
- this pass is comparison-only; it does not change the production thresholds, scoring formulas, or maps

The 2-region mosaic test is a proof of concept for common-grid alignment and weighted blending. It shows that two independently processed real regional products can be regridded onto one target grid, blended with the existing tapering weights, and written back out as one combined daily field without changing the scoring schema.

Current mosaic limitation:

- `southeast + southwest` mainly validates target-grid alignment, gap handling, and reusable mosaic diagnostics because those two pilot regions do not directly overlap
- `southeast + northeast` is now the first real adjacent seam-behavior test and should be preferred when evaluating overlap blending
- `southwest + rockies` is the second real adjacent seam-behavior test and is useful for western terrain/arid stress testing

Current seam diagnostics include:

- overlap cell count
- overlap fraction of covered cells
- pre-blend mean and max absolute score difference in the overlap zone
- overlap-zone category agreement and near-agreement fractions
- post-blend overlap mean score

For a quick side-by-side comparison of the two adjacent seam regimes, run:

```powershell
python -m nicewx.validation.seam_compare --date 2026-03-24
```

That writes a single CSV with the shared seam diagnostic fields for:

- `southeast + northeast`
- `southwest + rockies`

Why regions and overlaps matter:

- regional processing keeps future real-data ingestion manageable
- overlapping edges reduce seam artifacts when adjacent regions are merged
- later CONUS mosaicking can blend overlap zones onto a common target grid instead of stitching hard cut lines

What remains for a full CONUS product:

- real gridded regional ingestion
- explicit common-grid regridding
- broader adjacent-region seam testing and mosaic-wide pristine-gate recomputation
- production output assembly across all regions

## Tests

Basic sanity tests are included:

```powershell
pytest
```
