# Comfort Index

Comfort Index is a Python project for generating daily outdoor comfort maps and publishable stitched CONUS products from weather data.

The repository includes:
- regional and stitched mosaic product generation
- debug/science outputs and public presentation outputs
- publish bundles and dated archive workflows
- verification and diagnostics tooling
- GitHub Actions automation for daily archived runs
- GitHub Actions automation for verification benchmarks

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run a basic local build:

```powershell
python -m comfortwx.main
```

Run the archived pilot-day workflow locally:

```powershell
python -m comfortwx.main --pilot-day-archive --source openmeteo --publish-preset standard --presentation-theme shareable --pilot-cache-mode reuse
```

Run a multi-day archived forecast sequence locally:

```powershell
python -m comfortwx.main --pilot-day-archive --source openmeteo --pilot-span-days 7 --publish-preset standard --presentation-theme shareable --pilot-cache-mode reuse
```

Run a side-by-side candidate product build without changing the default public path:

```powershell
python -m comfortwx.main --pilot-day-archive --source openmeteo --pilot-span-days 7 --publish-preset standard --presentation-theme shareable --pilot-cache-mode reuse --aggregation-policy experimental_all_leads_soft
```

## Main Commands

Regional run:

```powershell
python -m comfortwx.main --source openmeteo --region southwest --date 2026-03-24
```

Stitched CONUS-style run:

```powershell
python -m comfortwx.main --source openmeteo --mosaic west_coast southwest rockies plains southeast northeast great_lakes --date 2026-03-24 --publish-preset standard --presentation-theme shareable
```

Verification run:

```powershell
python -m comfortwx.validation.verify_model --date 2026-03-20 --region southeast
```

By default, verification now uses NOAA `URMA` surface analysis with `RTMA` fallback as the truth source, while still preferring Open-Meteo `HRRR` for `Day 1` forecast verification and the existing forecast-model selection for longer leads.

To compare against the older Open-Meteo archive proxy truth explicitly:

```powershell
python -m comfortwx.validation.verify_model --date 2026-03-20 --region southeast --analysis-model best_match
```

Benchmark verification suite:

```powershell
python -m comfortwx.validation.verify_benchmark
```

Benchmark verification suite with explicit lead days:

```powershell
python -m comfortwx.validation.verify_benchmark --lead-days 1,2,3,7
```

Benchmark verification suite with an explicit tier:

```powershell
python -m comfortwx.validation.verify_benchmark --benchmark-tier full-seasonal --lead-days 1,2,3,7
```

Focused MAE verification tier for the highest-value tuning regions:

```powershell
python -m comfortwx.validation.verify_benchmark --benchmark-tier focused-mae --lead-days 1,2,3,7
```

Benchmark verification suite comparing baseline vs a candidate verification policy:

```powershell
python -m comfortwx.validation.verify_benchmark --benchmark-tier full-seasonal --lead-days 1,2,3,7 --aggregation-policies baseline,experimental_regime_aware
```

Rate-limit-safe incremental full-seasonal benchmark pass:

```powershell
python -m comfortwx.validation.verify_benchmark --benchmark-tier full-seasonal --lead-days 1,2,3,7 --case-cache-mode reuse --max-fresh-cases 12
```

Reuse existing per-case verification artifacts when iterating on benchmark reports and calibration:

```powershell
python -m comfortwx.validation.verify_benchmark --benchmark-tier default --lead-days 1,2,3,7 --case-cache-mode reuse
```

Verification-only daily aggregation tuning harness:

```powershell
python -m comfortwx.validation.tune_daily_aggregation --benchmark-tier default --lead-days 1,2,3,7 --case-cache-mode reuse
```

Rate-limit-safe incremental full-seasonal tuning pass:

```powershell
python -m comfortwx.validation.tune_daily_aggregation --benchmark-tier full-seasonal --lead-days 1,2,3,7 --case-cache-mode reuse --max-fresh-cases 12
```

## Outputs

Typical outputs include:
- daily fields NetCDF
- score PNG
- category PNG
- presentation score PNG
- presentation category PNG
- summary CSV
- publish bundle CSV/JSON
- pilot-day index and status files for archived runs

Public-facing product bundle:
- stitched CONUS presentation score map
- stitched CONUS presentation category map

Internal/debug outputs remain available for inspection and validation.

## GitHub Actions

The repository includes a scheduled product workflow:
- `Comfort Index Run Menu`

For a simple manual GitHub Actions menu, use:
- `Comfort Index Run Menu`

Run-menu builds are the stable product-build path and upload the archived output as a GitHub Actions artifact. The same workflow is scheduled to run at `00:20 UTC`, `06:20 UTC`, `12:20 UTC`, and `18:20 UTC`.

GitHub Pages publishing is handled separately by:
- `Comfort Index Site Publish`

That publisher now runs automatically after a successful `Comfort Index Run Menu` build and can also be run manually. It publishes the matching `comfortwx-archive` artifact from the triggering product run plus the latest successful verification artifact when available.

Separate verification workflows:
- `Comfort Index Verification Benchmark`
- `Comfort Index Verification Tuning`

Product workflow command:

```powershell
python -m comfortwx.main --pilot-day-archive --source openmeteo --pilot-span-days 7 --publish-preset standard --presentation-theme shareable --pilot-cache-mode reuse
```

Product workflow uploads:
- `comfortwx-archive`

When run manually from GitHub Actions, the workflow menu lets you choose:
- `pilot-day` or `pilot-day-archive`
- `1`, `2`, `3`, or `7` consecutive forecast days
- an optional start date in `YYYY-MM-DD`
- aggregation policy:
  - `baseline`
  - `experimental_regime_aware`
  - `experimental_regime_aware_plus`
  - `experimental_all_leads_soft`

Policy guidance:
- `baseline`
  - current public-control path
- `experimental_regime_aware`
  - older targeted softening candidate
- `experimental_regime_aware_plus`
  - refined targeted candidate that also softens the Northeast cool-season `D+2/D+3` path
- `experimental_all_leads_soft`
  - current best MAE candidate on the broader NOAA-truth verification gate

The product workflow keeps `baseline` as the default public-scoring path. The other policies are available only as explicit side-by-side evaluation options.

If GitHub Pages is enabled for the repository, the site-publish workflow updates the public site from the latest archived run artifact. If Pages is not available, the archive build still completes and uploads artifacts.

The public Pages view is map-first by design:
- the landing page highlights the stitched CONUS images and archived run galleries
- run pages focus on presentation PNGs
- supporting CSV, JSON, and NetCDF files remain in the archive but are not the main public navigation

Verification benchmark command:

```powershell
python -m comfortwx.validation.verify_benchmark
```

The `Comfort Index Verification Benchmark` GitHub Actions workflow runs the forecast-vs-analysis benchmark suite and uploads a `comfortwx-verification-benchmark` artifact containing:
- combined benchmark summary CSV
- benchmark HTML report
- benchmark summary PNG charts
- per-case forecast score maps
- per-case analysis score maps
- per-case difference maps
- per-case absolute error maps
- per-case category disagreement maps
- per-case missed high comfort / false high comfort masks
- per-case component attribution CSVs
- per-case summary and point CSVs

When run manually from GitHub Actions, the verification workflow lets you choose:
- benchmark tier:
  - `default`
  - `focused-mae`
  - `full-seasonal`
  - `ndfd-west-coast`
- optional benchmark date override in `YYYY-MM-DD`
- mesh profile (`standard` or `fine`)
- forecast leads as a comma-separated list such as `1,2,3,7`
- analysis model:
  - `noaa_urma_rtma` (default NOAA truth)
  - `best_match` (legacy Open-Meteo proxy truth)
- forecast model:
  - `gfs_seamless`
  - `ncep_hrrr_conus`
  - `ecmwf_ifs`
  - `ecmwf_ifs025`
  - `nws_ndfd_hourly`
  - `nws_ndfd_gfs_blend`
  - `nws_ndfd_cloud_blend`
  - `nws_ndfd_thermal_cloud_blend`
  - `nws_ndfd_surface_blend`
- forecast model mode:
  - `auto` prefers HRRR for `Day 1`
  - `exact` uses the selected model literally
- verification aggregation policies as a comma-separated list such as `baseline,experimental_regime_aware`
- case cache mode (`reuse` or `refresh`)
- optional fresh-case cap, cooldown, region filter, and date filter for incremental chunked runs

This workflow is the current backtesting path. It is separate from the daily public product workflow.
On GitHub Actions, the verification benchmark and tuning workflows now also restore/save the raw Open-Meteo HTTP cache, the scored-hourly verification cache, and the NOAA analysis cache so repeated chunked runs can build on earlier progress instead of starting from a cold cache every time.

Verification tuning command:

```powershell
python -m comfortwx.validation.tune_daily_aggregation
```

The `Comfort Index Verification Tuning` GitHub Actions workflow runs the held-out daily aggregation tuning harness and uploads a `comfortwx-verification-tuning` artifact containing:
- per-case candidate score CSVs
- candidate summary CSVs by lead and aggregation mode
- recommended mode by lead
- held-out selection CSVs
- experimental policy comparison CSVs
- tuning charts and HTML report

When run manually from GitHub Actions, the tuning workflow lets you choose:
- benchmark tier
- optional benchmark date override
- lead days
- analysis model
- forecast model
- forecast model mode
- candidate aggregation modes
- case cache mode (`reuse` or `refresh`)
- optional fresh-case cap, cooldown, region filter, and date filter for incremental chunked runs

Benchmark tiers:
- `default`
  - `southeast`, `southwest`, `plains`, `northeast`
  - dates: `2026-01-15`, `2026-03-20`
  - intended for quicker routine benchmark runs
- `focused-mae`
  - `southeast`, `plains`, `northeast`
  - dates: `2025-01-15`, `2025-03-20`, `2025-04-20`, `2025-06-20`, `2025-07-20`, `2025-08-20`, `2025-10-20`, `2025-11-15`
  - intended for faster MAE tuning across the regions currently driving the biggest verification misses
- `full-seasonal`
  - `west_coast`, `southwest`, `rockies`, `plains`, `southeast`, `northeast`, `great_lakes`
  - dates: `2025-01-15`, `2025-03-20`, `2025-05-15`, `2025-07-20`, `2025-09-20`, `2025-11-15`
  - intended for broader seasonal coverage and deeper accuracy review
- `ndfd-west-coast`
  - `west_coast`
  - dates: `2024-03-20`, `2024-05-15`, `2024-07-20`, `2024-08-20`
  - lead: `Day 1` only
  - intended for first-pass NWS/NDFD archive source comparison against NOAA truth

NDFD scaffold notes:
- `nws_ndfd_hourly` is a verification-only first scaffold built from archived NDFD core files at NCEI
- current scope is `west_coast` and `Day 1` only
- it currently ingests temperature, dew point, cloud cover, wind speed, gusts, and 12-hour PoP
- `nws_ndfd_gfs_blend` uses those NDFD surface fields and falls back to exact `gfs_seamless` for the remaining hazard/support fields such as QPF, weather code, thunder proxy, CAPE, and visibility
- narrower verification-only blend variants are also available:
  - `nws_ndfd_cloud_blend`
  - `nws_ndfd_thermal_cloud_blend`
  - `nws_ndfd_surface_blend`
- direct ECMWF historical forecast comparisons are also supported through Open-Meteo previous-runs:
  - `ecmwf_ifs` for IFS HRES 9 km
  - `ecmwf_ifs025` for IFS 0.25°
- QPF is still a PoP-based proxy in this scaffold, so treat it as a source-comparison path, not final production science

By default, the benchmark now verifies multiple forecast leads for each benchmark case:
- `Day 1`
- `Day 2`
- `Day 3`
- `Day 7`

The benchmark CSV, charts, and HTML report include the lead day so you can compare how Comfort Index skill changes with forecast horizon instead of only looking at a single one-day lead.

When multiple verification aggregation policies are requested, the benchmark writes a combined summary CSV with a `verification_aggregation_policy` column plus an aggregation-policy comparison CSV so baseline and candidate policies can be compared directly without changing the public product path.

The benchmark report layers visuals on top of the CSV outputs:
- MAE, agreement, bias-vs-RMSE, and ranked-case summary charts
- MAE, agreement, and bias time-series charts across benchmark dates, grouped by forecast lead
- component MAE heatmap across cases
- regional rollup CSV/chart with pass-rate and mean error context
- forecast-lead rollup CSV/chart so Day 1, Day 2, Day 3, and Day 7 can be compared directly
- region-by-lead MAE heatmap so the worst horizon/region combinations stand out immediately
- ranked priority-case CSV/chart so the most important misses are reviewed first
- component-priority CSV/chart so you can see whether temperature, dew point, cloud, precip, or reliability issues are driving the largest opportunities
- a simple HTML dashboard with summary metrics and linked map thumbnails
- threshold flags for MAE, near-category agreement, and mean bias so regressions are easier to spot
- held-out calibration review so raw vs calibrated MAE can be compared without changing the scoring engine

Each verification case now also writes a component-attribution CSV that summarizes whether the miss was mainly tied to temperature, dew point, cloud, precipitation, or daily reliability/disruption behavior.

Lead-specific benchmark thresholds are now applied so longer leads are judged against more realistic expectations while still being ranked consistently in the improvement-priority outputs.

The benchmark runner can also reuse already-written per-case verification artifacts. That makes it much easier to iterate on benchmark tiers, calibration summaries, and HTML reporting without refetching the same Open-Meteo cases on every pass.

To keep Open-Meteo rate limits under control, verification and tuning now also:
- use slower verification-specific request throttles than the public product path
- apply longer `429` backoff, including `Retry-After` support when the API provides it
- avoid exploding a rate-limited HRRR batch into point-by-point fallback requests
- cap uncached `full-seasonal` runs to a smaller number of fresh cases per run by default, so repeated `--case-cache-mode reuse` runs can fill the cache incrementally instead of bursting the API in one pass
- support `--regions`, `--dates`, `--max-fresh-cases`, and `--case-cooldown-seconds` for explicit chunking when needed

With the NOAA truth path, verification also caches raw NOAA analysis files under `output/noaa_analysis_cache` so repeated benchmark and tuning runs do not need to refetch the same URMA/RTMA GRIB files.

The daily aggregation tuning runner builds on the same benchmark cases, but compares multiple candidate daily aggregation modes instead of one fixed mode. It writes:
- per-case candidate score CSVs
- candidate summary CSVs by lead and aggregation mode
- recommended mode by forecast lead
- held-out mode-selection summary
- experimental policy comparison CSVs against baseline
- compact HTML and chart outputs for tuning review

The verification workflow also writes a self-contained verification mini-site under `output/verification_site/latest/`. The daily GitHub Pages build is prepared to pull in the most recent verification artifact and expose it under `/verification/` on the public site when a benchmark artifact is available.

## Repository Notes

- Required checked-in rendering asset:
  - `comfortwx/mapping/data/us_states.geojson`
- Detailed internal methodology and operational notes are intentionally not included in this public README.
- A local non-tracked internal reference copy can be kept as `README.internal.md`.

## First Rollout Checklist

1. Push the repository to GitHub.
2. Run the `Comfort Index Run Menu` workflow manually once.
3. Inspect the `comfortwx-archive` artifact.
4. Run `Comfort Index Site Publish` manually once and confirm the Pages site updates.
5. Confirm the stitched CONUS presentation outputs look correct.
6. Only then rely on the four-times-daily scheduled runs.
