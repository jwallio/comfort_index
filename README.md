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

By default, the verification runner now prefers Open-Meteo `HRRR` for `Day 1` forecast verification while keeping the longer-lead benchmark path on the current forecast model selection. That improves short-range source fidelity without changing the Comfort Index scoring engine.

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

Reuse existing per-case verification artifacts when iterating on benchmark reports and calibration:

```powershell
python -m comfortwx.validation.verify_benchmark --benchmark-tier default --lead-days 1,2,3,7 --case-cache-mode reuse
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

The repository includes a daily workflow:
- `Comfort Index Daily Schedule`

For a simple manual GitHub Actions menu, use:
- `Comfort Index Run Menu`

Manual run-menu builds support an optional `publish_site` choice so you can decide whether a custom run should update the live GitHub Pages site. Scheduled runs publish the latest archive automatically.

And a separate verification workflow:
- `Comfort Index Verification Benchmark`

It runs:

```powershell
python -m comfortwx.main --pilot-day-archive --source openmeteo --pilot-span-days 7 --publish-preset standard --presentation-theme shareable --pilot-cache-mode reuse
```

The workflow uploads:
- `comfortwx-archive`
- `comfortwx-pages-preview`

When run manually from GitHub Actions, the workflow menu lets you choose:
- `pilot-day` or `pilot-day-archive`
- `1`, `2`, `3`, or `7` consecutive forecast days
- an optional start date in `YYYY-MM-DD`

If GitHub Pages is enabled for the repository, the workflow is also prepared to publish the archived run output as a site. If Pages is not available, the archive build still completes and uploads artifacts.

The public Pages view is map-first by design:
- the landing page highlights the stitched CONUS images and archived run galleries
- run pages focus on presentation PNGs
- supporting CSV, JSON, and NetCDF files remain in the archive but are not the main public navigation

Verification workflow:

```powershell
python -m comfortwx.validation.verify_benchmark
```

The `Comfort Index Verification Benchmark` GitHub Actions workflow runs the proxy forecast-vs-analysis benchmark suite and uploads a `comfortwx-verification-benchmark` artifact containing:
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
  - `full-seasonal`
- optional benchmark date override in `YYYY-MM-DD`
- mesh profile (`standard` or `fine`)
- forecast leads as a comma-separated list such as `1,2,3,7`

This workflow is the current backtesting path. It is separate from the daily public product workflow.

Benchmark tiers:
- `default`
  - `southeast`, `southwest`, `plains`, `northeast`
  - dates: `2026-01-15`, `2026-03-20`
  - intended for quicker routine benchmark runs
- `full-seasonal`
  - `west_coast`, `southwest`, `rockies`, `plains`, `southeast`, `northeast`, `great_lakes`
  - dates: `2026-01-15`, `2026-03-20`, `2026-05-15`, `2026-07-20`, `2026-09-20`, `2026-11-15`
  - intended for broader seasonal coverage and deeper accuracy review

By default, the benchmark now verifies multiple forecast leads for each benchmark case:
- `Day 1`
- `Day 2`
- `Day 3`
- `Day 7`

The benchmark CSV, charts, and HTML report include the lead day so you can compare how Comfort Index skill changes with forecast horizon instead of only looking at a single one-day lead.

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

The verification workflow also writes a self-contained verification mini-site under `output/verification_site/latest/`. The daily GitHub Pages build is prepared to pull in the most recent verification artifact and expose it under `/verification/` on the public site when a benchmark artifact is available.

## Repository Notes

- Required checked-in rendering asset:
  - `comfortwx/mapping/data/us_states.geojson`
- Detailed internal methodology and operational notes are intentionally not included in this public README.
- A local non-tracked internal reference copy can be kept as `README.internal.md`.

## First Rollout Checklist

1. Push the repository to GitHub.
2. Run the `Comfort Index Manual Run` workflow manually once.
3. Inspect the `comfortwx-archive` artifact.
4. Confirm the stitched CONUS presentation outputs look correct.
5. Only then rely on the daily scheduled run.
