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

Benchmark verification suite:

```powershell
python -m comfortwx.validation.verify_benchmark
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
- `Comfort Index Daily Pilot Day`

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
- optional benchmark date override in `YYYY-MM-DD`
- mesh profile (`standard` or `fine`)

This workflow is the current backtesting path. It is separate from the daily public product workflow.

The benchmark report layers visuals on top of the CSV outputs:
- MAE, agreement, bias-vs-RMSE, and ranked-case summary charts
- MAE, agreement, and bias time-series charts across benchmark dates
- component MAE heatmap across cases
- regional rollup CSV/chart with pass-rate and mean error context
- a simple HTML dashboard with summary metrics and linked map thumbnails
- threshold flags for MAE, near-category agreement, and mean bias so regressions are easier to spot

Each verification case now also writes a component-attribution CSV that summarizes whether the miss was mainly tied to temperature, dew point, cloud, precipitation, or daily reliability/disruption behavior.

The verification workflow also writes a self-contained verification mini-site under `output/verification_site/latest/`. The daily GitHub Pages build is prepared to pull in the most recent verification artifact and expose it under `/verification/` on the public site when a benchmark artifact is available.

## Repository Notes

- Required checked-in rendering asset:
  - `comfortwx/mapping/data/us_states.geojson`
- Detailed internal methodology and operational notes are intentionally not included in this public README.
- A local non-tracked internal reference copy can be kept as `README.internal.md`.

## First Rollout Checklist

1. Push the repository to GitHub.
2. Run the `Comfort Index Daily Pilot Day` workflow manually once.
3. Inspect the `comfortwx-archive` artifact.
4. Confirm the stitched CONUS presentation outputs look correct.
5. Only then rely on the daily scheduled run.
