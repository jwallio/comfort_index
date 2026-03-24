# Comfort Index

Comfort Index is a Python project for generating daily outdoor comfort maps and publishable stitched CONUS products from weather data.

The repository includes:
- regional and stitched mosaic product generation
- debug/science outputs and public presentation outputs
- publish bundles and dated archive workflows
- verification and diagnostics tooling
- GitHub Actions automation for daily archived runs

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

It runs:

```powershell
python -m comfortwx.main --pilot-day-archive --source openmeteo --publish-preset standard --presentation-theme shareable --pilot-cache-mode reuse
```

The workflow uploads:
- `comfortwx-archive`
- `comfortwx-pages-preview`

If GitHub Pages is enabled for the repository, the workflow is also prepared to publish the archived run output as a site. If Pages is not available, the archive build still completes and uploads artifacts.

The public Pages view is map-first by design:
- the landing page highlights the stitched CONUS images and archived run galleries
- run pages focus on presentation PNGs
- supporting CSV, JSON, and NetCDF files remain in the archive but are not the main public navigation

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
