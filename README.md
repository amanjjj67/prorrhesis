# PRORRHESIS

**PRORRHESIS** is a local football match-prediction and season-simulation research lab for Europe's top five leagues. It combines statistical and machine-learning models with leakage-aware walk-forward evaluation, probability calibration, and an interactive local web application.

> Research software, not betting advice. Predictions are probabilistic estimates and are not guarantees of match outcomes.

## What it does

- Ingests historical results and upcoming fixtures for the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1 from football-data.co.uk.
- Builds team and match features from historical performance, with optional ClubElo rating snapshots.
- Compares baseline, Elo, Dixon–Coles, xG-Poisson, gradient-boosted, and market-informed models.
- Evaluates forecasts using time-ordered walk-forward backtests and held-out seasons; supports ensembles and probability calibration.
- Serves a local dashboard and JSON API for fixtures, predictions, model diagnostics, calibration, simulations, and a prediction ledger.
- Simulates season outcomes and supports what-if scenarios.

The optional Understat xG integration is not implemented. The xG-Poisson model currently uses the project's shot-based xG proxy. ClubElo is optional; the application is designed to keep working when its service is unavailable. See [DEVIATIONS.md](DEVIATIONS.md) for implementation choices and their rationale.

## Requirements

- Python 3.11 or newer
- Internet access for initial football-data.co.uk ingestion; ClubElo is optional

## Install

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` after creating it with `python3 -m venv .venv`.

## Quick start

```powershell
# Download and ingest historical results and fixtures
python -m prorrhesis ingest

# Optionally fetch ClubElo ratings; the rest of the app works without them
python -m prorrhesis clubelo fetch

# Start the local dashboard and API
python -m prorrhesis serve
```

Then open <http://127.0.0.1:8000>. FastAPI's interactive API reference is at <http://127.0.0.1:8000/api/docs>.

The application creates its working database, downloaded files, and logs under `data/`; these are local runtime artifacts rather than required source files. To see all command options, run `python -m prorrhesis --help` or `python -m prorrhesis <command> --help`.

## Evaluation workflow

Use chronological evaluation rather than a random split for forecasting work:

```powershell
# Inspect available commands and supported options
python -m prorrhesis --help

# Run a walk-forward evaluation on the tuning window
python -m prorrhesis backtest --kind tuning

# Build the ensemble and calibration diagnostics as part of a run
python -m prorrhesis backtest --kind tuning --ensemble

# Declare a frozen configuration before evaluating the held-out headline window
python -m prorrhesis holdout declare
python -m prorrhesis backtest --kind headline
```

The headline evaluation is guarded by a configuration lock to help prevent accidental tuning against the holdout. Review the CLI help before running compute-intensive tuning options.

## API overview

The local API includes health and league metadata, fixtures and match predictions, model and calibration diagnostics, season simulations, what-if requests, refresh jobs, and the prediction ledger. The complete request and response schemas are available at `/api/docs` while the server is running.

## Tests

Run the complete test suite from the repository root:

```powershell
python -m pytest -q
```

The project test suite has been verified locally: **236 passed**. A Starlette/httpx deprecation warning may be emitted by the installed test-client dependency stack.

## Configuration

Edit `config.yaml` to adjust league settings, feature options, model parameters, evaluation windows, simulation settings, and the local server port. The default server binds to `127.0.0.1`, so it is not exposed to the network by default.

## Data and attribution

Match data is fetched from football-data.co.uk. ClubElo ratings are an optional external input. Respect the terms and availability of upstream data providers when using or redistributing their data. The repository does not include an explicit software license; all rights remain with their respective owners unless a license is added.

## Project status

This is a research-oriented, locally run project. Model quality depends on data coverage, provider availability, configuration, and evaluation design. Check the data and diagnostics for a run before interpreting its outputs.
