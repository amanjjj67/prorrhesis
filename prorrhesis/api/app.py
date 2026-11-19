from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config, load_config
from .arena import build_arena, build_calibration, build_time_machine
from .health import build_health, build_leagues
from .jobs import MANAGER
from .ledger import build_ledger
from .live import LiveService
from .predictions import build_fixtures, build_match
from .schemas import (
    ArenaResponse,
    CalibrationResponse,
    FixturesResponse,
    HealthResponse,
    JobResponse,
    LedgerResponse,
    LeaguesResponse,
    MatchResponse,
    SimulateResponse,
    TimeMachineResponse,
    WhatifRequest,
    WhatifResponse,
)
from .sim import build_simulation, build_whatif

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="PRORRHESIS", version="0.1.0", docs_url="/api/docs")
    service = LiveService(cfg)
    app.state.service = service
    app.state.config = cfg

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> dict:
        return build_health(cfg)

    @app.get("/api/leagues", response_model=LeaguesResponse)
    def leagues() -> dict:
        return build_leagues(cfg)

    @app.get("/api/fixtures", response_model=FixturesResponse)
    def fixtures(league: str = Query(...), days: int = Query(7, ge=1, le=60)) -> dict:
        if league not in cfg.leagues:
            raise HTTPException(status_code=404, detail=f"unknown league: {league}")
        return build_fixtures(cfg, service, league, days)

    @app.get("/api/match", response_model=MatchResponse)
    def match(league: str = Query(...), date: str = Query(...),
              home: str = Query(...), away: str = Query(...)) -> dict:
        if league not in cfg.leagues:
            raise HTTPException(status_code=404, detail=f"unknown league: {league}")
        return build_match(cfg, service, league, date, home, away)

    @app.get("/api/arena", response_model=ArenaResponse)
    def arena(run: str | None = Query(None)) -> dict:
        return build_arena(cfg, service.connection(), run)

    @app.get("/api/calibration", response_model=CalibrationResponse)
    def calibration(model: str = Query("ensemble_cal"), run: str | None = Query(None)) -> dict:
        return build_calibration(service.connection(), model, run)

    @app.get("/api/time-machine", response_model=TimeMachineResponse)
    def time_machine(date: str = Query(...), run: str | None = Query(None)) -> dict:
        return build_time_machine(service.connection(), date, run)

    @app.get("/api/simulate", response_model=SimulateResponse)
    def simulate(league: str = Query(...), n: int = Query(10000, ge=100, le=200000)) -> dict:
        if league not in cfg.leagues:
            raise HTTPException(status_code=404, detail=f"unknown league: {league}")
        return build_simulation(cfg, service, league, n)

    @app.post("/api/whatif", response_model=WhatifResponse)
    def whatif(payload: WhatifRequest) -> dict:
        return build_whatif(cfg, service, payload.model_dump())

    @app.get("/api/ledger", response_model=LedgerResponse)
    def ledger(limit: int = Query(200, ge=1, le=5000)) -> dict:
        return build_ledger(service.connection(), limit)

    @app.post("/api/refresh", response_model=JobResponse)
    def refresh() -> dict:
        job = MANAGER.submit("refresh", _refresh_task(cfg))
        return job.as_dict()

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    def job(job_id: str) -> dict:
        found = MANAGER.get(job_id)
        if found is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return found.as_dict()

    @app.get("/api/jobs", response_model=list[JobResponse])
    def jobs() -> list[dict]:
        return [j.as_dict() for j in MANAGER.list()]

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    @app.exception_handler(Exception)
    async def unhandled(request, exc):  # pragma: no cover - safety net
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    return app


def _refresh_task(cfg: Config):
    """The work behind `POST /api/refresh`: ingest, then predict and log."""

    def run(job) -> dict:
        import argparse

        from ..__main__ import cmd_ingest, cmd_predict
        from ..db import open_db

        job.step("ingest", ok=True, detail="starting")
        # the ingest must use this app's config, not the default one
        args = argparse.Namespace(config=str(cfg.root / "config.yaml"), force=False,
                                  discover=False)
        try:
            cmd_ingest(args)
            job.step("ingest", ok=True, detail="done")
        except Exception as exc:
            job.step("ingest", ok=False, detail=f"{type(exc).__name__}: {exc}")

        job.step("predict", ok=True, detail="starting")
        conn = open_db(cfg.db_path)
        try:
            from ..predict import run_live_predictions

            report = run_live_predictions(cfg, conn)
            job.step("predict", ok=True,
                     detail=f"{report['appended']} appended, {report['skipped']} skipped, "
                            f"{report['late']} late")
            return report
        except Exception as exc:
            job.step("predict", ok=False, detail=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            conn.close()

    return run


app = create_app()
