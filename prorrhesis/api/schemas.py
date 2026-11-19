from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    config_hash: str
    database: dict[str, Any]
    sources: dict[str, Any]
    unresolved_names: dict[str, Any]
    column_coverage: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)


class LeagueInfo(BaseModel):
    code: str
    name: str
    current_season: str
    teams: int = 0
    matches: int = 0


class LeaguesResponse(BaseModel):
    leagues: list[LeagueInfo]


class FixturesResponse(BaseModel):
    league: str
    days: int = 0
    as_of: str = ""
    fixtures: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    method: str | None = None
    calibration_method: str | None = None
    note: str = ""


class MatchResponse(BaseModel):
    league: str
    date: str
    home: str = ""
    away: str = ""
    home_id: int | None = None
    away_id: int | None = None
    models: dict[str, Any] = Field(default_factory=dict)
    ensemble: dict[str, Any] = Field(default_factory=dict)
    ensemble_cal: dict[str, Any] = Field(default_factory=dict)
    grid: list[list[float]] | None = None
    top_scoreline: dict[str, Any] | None = None
    market_open: dict[str, Any] | None = None
    reasons: list[str] = Field(default_factory=list)
    disagreement: dict[str, Any] | None = None
    uncertainty: str | None = None
    lambdas: list[float] = Field(default_factory=list)
    method: str | None = None
    calibration_method: str | None = None
    error: str | None = None


class ArenaResponse(BaseModel):
    run_id: str | None = None
    run_kind: str = ""
    models: list[dict[str, Any]] = Field(default_factory=list)
    per_league: list[dict[str, Any]] = Field(default_factory=list)
    benchmark: str = "market_close"
    paired: dict[str, Any] | None = None
    window: dict[str, Any] = Field(default_factory=dict)
    holdout_broken: bool = False
    holdout_lock: dict[str, Any] | None = None
    note: str = ""


class CalibrationResponse(BaseModel):
    run_id: str | None = None
    model: str = "ensemble_cal"
    outcomes: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""


class TimeMachineResponse(BaseModel):
    run_id: str | None = None
    date: str = ""
    matches: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""


class SimulateResponse(BaseModel):
    league: str
    season: str | None = None
    n: int = 0
    disabled: bool = False
    reason: str = ""
    teams: list[dict[str, Any]] = Field(default_factory=list)
    remaining_fixtures: int = 0
    note: str = ""


class WhatifRequest(BaseModel):
    league: str
    home_id: int | None = None
    away_id: int | None = None
    home: str | None = None
    away: str | None = None
    attack: float = 0.0
    defence: float = 0.0
    side: str = "home"


class WhatifResponse(BaseModel):
    pH: float = 0.0
    pD: float = 0.0
    pA: float = 0.0
    lambdas: list[float] = Field(default_factory=list)
    adjustment: dict[str, Any] = Field(default_factory=dict)
    grid: list[list[float]] = Field(default_factory=list)
    note: str = ""
    league: str = ""
    home: str = ""
    away: str = ""
    stored: bool = False
    error: str | None = None


class LedgerResponse(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    chain_ok: bool = True
    chain: dict[str, Any] = Field(default_factory=dict)
    settled: int = 0
    open: int = 0
    running_log_loss: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class JobResponse(BaseModel):
    id: str
    kind: str = ""
    status: str = "pending"
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
