"""API tests (spec section 16).

Every endpoint must answer with schema-valid JSON on a tiny fixture database,
and the what-if tool must never write to the ledger.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from prorrhesis.api.app import create_app
from prorrhesis.api.schemas import (
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
    WhatifResponse,
)
from prorrhesis.config import load_config
from prorrhesis.db import open_db
from prorrhesis.names import SOURCE_FOOTBALLDATA, add_alias, ensure_team

from conftest import LEAGUE, TEAMS, build_history

RUN_ID = "tuning_testrun"
WEEKS = 80


def _write_config(root) -> None:
    """A one-league config so the test never touches the real project files."""
    raw = {
        "leagues": {LEAGUE: "Premier League"},
        "first_season": "1617",
        "data": {"user_agent": "prorrhesis-test", "request_delay_s": 0.0,
                 "current_season_refetch_min": 60},
        "features": {"use_clubelo": False, "use_understat_xg": False,
                     "ewma_half_lives": [5, 12], "venue_half_life": 8, "prior_n": 4},
        "elo": {"K": 20, "H": {LEAGUE: 65}, "rho_season": 0.25, "promoted_penalty": 50},
        "dc": {"window_seasons": 3, "half_life_days": 365, "gamma": 0.5, "max_goals": 10},
        "xgp": {"w": 0.5},
        "gbm": {"refit_weeks": 4, "early_stop_frac": 0.10},
        "stack": {"min_rows": 800, "recency_half_life_blocks": 40,
                  "methods": ["logit", "logpool", "equal"]},
        "calibrate": {"methods": ["none", "temperature", "dirichlet", "isotonic"],
                      "min_rows": {"dirichlet": 1500, "isotonic": 3000}},
        "eval": {"eval_start": "2017-08-01", "block_days": 7, "holdout_seasons": 2,
                 "bootstrap_resamples": 500, "seed": 42},
        "sim": {"n": 1000, "relegated": {LEAGUE: 3}, "top4": 4},
        "server": {"host": "127.0.0.1", "port": 8000},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")


def _insert_matches(conn, history: pd.DataFrame) -> None:
    payload = []
    for r in history.to_dict("records"):
        payload.append((
            LEAGUE, r["season"], r["date"], r["home_id"], r["away_id"], r["hg"], r["ag"],
            r["res"], r["hs"], r["as_"], r["hst"], r["ast"],
            float(r["odds_open_h"]), float(r["odds_open_d"]), float(r["odds_open_a"]),
            float(r["odds_open_h"]) * 1.02, float(r["odds_open_d"]) * 0.99,
            float(r["odds_open_a"]) * 1.01,
        ))
    conn.executemany(
        "INSERT INTO matches(league, season, date, home_id, away_id, hg, ag, res, hs, as_, "
        "hst, ast, odds_open_h, odds_open_d, odds_open_a, odds_close_h, odds_close_d, "
        "odds_close_a) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        payload,
    )
    conn.commit()


def _insert_oof(conn, history: pd.DataFrame) -> None:
    """A finished run with out-of-fold rows, so the arena has something to show."""
    conn.execute(
        "INSERT INTO runs(run_id, created_at, config_hash, kind, notes) VALUES (?,?,?,?,?)",
        (RUN_ID, "2020-01-01T00:00:00+00:00", "testhash", "tuning",
         "2017-08-01..2018-01-15"),
    )
    triples = {
        "freq": (0.44, 0.26, 0.30),
        "market_close": (0.46, 0.25, 0.29),
        "ensemble": (0.45, 0.255, 0.295),
        "ensemble_cal": (0.45, 0.255, 0.295),
    }
    ids = [int(row["id"]) for row in conn.execute(
        "SELECT id FROM matches WHERE league = ? ORDER BY date, home_id", (LEAGUE,)).fetchall()]
    blocks = []
    for record in history.to_dict("records"):
        day = pd.Timestamp(record["date"])
        blocks.append((day - pd.Timedelta(days=day.weekday())).strftime("%Y-%m-%d"))

    for model, (pH, pD, pA) in triples.items():
        rows = [(RUN_ID, model, mid, block, pH, pD, pA, "ok")
                for mid, block in zip(ids, blocks)]
        conn.executemany(
            "INSERT INTO oof_predictions(run_id, model, match_id, block_start, pH, pD, pA, "
            "status) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def _insert_fixture(conn) -> str:
    """One upcoming fixture, dated today, so the fixtures screen has a row."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO fixtures(id, league, season, date, kickoff_utc, home_id, away_id, "
        "odds_open_h, odds_open_d, odds_open_a, odds_open_src, fetched_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (9001, LEAGUE, "1718", today, f"{today}T15:00:00", 1, 2,
         1.80, 3.40, 4.20, "test", today),
    )
    conn.commit()
    return today


@pytest.fixture()
def cfg(tmp_path):
    _write_config(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    return load_config(tmp_path / "config.yaml")


@pytest.fixture()
def seeded_db(cfg):
    conn = open_db(cfg.db_path)
    for i, name in enumerate(TEAMS, start=1):
        tid = ensure_team(conn, name.lower().replace(" ", "-"))
        add_alias(conn, name, SOURCE_FOOTBALLDATA, tid)
    history = build_history(n_weeks=WEEKS)
    _insert_matches(conn, history)
    _insert_oof(conn, history)
    _insert_fixture(conn)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture()
def client(cfg, seeded_db):
    app = create_app(cfg)
    with TestClient(app) as test_client:
        yield test_client
    app.state.service.close()


# -- the endpoints ---------------------------------------------------------

def test_health_returns_schema_valid_json(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    payload = r.json()
    HealthResponse.model_validate(payload)
    assert payload["database"]["matches"] == WEEKS * 4
    assert payload["config_hash"]


def test_leagues_returns_schema_valid_json(client):
    r = client.get("/api/leagues")
    assert r.status_code == 200
    payload = r.json()
    LeaguesResponse.model_validate(payload)
    codes = [entry["code"] for entry in payload["leagues"]]
    assert LEAGUE in codes


def test_fixtures_returns_valid_probability_triples(client):
    r = client.get(f"/api/fixtures?league={LEAGUE}&days=7")
    assert r.status_code == 200
    payload = r.json()
    FixturesResponse.model_validate(payload)
    assert payload["count"] >= 1
    for row in payload["fixtures"]:
        total = row["pH"] + row["pD"] + row["pA"]
        assert abs(total - 1.0) < 1e-6
        assert all(0.0 <= row[k] <= 1.0 for k in ("pH", "pD", "pA"))


def test_fixtures_rejects_unknown_league(client):
    r = client.get("/api/fixtures?league=ZZ9&days=7")
    assert r.status_code == 404


def test_match_returns_schema_valid_json(client):
    r = client.get(f"/api/match?league={LEAGUE}&date=2026-01-01&home=Alpha FC&away=Beta FC")
    assert r.status_code == 200
    payload = r.json()
    MatchResponse.model_validate(payload)
    assert payload["home_id"] == 1 and payload["away_id"] == 2
    assert payload["models"], "the match detail must carry the per-model triples"
    assert payload["reasons"]
    total = payload["ensemble"]["pH"] + payload["ensemble"]["pD"] + payload["ensemble"]["pA"]
    assert abs(total - 1.0) < 1e-6


def test_arena_returns_schema_valid_json(client):
    r = client.get(f"/api/arena?run={RUN_ID}")
    assert r.status_code == 200
    payload = r.json()
    ArenaResponse.model_validate(payload)
    assert payload["run_id"] == RUN_ID
    names = {row["model"] for row in payload["models"]}
    assert {"ensemble", "market_close"} <= names
    assert payload["paired"] and payload["paired"]["n"] > 0


def test_calibration_returns_schema_valid_json(client):
    r = client.get(f"/api/calibration?model=ensemble_cal&run={RUN_ID}")
    assert r.status_code == 200
    payload = r.json()
    CalibrationResponse.model_validate(payload)
    assert [o["outcome"] for o in payload["outcomes"]] == ["H", "D", "A"]
    assert payload["outcomes"][0]["bins"]


def test_time_machine_returns_schema_valid_json(client, seeded_db):
    row = seeded_db.execute(
        "SELECT date FROM matches WHERE league = ? ORDER BY date LIMIT 1", (LEAGUE,)).fetchone()
    date = str(row["date"])
    r = client.get(f"/api/time-machine?date={date}&run={RUN_ID}")
    assert r.status_code == 200
    payload = r.json()
    TimeMachineResponse.model_validate(payload)
    assert payload["matches"], "the replay must find the matches of that date"
    first = payload["matches"][0]
    assert first["score"] and first["result"] in ("H", "D", "A")
    assert first["loss"] is None or first["loss"] >= 0.0


def test_simulate_returns_schema_valid_json(client):
    r = client.get(f"/api/simulate?league={LEAGUE}&n=200")
    assert r.status_code == 200
    payload = r.json()
    SimulateResponse.model_validate(payload)
    assert payload["teams"], "the simulation must rank the clubs"
    for team in payload["teams"]:
        assert 0.0 <= team["p_title"] <= 1.0
        # expected_points includes points already earned + expected from remaining fixtures
        # Upper bound: 38 games * 3 pts = 114 max, plus safety margin for edge cases
        assert 0.0 <= team["expected_points"] <= 150.0


def test_whatif_returns_schema_valid_json(client):
    r = client.post("/api/whatif", json={"league": LEAGUE, "home_id": 1, "away_id": 2,
                                          "attack": 0.2, "defence": 0.0, "side": "home"})
    assert r.status_code == 200
    payload = r.json()
    WhatifResponse.model_validate(payload)
    assert payload["stored"] is False
    total = payload["pH"] + payload["pD"] + payload["pA"]
    assert abs(total - 1.0) < 1e-6


def test_ledger_returns_schema_valid_json(client):
    r = client.get("/api/ledger?limit=50")
    assert r.status_code == 200
    payload = r.json()
    LedgerResponse.model_validate(payload)
    assert payload["chain_ok"] is True
    assert payload["count"] == len(payload["rows"])


def test_jobs_list_returns_schema_valid_json(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    for entry in payload:
        JobResponse.model_validate(entry)


def test_unknown_job_is_a_404(client):
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "PRORRHESIS" in r.text


# -- invariants ------------------------------------------------------------

def test_whatif_writes_nothing_to_the_ledger(client, seeded_db):
    before = seeded_db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    r = client.post("/api/whatif", json={"league": LEAGUE, "home_id": 1, "away_id": 3,
                                          "attack": -0.3, "defence": 0.15, "side": "away"})
    assert r.status_code == 200
    assert r.json()["stored"] is False
    after = seeded_db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    assert after == before


def test_reading_the_app_does_not_write_to_the_ledger(client, seeded_db):
    before = seeded_db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    client.get(f"/api/fixtures?league={LEAGUE}&days=7")
    client.get(f"/api/match?league={LEAGUE}&date=2026-01-01&home=Alpha FC&away=Beta FC")
    client.get(f"/api/simulate?league={LEAGUE}&n=200")
    client.get("/api/ledger?limit=10")
    after = seeded_db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    assert after == before
