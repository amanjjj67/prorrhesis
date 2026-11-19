from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS teams(id INTEGER PRIMARY KEY, canon TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS team_aliases(alias TEXT NOT NULL, source TEXT NOT NULL,
  team_id INTEGER NOT NULL REFERENCES teams(id), PRIMARY KEY(alias, source));

CREATE TABLE IF NOT EXISTS matches(
  id INTEGER PRIMARY KEY, league TEXT NOT NULL, season TEXT NOT NULL,
  date TEXT NOT NULL,
  kickoff_utc TEXT,
  home_id INTEGER NOT NULL REFERENCES teams(id), away_id INTEGER NOT NULL REFERENCES teams(id),
  hg INTEGER NOT NULL, ag INTEGER NOT NULL, hthg INTEGER, htag INTEGER,
  res TEXT NOT NULL CHECK(res IN('H','D','A')),
  hs INTEGER, as_ INTEGER, hst INTEGER, ast INTEGER,
  hc INTEGER, ac INTEGER, hf INTEGER, af INTEGER, hy INTEGER, ay INTEGER, hr INTEGER, ar INTEGER,
  odds_open_h REAL, odds_open_d REAL, odds_open_a REAL, odds_open_src TEXT,
  odds_close_h REAL, odds_close_d REAL, odds_close_a REAL, odds_close_src TEXT,
  UNIQUE(league, date, home_id, away_id));
CREATE INDEX IF NOT EXISTS ix_matches_date ON matches(date);

CREATE TABLE IF NOT EXISTS match_xg(match_id INTEGER PRIMARY KEY REFERENCES matches(id), xg_h REAL, xg_a REAL);

CREATE TABLE IF NOT EXISTS fixtures(
  id INTEGER PRIMARY KEY, league TEXT NOT NULL, season TEXT NOT NULL, date TEXT NOT NULL, kickoff_utc TEXT,
  home_id INTEGER NOT NULL, away_id INTEGER NOT NULL,
  odds_open_h REAL, odds_open_d REAL, odds_open_a REAL, odds_open_src TEXT,
  fetched_at TEXT NOT NULL, UNIQUE(league, date, home_id, away_id));

CREATE TABLE IF NOT EXISTS elo_ext(team_id INTEGER NOT NULL, asof TEXT NOT NULL, elo REAL NOT NULL,
  PRIMARY KEY(team_id, asof));

CREATE TABLE IF NOT EXISTS oof_predictions(
  run_id TEXT NOT NULL, model TEXT NOT NULL, match_id INTEGER NOT NULL,
  block_start TEXT NOT NULL, pH REAL, pD REAL, pA REAL, status TEXT NOT NULL DEFAULT 'ok',
  PRIMARY KEY(run_id, model, match_id));

CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, created_at TEXT, config_hash TEXT, kind TEXT, notes TEXT);

CREATE TABLE IF NOT EXISTS ensemble_diagnostics(
  run_id TEXT NOT NULL, block_start TEXT NOT NULL, stage TEXT NOT NULL,
  method TEXT NOT NULL, score REAL, selected INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(run_id, block_start, stage, method));

CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  league TEXT NOT NULL, date TEXT NOT NULL, home_id INTEGER NOT NULL, away_id INTEGER NOT NULL,
  model TEXT NOT NULL, pH REAL NOT NULL, pD REAL NOT NULL, pA REAL NOT NULL,
  grid_json TEXT, config_hash TEXT NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL UNIQUE);
CREATE TRIGGER IF NOT EXISTS predictions_no_update BEFORE UPDATE ON predictions
  BEGIN SELECT RAISE(ABORT,'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS predictions_no_delete BEFORE DELETE ON predictions
  BEGIN SELECT RAISE(ABORT,'ledger is append-only'); END;

CREATE TABLE IF NOT EXISTS settlements(
  league TEXT, date TEXT, home_id INTEGER, away_id INTEGER, res TEXT, hg INTEGER, ag INTEGER,
  settled_at TEXT, PRIMARY KEY(league, date, home_id, away_id));

CREATE TABLE IF NOT EXISTS ingest_log(ts TEXT, source TEXT, kind TEXT, rows INTEGER, rejected INTEGER, note TEXT);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def open_db(db_path: str | Path, create: bool = True) -> sqlite3.Connection:
    conn = connect(db_path)
    if create:
        init_db(conn)
    return conn


def log_ingest(conn: sqlite3.Connection, source: str, kind: str, rows: int, rejected: int, note: str = "") -> None:
    from datetime import datetime, timezone

    conn.execute(
        "INSERT INTO ingest_log(ts, source, kind, rows, rejected, note) VALUES (?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), source, kind, rows, rejected, note),
    )
    conn.commit()
