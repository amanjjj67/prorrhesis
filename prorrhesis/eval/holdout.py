"""Holdout discipline (spec section 10.5).

Tuning happens on the tuning window; the headline window is evaluated once
against a frozen config. If the config changes afterwards the arena shows
`holdout_broken` (invariant I8).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Config
from ..ingest.footballdata import current_season_code


@dataclass(frozen=True)
class Windows:
    eval_start: str
    tuning_start: str
    tuning_end: str
    headline_start: str
    headline_end: str


def season_windows(conn: sqlite3.Connection, cfg: Config) -> Windows:
    """Split the history into the tuning window and the frozen headline window."""
    rows = pd.read_sql_query("SELECT DISTINCT season, date FROM matches ORDER BY season, date", conn)
    if rows.empty:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        eval_start = str(cfg.eval.get("eval_start", "2017-08-01"))
        return Windows(eval_start, eval_start, today, today, today)

    by_season = rows.groupby("season")["date"].min()
    season_last = rows.groupby("season")["date"].max()
    current = current_season_code()
    completed = sorted(s for s in by_season.index if str(s) < current)
    holdout_n = int(cfg.eval.get("holdout_seasons", 2))
    eval_start = str(cfg.eval.get("eval_start", "2017-08-01"))
    last_date = str(rows["date"].max())

    if len(completed) <= holdout_n:
        # not enough history to hold anything back: tune and report on the same window
        return Windows(eval_start, eval_start, last_date, last_date, last_date)

    holdout = completed[-holdout_n:]
    headline_start = str(by_season.loc[holdout[0]])
    headline_end = str(season_last.loc[holdout[-1]])
    # the tuning window stops before the first headline block, so no tuning
    # block can overlap the headline window
    monday = pd.Timestamp(headline_start) - pd.Timedelta(days=int(pd.Timestamp(headline_start).weekday()))
    tuning_end = (monday - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return Windows(eval_start, eval_start, tuning_end, headline_start, headline_end)


def lock_path(cfg: Config) -> Path:
    return cfg.root / "holdout.lock"


def write_lock(cfg: Config, headline_start: str) -> Path:
    payload = {
        "config_hash": cfg.config_hash,
        "declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headline_window_start": headline_start,
    }
    p = lock_path(cfg)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def read_lock(cfg: Config) -> dict | None:
    p = lock_path(cfg)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def holdout_broken(cfg: Config, conn: sqlite3.Connection) -> bool:
    """True when a headline run exists but the config no longer matches the lock."""
    lock = read_lock(cfg)
    if not lock:
        return False
    row = conn.execute(
        "SELECT config_hash FROM runs WHERE kind = 'headline' LIMIT 1").fetchone()
    if row is None:
        return False
    locked = str(lock.get("config_hash"))
    if str(cfg.config_hash) != locked:
        return True
    return str(row["config_hash"]) != locked
