"""Upcoming fixtures from football-data.co.uk/fixtures.csv (spec section 4.1)."""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from ..config import Config
from ..names import SOURCE_FOOTBALLDATA, resolve
from .footballdata import LEAGUE_TZ, _kickoff_utc, _pick_odds, _read_csv, season_from_date

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def download_fixtures(cfg: Config, force: bool = False) -> tuple[Path | None, str]:
    cfg.ensure_dirs()
    path = cfg.cache_dir / "fixtures.csv"
    if path.exists() and not force:
        age_min = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 60.0
        if age_min < 30:
            return path, "cached"
    headers = {"User-Agent": str(cfg.data.get("user_agent", "prorrhesis/0.1 (personal research)"))}
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(FIXTURES_URL)
    except httpx.HTTPError as exc:
        return None, f"error {type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return None, f"error HTTP {resp.status_code}"
    path.write_bytes(resp.content)
    return path, "downloaded"


def parse_fixtures(path: Path, leagues: list[str]) -> pd.DataFrame:
    df, _ = _read_csv(path)
    if "Div" not in df.columns:
        return pd.DataFrame()
    df = df[df["Div"].isin(leagues)].copy()
    if df.empty:
        return df
    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df["date"] = dates.dt.strftime("%Y-%m-%d")
    df["season"] = dates.map(lambda d: season_from_date(d) if pd.notna(d) else None)
    out = pd.DataFrame({
        "league": df["Div"].astype("string"),
        "season": df["season"],
        "date": df["date"],
        "home_name": df["HomeTeam"].astype("string").str.strip(),
        "away_name": df["AwayTeam"].astype("string").str.strip(),
    })
    kickoff = pd.Series([None] * len(df), index=df.index, dtype=object)
    for league in out["league"].dropna().unique():
        idx = out.index[out["league"] == league]
        sub = df.loc[idx].copy()
        sub["league"] = league
        kickoff.loc[idx] = _kickoff_utc(sub, league).values
    out["kickoff_utc"] = kickoff.values
    h, d, a, src, _ = _pick_odds(df, "open")
    out["odds_open_h"], out["odds_open_d"], out["odds_open_a"], out["odds_open_src"] = h, d, a, src
    out = out.dropna(subset=["date", "home_name", "away_name"])
    return out.reset_index(drop=True)


def upsert_fixtures(conn: sqlite3.Connection, frame: pd.DataFrame) -> dict:
    stats = {"rows": len(frame), "inserted": 0, "rejected": 0, "unresolved": []}
    if frame.empty:
        return stats
    cache: dict[str, int | None] = {}

    def resolve_name(name) -> int | None:
        if name not in cache:
            cache[name] = resolve(conn, str(name), SOURCE_FOOTBALLDATA)
        return cache[name]

    payload = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for _, row in frame.iterrows():
        h = resolve_name(row["home_name"])
        a = resolve_name(row["away_name"])
        if h is None or a is None:
            stats["rejected"] += 1
            missing = str(row["home_name"] if h is None else row["away_name"])
            if missing not in stats["unresolved"]:
                stats["unresolved"].append(missing)
            continue
        payload.append({
            "league": row["league"], "season": row["season"], "date": row["date"],
            "kickoff_utc": row["kickoff_utc"], "home_id": h, "away_id": a,
            "odds_open_h": _f(row["odds_open_h"]), "odds_open_d": _f(row["odds_open_d"]),
            "odds_open_a": _f(row["odds_open_a"]), "odds_open_src": _s(row["odds_open_src"]),
            "fetched_at": now,
        })
    if payload:
        sql = (
            "INSERT INTO fixtures (league, season, date, kickoff_utc, home_id, away_id, "
            "odds_open_h, odds_open_d, odds_open_a, odds_open_src, fetched_at) "
            "VALUES (:league,:season,:date,:kickoff_utc,:home_id,:away_id,"
            ":odds_open_h,:odds_open_d,:odds_open_a,:odds_open_src,:fetched_at) "
            "ON CONFLICT(league, date, home_id, away_id) DO UPDATE SET "
            "season=excluded.season, kickoff_utc=excluded.kickoff_utc, "
            "odds_open_h=excluded.odds_open_h, odds_open_d=excluded.odds_open_d, "
            "odds_open_a=excluded.odds_open_a, odds_open_src=excluded.odds_open_src, "
            "fetched_at=excluded.fetched_at"
        )
        conn.executemany(sql, payload)
        conn.commit()
        stats["inserted"] = len(payload)
    return stats


def _f(v) -> float | None:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _s(v) -> str | None:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return str(v)
