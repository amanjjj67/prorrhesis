from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from ..config import Config, load_config
from ..db import open_db
from ..ingest import footballdata as fd
from ..names import load_discovered, unresolved_aliases


def _count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return 0


def _last_ingest(conn: sqlite3.Connection, source: str) -> str | None:
    row = conn.execute(
        "SELECT ts FROM ingest_log WHERE source = ? ORDER BY ts DESC LIMIT 1", (source,)
    ).fetchone()
    return row["ts"] if row else None


def column_coverage(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT league, season, COUNT(*) n, "
        "SUM(CASE WHEN odds_close_h IS NOT NULL THEN 1 ELSE 0 END) close_odds, "
        "SUM(CASE WHEN odds_open_h IS NOT NULL THEN 1 ELSE 0 END) open_odds, "
        "SUM(CASE WHEN hs IS NOT NULL THEN 1 ELSE 0 END) shots, "
        "SUM(CASE WHEN hc IS NOT NULL THEN 1 ELSE 0 END) cards, "
        "SUM(CASE WHEN hthg IS NOT NULL THEN 1 ELSE 0 END) ht "
        "FROM matches GROUP BY league, season ORDER BY league, season"
    ).fetchall()
    out = []
    for r in rows:
        n = max(int(r["n"]), 1)
        out.append({
            "league": r["league"], "season": r["season"], "matches": int(r["n"]),
            "open_odds_pct": round(100.0 * int(r["open_odds"]) / n, 1),
            "close_odds_pct": round(100.0 * int(r["close_odds"]) / n, 1),
            "shots_pct": round(100.0 * int(r["shots"]) / n, 1),
            "cards_pct": round(100.0 * int(r["cards"]) / n, 1),
            "ht_pct": round(100.0 * int(r["ht"]) / n, 1),
        })
    return out


def build_health(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    conn = open_db(cfg.db_path, create=False)
    try:
        warnings: list[str] = []
        discovered = load_discovered(cfg.discovered_names_path)
        unresolved = unresolved_aliases(conn, discovered)
        if unresolved:
            warnings.append(
                f"{len(unresolved)} unresolved club name(s) - run 'python -m prorrhesis names suggest'")

        use_clubelo = bool(cfg.features.get("use_clubelo", True))
        use_xg = bool(cfg.features.get("use_understat_xg", False))
        if not use_xg:
            warnings.append("Understat xG is off (default); the xG proxy is used instead")
        clubelo_rows = _count(conn, "elo_ext")
        clubelo_snapshots = 0
        clubelo_latest = None
        clubelo_error = None
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT asof) n, MAX(asof) latest FROM elo_ext").fetchone()
            clubelo_snapshots = int(row["n"])
            clubelo_latest = row["latest"]
        except sqlite3.Error:
            pass
        log = conn.execute(
            "SELECT note FROM ingest_log WHERE source = 'clubelo' ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if log and log["note"]:
            clubelo_error = log["note"]
        if use_clubelo and clubelo_rows == 0:
            if clubelo_error:
                warnings.append(f"ClubElo unreachable, feature disabled for this run: {clubelo_error}")
            else:
                warnings.append("ClubElo enabled but not yet fetched - run 'python -m prorrhesis clubelo fetch'")
        if clubelo_rows and use_clubelo:
            warnings.append(f"ClubElo in use ({clubelo_rows} ratings from {clubelo_snapshots} snapshots)")

        current = fd.current_season_code()
        db_matches = _count(conn, "matches")
        coverage = column_coverage(conn)
        expected_seasons = len(coverage)
        if expected_seasons and len(coverage) < 5 * expected_seasons // 5:
            warnings.append("partial season coverage")

        return {
            "status": "ok" if db_matches else "empty",
            "config_hash": cfg.config_hash,
            "database": {
                "path": str(cfg.db_path),
                "matches": db_matches,
                "teams": _count(conn, "teams"),
                "aliases": _count(conn, "team_aliases"),
                "fixtures": _count(conn, "fixtures"),
                "runs": _count(conn, "runs"),
                "oof_predictions": _count(conn, "oof_predictions"),
                "ledger_rows": _count(conn, "predictions"),
                "settlements": _count(conn, "settlements"),
            },
            "sources": {
                "footballdata": {
                    "status": "ok" if db_matches else "no data",
                    "last_ingest": _last_ingest(conn, "footballdata"),
                    "current_season": current,
                    "raw_files": len(list(cfg.raw_dir.glob("*.csv"))) if cfg.raw_dir.exists() else 0,
                },
                "clubelo": {
                    "status": ("ok" if clubelo_rows else
                               "enabled, not fetched" if use_clubelo else "disabled"),
                    "rows": clubelo_rows,
                    "snapshots": clubelo_snapshots,
                    "latest_snapshot": clubelo_latest,
                    "last_error": clubelo_error,
                },
                "understat": {
                    "status": "enabled" if use_xg else "disabled",
                    "rows": _count(conn, "match_xg"),
                },
            },
            "unresolved_names": {
                "count": len(unresolved),
                "sample": [f"{n} [{s}]" for n, s in unresolved[:20]],
            },
            "column_coverage": coverage,
            "warnings": warnings,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    finally:
        conn.close()


def build_leagues(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    conn = open_db(cfg.db_path, create=False)
    try:
        current = fd.current_season_code()
        out = []
        for code, name in cfg.leagues.items():
            n_matches = int(conn.execute(
                "SELECT COUNT(*) FROM matches WHERE league = ?", (code,)).fetchone()[0])
            n_teams = int(conn.execute(
                "SELECT COUNT(DISTINCT home_id) FROM matches WHERE league = ?", (code,)).fetchone()[0])
            out.append({"code": code, "name": name, "current_season": current,
                        "teams": n_teams, "matches": n_matches})
        return {"leagues": out}
    finally:
        conn.close()
