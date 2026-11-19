"""Append-only prediction ledger (spec section 13).

Every live prediction is appended to a hash chain. The chain makes the history
auditable: if a row is edited after the fact, `verify` finds the break and
names the row. The table has triggers that refuse UPDATE and DELETE, so the
history cannot be rewritten even by accident.

Rules:
  * a row may only be appended before kick-off; later rows are rejected and
    logged as `late_prediction`;
  * at most one row per (fixture, model, config hash, day) - predicting twice
    in a day with unchanged data is a no-op;
  * settlements are written in a separate table and never touch `predictions`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

ZERO_HASH = "0" * 64

HASH_COLUMNS = ("id", "prev_hash", "hash")
ROW_COLUMNS = ("created_at", "league", "date", "home_id", "away_id", "model",
               "pH", "pD", "pA", "grid_json", "config_hash")


def row_hash(prev_hash: str, row: dict) -> str:
    """Hash of a row chained onto `prev_hash` (spec section 13)."""
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev_hash + payload).encode()).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _day(created_at: str) -> str:
    return str(created_at)[:10]


def _kickoff_deadline(kickoff_utc: str | None, date: str) -> datetime:
    if kickoff_utc:
        text = str(kickoff_utc).replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            moment = datetime.fromisoformat(f"{date}T00:00:00+00:00")
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)
    return datetime.fromisoformat(f"{date}T00:00:00+00:00")


@dataclass
class AppendResult:
    appended: int = 0
    skipped: int = 0
    late: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"appended": self.appended, "skipped": self.skipped,
                "late": self.late, "errors": list(self.errors)}


def last_hash(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT hash FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
    return str(row["hash"]) if row and row["hash"] else ZERO_HASH


def is_duplicate(conn: sqlite3.Connection, league: str, date: str, home_id: int,
                 away_id: int, model: str, config_hash: str, created_at: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM predictions WHERE league=? AND date=? AND home_id=? AND away_id=? "
        "AND model=? AND config_hash=? AND substr(created_at,1,10)=? LIMIT 1",
        (league, date, int(home_id), int(away_id), model, config_hash, _day(created_at)),
    ).fetchone()
    return row is not None


def append_prediction(
    conn: sqlite3.Connection,
    *,
    league: str,
    date: str,
    home_id: int,
    away_id: int,
    model: str,
    pH: float,
    pD: float,
    pA: float,
    config_hash: str,
    grid_json: str | None = None,
    kickoff_utc: str | None = None,
    created_at: datetime | None = None,
) -> str | None:
    """Append one prediction to the chain.

    Returns the new hash, or None when the row was skipped as a duplicate or
    rejected because kick-off had already passed.
    """
    moment = (created_at or _utc_now()).astimezone(timezone.utc)
    stamp = moment.isoformat(timespec="seconds")

    if moment >= _kickoff_deadline(kickoff_utc, date):
        _log_late(conn, league, date, home_id, away_id, model, stamp)
        return None

    if is_duplicate(conn, league, date, home_id, away_id, model, config_hash, stamp):
        return None

    prev = last_hash(conn)
    row = {
        "created_at": stamp,
        "league": str(league),
        "date": str(date),
        "home_id": int(home_id),
        "away_id": int(away_id),
        "model": str(model),
        "pH": float(pH),
        "pD": float(pD),
        "pA": float(pA),
        "grid_json": grid_json,
        "config_hash": str(config_hash),
    }
    digest = row_hash(prev, row)
    conn.execute(
        "INSERT INTO predictions(created_at, league, date, home_id, away_id, model, "
        "pH, pD, pA, grid_json, config_hash, prev_hash, hash) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (row["created_at"], row["league"], row["date"], row["home_id"], row["away_id"],
         row["model"], row["pH"], row["pD"], row["pA"], row["grid_json"],
         row["config_hash"], prev, digest),
    )
    conn.commit()
    return digest


def _log_late(conn: sqlite3.Connection, league: str, date: str, home_id: int,
              away_id: int, model: str, stamp: str) -> None:
    from .db import log_ingest

    log_ingest(conn, "ledger", "late_prediction", 0, 1,
               f"{league} {date} {home_id}-{away_id} {model} at {stamp}")
    conn.commit()


def verify(conn: sqlite3.Connection) -> dict:
    """Recompute the whole chain. Returns the first break, if any."""
    rows = conn.execute(
        "SELECT id, created_at, league, date, home_id, away_id, model, pH, pD, pA, "
        "grid_json, config_hash, prev_hash, hash FROM predictions ORDER BY id"
    ).fetchall()
    prev = ZERO_HASH
    for r in rows:
        if str(r["prev_hash"]) != prev:
            return {"ok": False, "row_id": int(r["id"]), "reason": "prev_hash does not match",
                    "checked": int(r["id"]), "total": len(rows)}
        row = {col: r[col] for col in ROW_COLUMNS}
        expected = row_hash(prev, row)
        if expected != str(r["hash"]):
            return {"ok": False, "row_id": int(r["id"]), "reason": "hash does not match",
                    "checked": int(r["id"]), "total": len(rows)}
        prev = str(r["hash"])
    return {"ok": True, "row_id": None, "reason": None, "checked": len(rows), "total": len(rows)}


def settle(conn: sqlite3.Connection, matches) -> int:
    """Copy finished results into `settlements`. Never touches `predictions`."""
    count = 0
    stamp = _utc_now().isoformat(timespec="seconds")
    for m in matches:
        conn.execute(
            "INSERT INTO settlements(league, date, home_id, away_id, res, hg, ag, settled_at) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(league, date, home_id, away_id) DO UPDATE SET "
            "res=excluded.res, hg=excluded.hg, ag=excluded.ag, settled_at=excluded.settled_at",
            (str(m["league"]), str(m["date"]), int(m["home_id"]), int(m["away_id"]),
             str(m["res"]), int(m["hg"]), int(m["ag"]), stamp),
        )
        count += 1
    conn.commit()
    return count


def rows(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    """Recent ledger rows with their settlement."""
    out = []
    for r in conn.execute(
        "SELECT p.id, p.created_at, p.league, p.date, p.home_id, p.away_id, p.model, "
        "p.pH, p.pD, p.pA, p.config_hash, p.prev_hash, p.hash, s.res, s.hg, s.ag "
        "FROM predictions p LEFT JOIN settlements s "
        "ON s.league=p.league AND s.date=p.date AND s.home_id=p.home_id AND s.away_id=p.away_id "
        "ORDER BY p.id DESC LIMIT ?", (int(limit),)
    ):
        out.append({
            "id": int(r["id"]),
            "created_at": r["created_at"],
            "league": r["league"],
            "date": r["date"],
            "home_id": int(r["home_id"]),
            "away_id": int(r["away_id"]),
            "model": r["model"],
            "pH": r["pH"],
            "pD": r["pD"],
            "pA": r["pA"],
            "config_hash": r["config_hash"],
            "prev_hash": r["prev_hash"],
            "hash": r["hash"],
            "result": r["res"],
            "hg": r["hg"],
            "ag": r["ag"],
        })
    return out
