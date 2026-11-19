"""ClubElo snapshots (optional source, spec section 4.2).

ClubElo gives every club a rating on one pan-European scale, which links the
five domestic leagues through European competitions that football-data does not
carry. The service is free but not always reachable, so every failure here is
contained: the caller disables the feature and the app keeps running.

Leakage rule: for a match on date d we use the snapshot dated d - 1, never d.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd

from ..config import Config
from ..names import SOURCE_CLUBELO, record_discovered, resolve

BASE_URL = "http://api.clubelo.com/{date}"
SNAPSHOT_COLUMNS = ["Rank", "Club", "Country", "Level", "Elo", "From", "To"]


@dataclass
class ClubEloReport:
    requested: int = 0
    fetched: int = 0
    cached: int = 0
    failed: int = 0
    rows: int = 0
    unresolved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def reachable(self) -> bool:
        return self.fetched > 0 or self.cached > 0


def snapshot_url(date: str) -> str:
    return BASE_URL.format(date=date)


def cache_dir(cfg: Config) -> Path:
    d = cfg.data_dir / "cache" / "clubelo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_snapshot(date: str, cfg: Config, force: bool = False,
                      timeout: float = 25.0) -> str | None:
    """Return the raw CSV for a date, or None when the service is unreachable."""
    path = cache_dir(cfg) / f"{date}.csv"
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")

    headers = {"User-Agent": str(cfg.data.get("user_agent", "prorrhesis/0.1"))}
    delay = float(cfg.data.get("request_delay_s", 1.0))
    time.sleep(delay)
    try:
        response = httpx.get(snapshot_url(date), headers=headers, timeout=timeout,
                             follow_redirects=True)
    except Exception as exc:
        raise ClubEloUnavailable(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code != 200 or not response.text.strip():
        raise ClubEloUnavailable(f"HTTP {response.status_code}")
    path.write_text(response.text, encoding="utf-8")
    return response.text


class ClubEloUnavailable(RuntimeError):
    """The ClubElo service could not be reached."""


def parse_snapshot(text: str) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    if "Club" not in df.columns or "Elo" not in df.columns:
        raise ValueError("unexpected ClubElo columns")
    df["Club"] = df["Club"].astype(str).str.strip()
    df["Elo"] = pd.to_numeric(df["Elo"], errors="coerce")
    return df


def upsert_elo(conn: sqlite3.Connection, snapshot: pd.DataFrame, asof: str) -> tuple[int, list[str]]:
    """Store snapshot rows against canonical team ids. Returns (rows, unresolved names)."""
    rows = 0
    unresolved: list[str] = []
    for club, elo in zip(snapshot["Club"], snapshot["Elo"]):
        if elo is None or pd.isna(elo):
            continue
        team_id = resolve(conn, str(club), SOURCE_CLUBELO)
        if team_id is None:
            if str(club) not in unresolved:
                unresolved.append(str(club))
            continue
        conn.execute(
            "INSERT INTO elo_ext(team_id, asof, elo) VALUES (?,?,?) "
            "ON CONFLICT(team_id, asof) DO UPDATE SET elo = excluded.elo",
            (int(team_id), asof, float(elo)),
        )
        rows += 1
    conn.commit()
    return rows, unresolved


def block_snapshot_dates(match_dates: list[str], block_days: int = 7) -> list[str]:
    """One snapshot per weekly block, dated the day before the block starts."""
    if not match_dates:
        return []
    stamps = sorted({pd.Timestamp(d) for d in match_dates})
    out: list[str] = []
    seen: set[str] = set()
    cur = stamps[0] - pd.Timedelta(days=int(stamps[0].weekday()))
    last = stamps[-1]
    while cur <= last:
        snap = (cur - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if snap not in seen:
            seen.add(snap)
            out.append(snap)
        cur += pd.Timedelta(days=block_days)
    return out


def fetch_snapshots(
    conn: sqlite3.Connection,
    cfg: Config,
    dates: list[str],
    force: bool = False,
    discovered: dict[str, dict[str, dict]] | None = None,
    max_consecutive_failures: int = 3,
) -> ClubEloReport:
    """Fetch and store snapshots. Raises nothing: failures land in the report."""
    report = ClubEloReport(requested=len(dates))
    consecutive = 0
    for position, date in enumerate(dates):
        path = cache_dir(cfg) / f"{date}.csv"
        was_cached = path.exists() and not force
        try:
            text = download_snapshot(date, cfg, force=force)
        except ClubEloUnavailable as exc:
            report.failed += 1
            consecutive += 1
            if len(report.errors) < 5:
                report.errors.append(f"{date}: {exc}")
            if consecutive >= max_consecutive_failures:
                report.failed += len(dates) - position - 1
                report.errors.append(
                    f"stopped after {consecutive} failed snapshots: the service did not answer")
                break
            continue
        consecutive = 0
        if text is None:
            report.failed += 1
            continue
        if was_cached:
            report.cached += 1
        else:
            report.fetched += 1
        try:
            snapshot = parse_snapshot(text)
        except Exception as exc:
            report.failed += 1
            if len(report.errors) < 5:
                report.errors.append(f"{date}: {type(exc).__name__}: {exc}")
            continue
        rows, unresolved = upsert_elo(conn, snapshot, date)
        report.rows += rows
        for name in unresolved:
            if name not in report.unresolved:
                report.unresolved.append(name)
        if discovered is not None:
            for club in snapshot["Club"]:
                record_discovered(discovered, SOURCE_CLUBELO, str(club), "", "")
    return report


def fetch_for_matches(conn: sqlite3.Connection, cfg: Config, matches: pd.DataFrame,
                      force: bool = False) -> ClubEloReport:
    """Fetch one snapshot per weekly block covering the given matches."""
    dates = block_snapshot_dates(list(matches["date"].astype(str).unique()),
                                 int(cfg.eval.get("block_days", 7)))
    return fetch_snapshots(conn, cfg, dates, force=force)
