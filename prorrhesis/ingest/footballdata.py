"""football-data.co.uk adapter (spec section 4.1).

Downloads historical and current-season CSVs, parses them tolerantly (missing
columns, mixed encodings, both date formats), and upserts matches idempotently.
"""

from __future__ import annotations

import io
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from ..config import Config
from ..names import SOURCE_FOOTBALLDATA, resolve

BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUE_TZ = {
    "E0": "Europe/London",
    "SP1": "Europe/Madrid",
    "D1": "Europe/Berlin",
    "I1": "Europe/Rome",
    "F1": "Europe/Paris",
}

RESULT_COLS = {"HomeTeam": "home_name", "AwayTeam": "away_name", "FTHG": "hg", "FTAG": "ag"}
HT_COLS = {"HTHG": "hthg", "HTAG": "htag"}
STAT_COLS = {
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
    "HF": "hf", "AF": "af", "HC": "hc", "AC": "ac",
    "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
}
ODDS_CHAINS = {
    "open": (("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A")),
    "close": (("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")),
}
ODDS_SRC_NAMES = {"Avg": "Avg", "PS": "PS", "B365": "B365", "PSC": "PSC", "AvgC": "AvgC", "B365C": "B365C"}


@dataclass
class DownloadResult:
    path: Path | None
    status: str
    detail: str = ""


@dataclass
class ParseReport:
    rows_in: int = 0
    rows_kept: int = 0
    dropped_no_teams: int = 0
    dropped_unplayed: int = 0
    bad_dates: int = 0
    res_disagreements: int = 0
    notes: list[str] = field(default_factory=list)


def current_season_code(today: datetime | None = None) -> str:
    """Two-digit start year of the season containing `today` (European seasons start in July)."""
    now = today or datetime.now(timezone.utc)
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(code: str) -> str:
    return f"{code[:2]}/{code[2:]}"


def season_from_date(d: pd.Timestamp) -> str:
    start_year = d.year if d.month >= 7 else d.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _meta_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".meta.json")


def _is_fresh(csv_path: Path, refetch_min: int) -> bool:
    if not csv_path.exists():
        return False
    age_min = (datetime.now(timezone.utc).timestamp() - csv_path.stat().st_mtime) / 60.0
    return age_min < refetch_min


def download_season(cfg: Config, season: str, code: str, force: bool = False) -> DownloadResult:
    """Fetch one season file into data/raw, honouring the cache rules."""
    csv_path = cfg.raw_dir / f"{season}_{code}.csv"
    current = current_season_code()
    if csv_path.exists() and not force:
        if season != current or _is_fresh(csv_path, int(cfg.data.get("current_season_refetch_min", 60))):
            return DownloadResult(csv_path, "cached")

    cfg.ensure_dirs()
    url = f"{BASE_URL}/{season}/{code}.csv"
    headers = {"User-Agent": str(cfg.data.get("user_agent", "prorrhesis/0.1 (personal research)"))}
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        return DownloadResult(None, "error", f"{type(exc).__name__}: {exc}")

    if resp.status_code == 404:
        return DownloadResult(None, "missing", "404")
    if resp.status_code != 200:
        return DownloadResult(None, "error", f"HTTP {resp.status_code}")

    csv_path.write_bytes(resp.content)
    meta = {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_modified": resp.headers.get("Last-Modified"),
        "etag": resp.headers.get("ETag"),
        "status": resp.status_code,
    }
    _meta_path(csv_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return DownloadResult(csv_path, "downloaded")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    text = _read_text(path)
    try:
        df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
        mode = "strict"
    except Exception:
        df = pd.read_csv(io.StringIO(text), skipinitialspace=True, on_bad_lines="skip")
        mode = "skipped_bad_lines"
    df.columns = [str(c).strip() for c in df.columns]
    return df, mode


def _to_float_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _pick_odds(df: pd.DataFrame, chain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First complete triple in the chain wins (spec 4.1)."""
    n = len(df)
    families = ODDS_CHAINS[chain]
    arrays = []
    for fam in families:
        triple = [_to_float_array(df, c) for c in fam]
        mask = np.ones(n, dtype=bool)
        for arr in triple:
            mask &= np.isfinite(arr) & (arr > 1.0)
        arrays.append((fam[0], triple, mask))
    for src_name, triple, mask in arrays:
        if mask.any():
            h = np.where(mask, triple[0], np.nan)
            d = np.where(mask, triple[1], np.nan)
            a = np.where(mask, triple[2], np.nan)
            src = np.where(mask, src_name, None)
            return h, d, a, src, mask
    empty = np.full(n, np.nan)
    return empty, empty.copy(), empty.copy(), np.full(n, None, dtype=object), np.zeros(n, dtype=bool)


def parse_season_file(path: Path, league: str, season: str) -> tuple[pd.DataFrame, ParseReport]:
    """Parse one season CSV into the normalized internal frame."""
    report = ParseReport()
    df, mode = _read_csv(path)
    if mode != "strict":
        report.notes.append(f"{path.name}: {mode}")
    report.rows_in = len(df)

    if "Date" not in df.columns or "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        report.notes.append(f"{path.name}: missing required columns")
        return pd.DataFrame(), report

    home = df["HomeTeam"].astype("string").str.strip()
    away = df["AwayTeam"].astype("string").str.strip()
    no_teams = home.isna() | away.isna() | (home == "") | (away == "")
    report.dropped_no_teams = int(no_teams.sum())

    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    report.bad_dates = int(dates.isna().sum())

    hg = pd.to_numeric(df["FTHG"], errors="coerce") if "FTHG" in df.columns else pd.Series(np.nan, index=df.index)
    ag = pd.to_numeric(df["FTAG"], errors="coerce") if "FTAG" in df.columns else pd.Series(np.nan, index=df.index)
    unplayed = hg.isna() | ag.isna()
    report.dropped_unplayed = int(unplayed.sum())

    out = pd.DataFrame({
        "league": league,
        "season": season,
        "date": dates.dt.strftime("%Y-%m-%d"),
        "home_name": home,
        "away_name": away,
        "hg": hg.astype("Int64"),
        "ag": ag.astype("Int64"),
    })

    for src_col, dst in HT_COLS.items():
        out[dst] = pd.to_numeric(df[src_col], errors="coerce").astype("Int64") if src_col in df.columns else pd.Series(
            pd.NA, index=df.index, dtype="Int64"
        )
    for src_col, dst in STAT_COLS.items():
        out[dst] = pd.to_numeric(df[src_col], errors="coerce").astype("Int64") if src_col in df.columns else pd.Series(
            pd.NA, index=df.index, dtype="Int64"
        )

    res = np.where(out["hg"].fillna(-1) > out["ag"].fillna(-1), "H",
                   np.where(out["hg"].fillna(-1) == out["ag"].fillna(-1), "D", "A"))
    out["res"] = res
    if "FTR" in df.columns:
        ftr = df["FTR"].astype("string").str.strip()
        mismatch = (ftr.notna()) & (ftr != pd.Series(res, index=df.index))
        report.res_disagreements = int(mismatch.sum())
        if report.res_disagreements:
            report.notes.append(f"{path.name}: {report.res_disagreements} FTR disagreements, trusting goals")

    kickoff = _kickoff_utc(df, league)
    out["kickoff_utc"] = kickoff

    for chain in ("open", "close"):
        h, d, a, src, _ = _pick_odds(df, chain)
        out[f"odds_{chain}_h"] = h
        out[f"odds_{chain}_d"] = d
        out[f"odds_{chain}_a"] = a
        out[f"odds_{chain}_src"] = src

    keep = ~(no_teams | unplayed | dates.isna())
    out = out.loc[keep].reset_index(drop=True)
    report.rows_kept = len(out)
    return out, report


def _kickoff_utc(df: pd.DataFrame, league: str) -> pd.Series:
    if "Time" not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    times = pd.to_datetime(df["Time"], format="mixed", errors="coerce")
    naive = dates.dt.normalize() + (times - times.dt.normalize())
    tz = ZoneInfo(LEAGUE_TZ.get(league, "Europe/London"))
    localized = naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    return localized.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(localized.notna(), None)


MATCH_COLUMNS = [
    "league", "season", "date", "kickoff_utc", "home_id", "away_id", "hg", "ag", "hthg", "htag", "res",
    "hs", "as_", "hst", "ast", "hc", "ac", "hf", "af", "hy", "ay", "hr", "ar",
    "odds_open_h", "odds_open_d", "odds_open_a", "odds_open_src",
    "odds_close_h", "odds_close_d", "odds_close_a", "odds_close_src",
]


def upsert_matches(conn: sqlite3.Connection, frame: pd.DataFrame) -> dict:
    """Resolve names and upsert matches. Unresolved names reject only their rows."""
    stats = {"rows": len(frame), "inserted": 0, "updated": 0, "rejected": 0, "unresolved": []}
    if frame.empty:
        return stats

    resolved_cache: dict[str, int | None] = {}

    def resolve_name(name) -> int | None:
        if name in resolved_cache:
            return resolved_cache[name]
        tid = resolve(conn, str(name), SOURCE_FOOTBALLDATA)
        resolved_cache[name] = tid
        return tid

    keep_rows = []
    for _, row in frame.iterrows():
        h = resolve_name(row["home_name"])
        a = resolve_name(row["away_name"])
        if h is None or a is None:
            missing = row["home_name"] if h is None else row["away_name"]
            stats["rejected"] += 1
            if missing not in stats["unresolved"]:
                stats["unresolved"].append(str(missing))
            continue
        keep_rows.append((row, h, a))

    if not keep_rows:
        return stats

    cols = ", ".join(MATCH_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in MATCH_COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in MATCH_COLUMNS if c not in ("league", "date", "home_id", "away_id"))
    sql = f"INSERT INTO matches ({cols}) VALUES ({placeholders}) ON CONFLICT(league, date, home_id, away_id) DO UPDATE SET {updates}"

    before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    payload = []
    for row, h, a in keep_rows:
        payload.append({
            "league": row["league"], "season": row["season"], "date": row["date"],
            "kickoff_utc": row["kickoff_utc"], "home_id": h, "away_id": a,
            "hg": int(row["hg"]), "ag": int(row["ag"]),
            "hthg": _int_or_none(row["hthg"]), "htag": _int_or_none(row["htag"]),
            "res": row["res"],
            "hs": _int_or_none(row["hs"]), "as_": _int_or_none(row["as_"]),
            "hst": _int_or_none(row["hst"]), "ast": _int_or_none(row["ast"]),
            "hc": _int_or_none(row["hc"]), "ac": _int_or_none(row["ac"]),
            "hf": _int_or_none(row["hf"]), "af": _int_or_none(row["af"]),
            "hy": _int_or_none(row["hy"]), "ay": _int_or_none(row["ay"]),
            "hr": _int_or_none(row["hr"]), "ar": _int_or_none(row["ar"]),
            "odds_open_h": _float_or_none(row["odds_open_h"]), "odds_open_d": _float_or_none(row["odds_open_d"]),
            "odds_open_a": _float_or_none(row["odds_open_a"]), "odds_open_src": _str_or_none(row["odds_open_src"]),
            "odds_close_h": _float_or_none(row["odds_close_h"]), "odds_close_d": _float_or_none(row["odds_close_d"]),
            "odds_close_a": _float_or_none(row["odds_close_a"]), "odds_close_src": _str_or_none(row["odds_close_src"]),
        })
    conn.executemany(sql, payload)
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    stats["inserted"] = after - before
    stats["updated"] = len(payload) - stats["inserted"]
    return stats


def _int_or_none(v) -> int | None:
    try:
        if v is None or pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _str_or_none(v) -> str | None:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return str(v)
