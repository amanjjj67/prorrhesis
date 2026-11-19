"""Canonical team identity (spec section 5).

Resolution is *exact* on (alias, source) and never fuzzy at runtime. Fuzzy
matching exists only in the seeding/suggestion workflow, where every automatic
match is logged for human review.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

SOURCE_FOOTBALLDATA = "footballdata"
SOURCE_CLUBELO = "clubelo"
SOURCE_UNDERSTAT = "understat"
ALL_SOURCES = (SOURCE_FOOTBALLDATA, SOURCE_CLUBELO, SOURCE_UNDERSTAT)

FUZZY_AUTO_THRESHOLD = 0.86
REVIEW_CANDIDATES = 3


def slugify(name: str) -> str:
    """Canonical string for a club name: lowercase, ascii-folded, dash separated."""
    folded = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(ch for ch in folded if not unicodedata.combining(ch))
    ascii_name = ascii_name.lower().replace("&", " and ")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def ensure_team(conn: sqlite3.Connection, canon: str) -> int:
    row = conn.execute("SELECT id FROM teams WHERE canon = ?", (canon,)).fetchone()
    if row is not None:
        return int(row["id"])
    cur = conn.execute("INSERT INTO teams(canon) VALUES (?)", (canon,))
    return int(cur.lastrowid)


def add_alias(conn: sqlite3.Connection, alias: str, source: str, team_id: int) -> None:
    conn.execute(
        "INSERT INTO team_aliases(alias, source, team_id) VALUES (?,?,?) "
        "ON CONFLICT(alias, source) DO UPDATE SET team_id = excluded.team_id",
        (alias, source, team_id),
    )


def resolve(conn: sqlite3.Connection, name: str, source: str) -> int | None:
    """Exact (alias, source) lookup. Returns None when unknown."""
    row = conn.execute(
        "SELECT team_id FROM team_aliases WHERE alias = ? AND source = ?", (name, source)
    ).fetchone()
    return int(row["team_id"]) if row is not None else None


def all_canon(conn: sqlite3.Connection) -> list[str]:
    return [str(r["canon"]) for r in conn.execute("SELECT canon FROM teams ORDER BY canon")]


def all_aliases(conn: sqlite3.Connection, source: str | None = None) -> list[tuple[str, str, int]]:
    if source is None:
        rows = conn.execute(
            "SELECT a.alias, a.source, a.team_id FROM team_aliases a ORDER BY a.source, a.alias"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT a.alias, a.source, a.team_id FROM team_aliases a WHERE a.source = ? ORDER BY a.alias",
            (source,),
        ).fetchall()
    return [(str(r["alias"]), str(r["source"]), int(r["team_id"])) for r in rows]


# --------------------------------------------------------------------------
# discovered names (sidecar file, so the DB schema stays exactly as specified)
# --------------------------------------------------------------------------

def load_discovered(path: str | Path) -> dict[str, dict[str, dict]]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_discovered(path: str | Path, discovered: dict[str, dict[str, dict]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(discovered, fh, indent=2, sort_keys=True)


def record_discovered(
    discovered: dict[str, dict[str, dict]], source: str, name: str, league: str, season: str
) -> None:
    bucket = discovered.setdefault(source, {})
    entry = bucket.setdefault(name, {"count": 0, "leagues": [], "seasons": []})
    entry["count"] += 1
    for lst, value in ((entry["leagues"], league), (entry["seasons"], season)):
        if value and value not in lst:
            lst.append(value)


# --------------------------------------------------------------------------
# overrides file (human-confirmed): alias,source,canon
# --------------------------------------------------------------------------

def load_overrides(conn: sqlite3.Connection, path: str | Path) -> int:
    """Apply data/alias_overrides.csv. Returns the number of overrides applied."""
    p = Path(path)
    if not p.exists():
        return 0
    applied = 0
    with p.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            alias = (row.get("alias") or "").strip()
            source = (row.get("source") or "").strip()
            canon = (row.get("canon") or "").strip()
            if not alias or not source or not canon:
                continue
            team_id = ensure_team(conn, canon)
            add_alias(conn, alias, source, team_id)
            applied += 1
    conn.commit()
    return applied


def write_override(path: str | Path, rows: list[tuple[str, str, str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alias", "source", "canon"])
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# seeding and suggestion
# --------------------------------------------------------------------------

@dataclass
class SeedReport:
    created_teams: int
    new_aliases: int
    fuzzy_matches: list[tuple[str, str, str, str, float]]
    unresolved: list[tuple[str, str]]
    review_rows: list[tuple[str, str, str]]


def _best_candidates(name: str, canon_list: list[str], n: int = REVIEW_CANDIDATES) -> list[str]:
    return get_close_matches(name, canon_list, n=n, cutoff=0.4)


def seed_foundation(
    conn: sqlite3.Connection, discovered: dict[str, dict[str, dict]]
) -> SeedReport:
    """Create canonical teams from discovered football-data names and map other sources.

    Football-data names are the source of truth: the canonical is the slug of the
    name. Names from other sources are matched exactly first, then fuzzily (with
    every automatic match written to the review file).
    """
    canon_list = all_canon(conn)
    report = SeedReport(0, 0, [], [], [])

    fd_names = sorted(discovered.get(SOURCE_FOOTBALLDATA, {}).keys())
    for name in fd_names:
        canon = slugify(name)
        if not canon:
            continue
        existing = conn.execute("SELECT id FROM teams WHERE canon = ?", (canon,)).fetchone()
        if existing is None:
            # a renamed club keeps its old team id: fuzzy-match against known canons
            candidates = get_close_matches(canon, canon_list, n=1, cutoff=FUZZY_AUTO_THRESHOLD)
            if candidates and candidates[0] != canon:
                team_id = ensure_team(conn, candidates[0])
                report.fuzzy_matches.append((name, SOURCE_FOOTBALLDATA, canon, candidates[0], 1.0))
                report.review_rows.append((name, SOURCE_FOOTBALLDATA, candidates[0]))
            else:
                team_id = ensure_team(conn, canon)
                report.created_teams += 1
                canon_list = all_canon(conn)
        else:
            team_id = int(existing["id"])
        if resolve(conn, name, SOURCE_FOOTBALLDATA) != team_id:
            add_alias(conn, name, SOURCE_FOOTBALLDATA, team_id)
            report.new_aliases += 1

    conn.commit()

    for source in ALL_SOURCES:
        if source == SOURCE_FOOTBALLDATA:
            continue
        for name in sorted(discovered.get(source, {}).keys()):
            if resolve(conn, name, source) is not None:
                continue
            canon = slugify(name)
            exact = conn.execute("SELECT id FROM teams WHERE canon = ?", (canon,)).fetchone()
            if exact is not None:
                add_alias(conn, name, source, int(exact["id"]))
                report.new_aliases += 1
                report.review_rows.append((name, source, canon))
                continue
            candidates = get_close_matches(canon, canon_list, n=1, cutoff=FUZZY_AUTO_THRESHOLD)
            if candidates:
                target = candidates[0]
                add_alias(conn, name, source, ensure_team(conn, target))
                report.new_aliases += 1
                report.fuzzy_matches.append((name, source, canon, target, 1.0))
                report.review_rows.append((name, source, target))
            else:
                report.unresolved.append((name, source))
    conn.commit()
    return report


def unresolved_aliases(conn: sqlite3.Connection, discovered: dict[str, dict[str, dict]]) -> list[tuple[str, str]]:
    out = []
    for source, names in discovered.items():
        for name in names:
            if resolve(conn, name, source) is None:
                out.append((name, source))
    return sorted(out)


def write_suggestions(
    path: str | Path, conn: sqlite3.Connection, unresolved: list[tuple[str, str]]
) -> int:
    """Write fuzzy candidates for unresolved aliases to a text file for human review."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    canon_list = all_canon(conn)
    lines = [
        "# PRORRHESIS alias suggestions",
        "# For each unresolved alias, review the candidates and write the chosen",
        "# canonical into data/alias_overrides.csv as: alias,source,canon",
        "",
    ]
    for name, source in unresolved:
        canon = slugify(name)
        cands = _best_candidates(canon, canon_list) or _best_candidates(name, canon_list)
        lines.append(f"alias: {name}  source: {source}  slug: {canon}")
        if cands:
            for c in cands:
                lines.append(f"    - {c}")
        else:
            lines.append("    - (no candidate)")
        lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return len(unresolved)
