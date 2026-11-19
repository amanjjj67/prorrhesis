from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from prorrhesis.config import load_config
from prorrhesis.db import open_db
from prorrhesis.names import (
    SOURCE_FOOTBALLDATA,
    all_aliases,
    ensure_team,
    load_discovered,
    load_overrides,
    resolve,
    seed_foundation,
    slugify,
    unresolved_aliases,
    write_override,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="module")
def live_db(cfg):
    if not cfg.db_path.exists():
        pytest.skip("no ingested database yet - run: python -m prorrhesis ingest")
    conn = open_db(cfg.db_path, create=False)
    yield conn
    conn.close()


@pytest.fixture()
def tmp_db(tmp_path):
    conn = open_db(tmp_path / "t.db")
    yield conn
    conn.close()


def test_slugify():
    assert slugify("Man United") == "man-united"
    assert slugify("Bayern Munchen") == "bayern-munchen"
    assert slugify("  Paris SG  ") == "paris-sg"
    assert slugify("Nott'm Forest") == "nott-m-forest"


def test_resolve_is_exact_only(tmp_db):
    tid = ensure_team(tmp_db, "ac-fc")
    tmp_db.execute("INSERT INTO team_aliases(alias, source, team_id) VALUES (?,?,?)",
                   ("AC Milan", "footballdata", tid))
    tmp_db.commit()
    assert resolve(tmp_db, "AC Milan", "footballdata") == tid
    assert resolve(tmp_db, "ac milan", "footballdata") is None
    assert resolve(tmp_db, "AC Milan", "clubelo") is None


def test_two_clubs_never_share_a_team_id(tmp_db):
    a = ensure_team(tmp_db, "club-a")
    b = ensure_team(tmp_db, "club-b")
    assert a != b


def test_renamed_alias_maps_to_same_id(tmp_db):
    tid = ensure_team(tmp_db, "new-name-club")
    for alias in ("Old Name FC", "New Name FC"):
        tmp_db.execute("INSERT INTO team_aliases(alias, source, team_id) VALUES (?,?,?)",
                       (alias, SOURCE_FOOTBALLDATA, tid))
    tmp_db.commit()
    assert resolve(tmp_db, "Old Name FC", SOURCE_FOOTBALLDATA) == resolve(
        tmp_db, "New Name FC", SOURCE_FOOTBALLDATA)


def test_overrides_are_idempotent(tmp_path, tmp_db):
    ov = tmp_path / "alias_overrides.csv"
    write_override(ov, [("FC Test", SOURCE_FOOTBALLDATA, "fc-test")])
    first = load_overrides(tmp_db, ov)
    count_after_first = tmp_db.execute("SELECT COUNT(*) FROM team_aliases").fetchone()[0]
    second = load_overrides(tmp_db, ov)
    count_after_second = tmp_db.execute("SELECT COUNT(*) FROM team_aliases").fetchone()[0]
    assert first == 1 and second == 1
    assert count_after_first == count_after_second


def test_seeding_resolves_all_football_data_names(tmp_db):
    discovered = {
        SOURCE_FOOTBALLDATA: {
            "Man United": {"count": 10, "leagues": ["E0"], "seasons": ["1415"]},
            "Manchester City": {"count": 10, "leagues": ["E0"], "seasons": ["1415"]},
        }
    }
    report = seed_foundation(tmp_db, discovered)
    assert report.created_teams == 2
    assert unresolved_aliases(tmp_db, discovered) == []
    assert resolve(tmp_db, "Man United", SOURCE_FOOTBALLDATA) is not None
    assert resolve(tmp_db, "Manchester City", SOURCE_FOOTBALLDATA) is not None
    assert resolve(tmp_db, "Man United", SOURCE_FOOTBALLDATA) != resolve(
        tmp_db, "Manchester City", SOURCE_FOOTBALLDATA)


def test_every_team_id_maps_to_exactly_one_canon(tmp_db):
    discovered = {SOURCE_FOOTBALLDATA: {"Alpha FC": {"count": 1, "leagues": ["E0"], "seasons": ["1415"]}}}
    seed_foundation(tmp_db, discovered)
    rows = tmp_db.execute(
        "SELECT a.team_id, COUNT(DISTINCT t.canon) AS n FROM team_aliases a "
        "JOIN teams t ON t.id = a.team_id GROUP BY a.team_id"
    ).fetchall()
    assert all(r["n"] == 1 for r in rows)


def test_live_db_all_names_resolve(live_db, cfg):
    discovered = load_discovered(cfg.discovered_names_path)
    if not discovered:
        pytest.skip("no discovered names file")
    unresolved = unresolved_aliases(live_db, discovered)
    assert unresolved == [], f"unresolved names: {unresolved[:20]}"


def test_live_db_no_duplicate_canon(live_db):
    dupes = live_db.execute(
        "SELECT canon, COUNT(*) c FROM teams GROUP BY canon HAVING c > 1").fetchall()
    assert dupes == []
    bad = live_db.execute(
        "SELECT id, COUNT(DISTINCT canon) n FROM teams GROUP BY id HAVING n > 1").fetchall()
    assert bad == []
