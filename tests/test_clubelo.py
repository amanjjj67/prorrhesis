from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pandas as pd
import pytest

from prorrhesis.config import load_config
from prorrhesis.db import open_db
from prorrhesis.ingest import clubelo as ce
from prorrhesis.names import SOURCE_CLUBELO, add_alias, ensure_team

ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT = (
    "Rank,Club,Country,Level,Elo,From,To\n"
    "1,Arsenal,ENG,1,1900.5,2016-08-01,2017-08-01\n"
    "2,Chelsea,ENG,1,1850.0,2016-08-01,2017-08-01\n"
)


@pytest.fixture()
def cfg(tmp_path):
    base = load_config(ROOT / "config.yaml")
    return replace(base, root=tmp_path,
                   data={"user_agent": "prorrhesis-test", "request_delay_s": 0.0})


@pytest.fixture()
def tmp_db(tmp_path):
    conn = open_db(tmp_path / "t.db")
    yield conn
    conn.close()


@pytest.fixture()
def teams(tmp_db):
    arsenal = ensure_team(tmp_db, "arsenal")
    chelsea = ensure_team(tmp_db, "chelsea")
    add_alias(tmp_db, "Arsenal", SOURCE_CLUBELO, arsenal)
    add_alias(tmp_db, "Chelsea", SOURCE_CLUBELO, chelsea)
    tmp_db.commit()
    return {"Arsenal": arsenal, "Chelsea": chelsea}


def _snapshot_text(clubs: dict[str, str | None]) -> str:
    lines = ["Rank,Club,Country,Level,Elo,From,To"]
    for i, (club, elo) in enumerate(clubs.items(), start=1):
        lines.append(f"{i},{club},ENG,1,{elo if elo is not None else 'NA'},2016-08-01,2017-08-01")
    return "\n".join(lines) + "\n"


# -- URL and cache ---------------------------------------------------------

def test_snapshot_url_has_the_date():
    assert ce.snapshot_url("2017-08-01") == "http://api.clubelo.com/2017-08-01"


def test_cache_path_is_under_data_cache(cfg):
    path = ce.cache_dir(cfg) / "2017-08-01.csv"
    assert path.parent == cfg.data_dir / "cache" / "clubelo"


def test_download_snapshot_uses_the_cache_without_touching_the_network(cfg, monkeypatch):
    path = ce.cache_dir(cfg) / "2017-08-01.csv"
    path.write_text(SNAPSHOT, encoding="utf-8")

    def boom(*args, **kwargs):
        raise AssertionError("the network must not be used for a cached snapshot")

    monkeypatch.setattr(ce.httpx, "get", boom)
    assert ce.download_snapshot("2017-08-01", cfg) == SNAPSHOT


def test_download_snapshot_stores_what_it_downloads(cfg, monkeypatch):
    class Response:
        status_code = 200
        text = SNAPSHOT

    monkeypatch.setattr(ce.httpx, "get", lambda *a, **k: Response())
    assert ce.download_snapshot("2017-08-01", cfg) == SNAPSHOT
    assert (ce.cache_dir(cfg) / "2017-08-01.csv").read_text(encoding="utf-8") == SNAPSHOT


def test_download_snapshot_raises_when_the_service_errors(cfg, monkeypatch):
    class Response:
        status_code = 502
        text = "Bad Gateway"

    monkeypatch.setattr(ce.httpx, "get", lambda *a, **k: Response())
    with pytest.raises(ce.ClubEloUnavailable):
        ce.download_snapshot("2017-08-01", cfg)


def test_download_snapshot_raises_when_the_service_is_unreachable(cfg, monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ce.httpx, "get", boom)
    with pytest.raises(ce.ClubEloUnavailable):
        ce.download_snapshot("2017-08-01", cfg)


# -- parsing ---------------------------------------------------------------

def test_parse_snapshot_reads_club_and_elo():
    df = ce.parse_snapshot(SNAPSHOT)
    assert list(df["Club"]) == ["Arsenal", "Chelsea"]
    assert df["Elo"].tolist() == [1900.5, 1850.0]


def test_parse_snapshot_strips_whitespace():
    df = ce.parse_snapshot("Club,Elo\n  Arsenal  ,1900\n")
    assert df["Club"].tolist() == ["Arsenal"]


def test_parse_snapshot_turns_missing_elo_into_nan():
    df = ce.parse_snapshot(_snapshot_text({"Arsenal": None}))
    assert pd.isna(df["Elo"].iloc[0])


def test_parse_snapshot_rejects_an_unexpected_layout():
    with pytest.raises(ValueError):
        ce.parse_snapshot("foo,bar\n1,2\n")


# -- snapshot dates --------------------------------------------------------

def test_block_snapshot_dates_is_empty_without_matches():
    assert ce.block_snapshot_dates([]) == []


def test_block_snapshot_dates_uses_one_snapshot_per_weekly_block():
    dates = ce.block_snapshot_dates(["2017-08-01", "2017-08-02", "2017-08-08", "2017-08-09"])
    assert dates == ["2017-07-30", "2017-08-06"]


def test_every_snapshot_is_strictly_before_the_matches_it_covers():
    match_dates = ["2017-08-01", "2017-08-02", "2017-08-08", "2017-08-09"]
    snapshots = ce.block_snapshot_dates(match_dates)
    for d in match_dates:
        before = [s for s in snapshots if s < d]
        assert before, f"no snapshot before {d}"
        latest = max(before)
        age = (pd.Timestamp(d) - pd.Timestamp(latest)).days
        assert 1 <= age <= 7, f"snapshot {latest} is {age} days before {d}"


def test_block_snapshot_dates_deduplicates_repeated_dates():
    dates = ce.block_snapshot_dates(["2017-08-01", "2017-08-01", "2017-08-02"])
    assert len(dates) == len(set(dates))


# -- upsert ---------------------------------------------------------------

def test_upsert_elo_resolves_names_through_the_clubelo_alias_source(tmp_db, teams):
    df = ce.parse_snapshot(SNAPSHOT)
    rows, unresolved = ce.upsert_elo(tmp_db, df, "2017-08-01")
    assert rows == 2
    assert unresolved == []
    stored = tmp_db.execute(
        "SELECT team_id, elo FROM elo_ext WHERE asof = '2017-08-01' ORDER BY team_id"
    ).fetchall()
    assert [(int(r["team_id"]), float(r["elo"])) for r in stored] == [
        (teams["Arsenal"], 1900.5), (teams["Chelsea"], 1850.0)
    ]


def test_upsert_elo_reports_unknown_names_instead_of_dropping_them(tmp_db, teams):
    df = ce.parse_snapshot(_snapshot_text({"Arsenal": "1900.5", "Unknown FC": "1700.0"}))
    rows, unresolved = ce.upsert_elo(tmp_db, df, "2017-08-01")
    assert rows == 1
    assert unresolved == ["Unknown FC"]


def test_upsert_elo_is_idempotent(tmp_db, teams):
    df = ce.parse_snapshot(SNAPSHOT)
    ce.upsert_elo(tmp_db, df, "2017-08-01")
    ce.upsert_elo(tmp_db, df, "2017-08-01")
    count = tmp_db.execute("SELECT COUNT(*) FROM elo_ext").fetchone()[0]
    assert count == 2


def test_upsert_elo_skips_rows_without_a_rating(tmp_db, teams):
    df = ce.parse_snapshot(_snapshot_text({"Arsenal": None, "Chelsea": "1850"}))
    rows, unresolved = ce.upsert_elo(tmp_db, df, "2017-08-01")
    assert rows == 1
    assert unresolved == []


def test_upsert_elo_keeps_snapshots_of_different_dates_apart(tmp_db, teams):
    df = ce.parse_snapshot(SNAPSHOT)
    ce.upsert_elo(tmp_db, df, "2017-08-01")
    df2 = ce.parse_snapshot(_snapshot_text({"Arsenal": "1950", "Chelsea": "1900"}))
    ce.upsert_elo(tmp_db, df2, "2017-08-08")
    rows = tmp_db.execute(
        "SELECT asof, elo FROM elo_ext WHERE team_id = ? ORDER BY asof",
        (teams["Arsenal"],),
    ).fetchall()
    assert [(str(r["asof"]), float(r["elo"])) for r in rows] == [
        ("2017-08-01", 1900.5), ("2017-08-08", 1950.0)
    ]


# -- fetching --------------------------------------------------------------

def test_fetch_snapshots_reads_cached_files(tmp_db, cfg, teams, monkeypatch):
    (ce.cache_dir(cfg) / "2017-08-01.csv").write_text(SNAPSHOT, encoding="utf-8")
    monkeypatch.setattr(ce.httpx, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    report = ce.fetch_snapshots(tmp_db, cfg, ["2017-08-01"])
    assert report.cached == 1
    assert report.failed == 0
    assert report.rows == 2
    assert report.reachable


def test_fetch_snapshots_degrades_when_the_service_is_down(tmp_db, cfg, teams, monkeypatch):
    class Response:
        status_code = 502
        text = "Bad Gateway"

    monkeypatch.setattr(ce.httpx, "get", lambda *a, **k: Response())
    report = ce.fetch_snapshots(tmp_db, cfg, ["2017-08-01"])
    assert report.failed == 1
    assert report.fetched == 0
    assert report.rows == 0
    assert not report.reachable
    assert report.errors


def test_fetch_snapshots_records_unknown_clubs(tmp_db, cfg, teams, monkeypatch):
    class Response:
        status_code = 200
        text = _snapshot_text({"Arsenal": "1900.5", "Unknown FC": "1700"})

    monkeypatch.setattr(ce.httpx, "get", lambda *a, **k: Response())
    report = ce.fetch_snapshots(tmp_db, cfg, ["2017-08-01"])
    assert report.rows == 1
    assert report.unresolved == ["Unknown FC"]


def test_fetch_snapshots_stops_after_repeated_failures(tmp_db, cfg, teams, monkeypatch):
    """An unreachable service must not be retried for every remaining date."""
    class Response:
        status_code = 502
        text = "Bad Gateway"

    calls: list[str] = []
    monkeypatch.setattr(ce.httpx, "get", lambda *a, **k: (calls.append(a[0]), Response())[1])
    dates = [f"2017-08-{d:02d}" for d in range(1, 21)]
    report = ce.fetch_snapshots(tmp_db, cfg, dates)
    assert len(calls) == 3
    assert report.failed == len(dates)
    assert not report.reachable
    assert any("stopped" in e for e in report.errors)


def test_fetch_for_matches_covers_the_whole_span(tmp_db, cfg, teams, monkeypatch):
    seen: list[str] = []

    def fake_download(date, cfg_, force=False, timeout=25.0):
        seen.append(date)
        return SNAPSHOT

    monkeypatch.setattr(ce, "download_snapshot", fake_download)
    matches = pd.DataFrame({"date": ["2017-08-01", "2017-08-08", "2017-08-15"]})
    report = ce.fetch_for_matches(tmp_db, cfg, matches)
    assert seen == ["2017-07-30", "2017-08-06", "2017-08-13"]
    assert report.rows == 6  # two clubs per snapshot


# -- feature integration ---------------------------------------------------

def test_load_clubelo_groups_rows_by_date(tmp_db, teams):
    from prorrhesis.features.store import load_clubelo

    df = ce.parse_snapshot(SNAPSHOT)
    ce.upsert_elo(tmp_db, df, "2017-08-01")
    snapshots = load_clubelo(tmp_db)
    assert set(snapshots) == {"2017-08-01"}
    assert snapshots["2017-08-01"][teams["Arsenal"]] == 1900.5
