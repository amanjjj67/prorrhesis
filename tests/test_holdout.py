from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from prorrhesis.config import load_config
from prorrhesis.eval.holdout import (
    holdout_broken,
    read_lock,
    season_windows,
    write_lock,
)
from prorrhesis.eval.walkforward import create_run

from conftest import LEAGUE, TEAMS, build_history

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _season_rows(season: str, start: date, weeks: int = 6):
    rows = []
    for w in range(weeks):
        d = start + timedelta(days=7 * w)
        rows.append((LEAGUE, season, d.isoformat(), 1, 2, 1, 0, "H"))
    return rows


@pytest.fixture()
def multi_season_db(synthetic_db):
    conn = synthetic_db
    payload = []
    for season, start in (("1415", date(2014, 8, 16)), ("1516", date(2015, 8, 15)),
                          ("1617", date(2016, 8, 13)), ("1718", date(2017, 8, 12))):
        payload += _season_rows(season, start)
    conn.executemany(
        "INSERT INTO matches(league, season, date, home_id, away_id, hg, ag, res) "
        "VALUES (?,?,?,?,?,?,?,?)", payload)
    conn.commit()
    return conn


def test_the_headline_window_is_the_last_two_completed_seasons(multi_season_db):
    cfg = load_config(ROOT / "config.yaml")
    win = season_windows(multi_season_db, cfg)
    assert win.headline_start == "2016-08-13"
    assert win.headline_end == "2017-09-16"


def test_the_tuning_window_stops_before_the_headline_window(multi_season_db):
    cfg = load_config(ROOT / "config.yaml")
    win = season_windows(multi_season_db, cfg)
    assert win.tuning_start == win.eval_start
    assert win.tuning_end < win.headline_start


def test_the_lock_records_the_config_hash(tmp_path):
    cfg = replace(load_config(ROOT / "config.yaml"), root=tmp_path)
    path = write_lock(cfg, "2022-08-14")
    assert path.exists()
    lock = read_lock(cfg)
    assert lock["config_hash"] == cfg.config_hash
    assert lock["headline_window_start"] == "2022-08-14"


def test_holdout_is_broken_when_the_config_changes_after_the_headline_run(multi_season_db, tmp_path):
    cfg = replace(load_config(ROOT / "config.yaml"), root=tmp_path)
    conn = multi_season_db
    write_lock(cfg, "2022-08-14")
    create_run(conn, "headline_1", "headline", cfg)
    assert not holdout_broken(cfg, conn)

    changed = replace(cfg, config_hash="deadbeef" * 8)
    assert holdout_broken(changed, conn)


def test_no_headline_run_means_no_broken_holdout(multi_season_db, tmp_path):
    cfg = replace(load_config(ROOT / "config.yaml"), root=tmp_path)
    conn = multi_season_db
    write_lock(cfg, "2022-08-14")
    assert not holdout_broken(cfg, conn)


def test_with_little_history_the_windows_collapse(multi_season_db):
    cfg = replace(load_config(ROOT / "config.yaml"),
                  eval={"eval_start": "2017-08-01", "holdout_seasons": 9})
    win = season_windows(multi_season_db, cfg)
    assert win.tuning_start == "2017-08-01"
    assert win.headline_start == win.headline_end == win.tuning_end
