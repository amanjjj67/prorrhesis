from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prorrhesis.config import load_config
from prorrhesis.db import open_db
from prorrhesis.eval.walkforward import (
    RESULT_COLUMNS,
    load_matches,
    save_oof,
    walk_forward,
    week_blocks,
)
from prorrhesis.eval.walkforward import test_as_fixtures as make_fixtures

from conftest import LEAGUE, TEAMS, build_history


def insert_history(conn, history: pd.DataFrame) -> None:
    payload = [
        (LEAGUE, r["season"], r["date"], r["home_id"], r["away_id"], r["hg"], r["ag"], r["res"])
        for r in history.to_dict("records")
    ]
    conn.executemany(
        "INSERT INTO matches(league, season, date, home_id, away_id, hg, ag, res) VALUES (?,?,?,?,?,?,?,?)",
        payload,
    )
    conn.commit()


class SpyModel:
    name = "spy"
    needs = {"results"}

    def __init__(self):
        self.seen_max_dates = []

    def fit(self, history, as_of):
        self.seen_max_dates.append(str(history["date"].max()) if len(history) else None)

    def predict(self, fixtures):
        n = len(fixtures)
        P = np.full((n, 3), 1 / 3)
        return pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)


class BrokenModel:
    name = "broken"
    needs = {"results"}

    def fit(self, history, as_of):
        raise RuntimeError("boom")

    def predict(self, fixtures):
        raise AssertionError("predict must not be reached")


class WorkingModel:
    name = "working"
    needs = {"results"}

    def fit(self, history, as_of):
        return None

    def predict(self, fixtures):
        n = len(fixtures)
        P = np.full((n, 3), 1 / 3)
        return pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)


class HistoryModel:
    """Predictions depend on the history the model was fitted on."""

    name = "hist"
    needs = {"results"}

    def __init__(self):
        self.rate = 1 / 3

    def fit(self, history, as_of):
        if len(history):
            self.rate = float((history["res"].astype(str) == "H").mean())
        return None

    def predict(self, fixtures):
        n = len(fixtures)
        pH = float(min(max(self.rate, 0.05), 0.9))
        rest = (1.0 - pH) / 2.0
        return pd.DataFrame({"pH": np.full(n, pH), "pD": np.full(n, rest),
                             "pA": np.full(n, rest)}, index=fixtures.index)


def test_ensemble_rows_of_early_blocks_do_not_depend_on_later_results(db):
    """I4 at driver level: rewriting later results must not move the ensemble
    rows of the blocks before the rewrite."""
    cfg = load_config()
    cut = "2016-11-28"

    walk_forward(db, [HistoryModel()], cfg, "wf_i4_a", "2016-10-01", "2016-12-31",
                 ensemble_members=["hist"])
    early_a = db.execute(
        "SELECT match_id, pH, pD, pA FROM oof_predictions WHERE run_id='wf_i4_a' "
        "AND model='ensemble' AND block_start <= ? ORDER BY match_id", (cut,)).fetchall()
    late_a = db.execute(
        "SELECT match_id, pH FROM oof_predictions WHERE run_id='wf_i4_a' "
        "AND model='hist' AND block_start > ? ORDER BY match_id", (cut,)).fetchall()

    db.execute("UPDATE matches SET res='D' WHERE date > ?", (cut,))
    db.commit()

    walk_forward(db, [HistoryModel()], cfg, "wf_i4_b", "2016-10-01", "2016-12-31",
                 ensemble_members=["hist"])
    early_b = db.execute(
        "SELECT match_id, pH, pD, pA FROM oof_predictions WHERE run_id='wf_i4_b' "
        "AND model='ensemble' AND block_start <= ? ORDER BY match_id", (cut,)).fetchall()
    late_b = db.execute(
        "SELECT match_id, pH FROM oof_predictions WHERE run_id='wf_i4_b' "
        "AND model='hist' AND block_start > ? ORDER BY match_id", (cut,)).fetchall()

    assert len(early_a) > 0
    assert [(r["match_id"], r["pH"], r["pD"], r["pA"]) for r in early_a] == \
           [(r["match_id"], r["pH"], r["pD"], r["pA"]) for r in early_b]
    # the rewrite really did change the later blocks, so the check is not vacuous
    assert [(r["match_id"], r["pH"]) for r in late_a] != \
           [(r["match_id"], r["pH"]) for r in late_b]


def test_running_the_same_window_twice_gives_identical_numbers(db):
    """M6: the evaluation is deterministic - two runs must agree exactly."""
    cfg = load_config()
    models = [WorkingModel(), HistoryModel()]
    members = ["working", "hist"]
    walk_forward(db, models, cfg, "det_a", "2016-10-01", "2016-12-31",
                 ensemble_members=members)
    walk_forward(db, models, cfg, "det_b", "2016-10-01", "2016-12-31",
                 ensemble_members=members)

    sql = ("SELECT model, match_id, pH, pD, pA, status FROM oof_predictions "
           "WHERE run_id = ? ORDER BY model, match_id")
    a = db.execute(sql, ("det_a",)).fetchall()
    b = db.execute(sql, ("det_b",)).fetchall()
    assert len(a) > 0
    assert [(r["model"], r["match_id"], r["pH"], r["pD"], r["pA"], r["status"]) for r in a] == \
           [(r["model"], r["match_id"], r["pH"], r["pD"], r["pA"], r["status"]) for r in b]


def test_market_close_is_never_an_ensemble_member(db):
    """I5: closing odds are a benchmark only."""
    cfg = load_config()
    res = walk_forward(db, [WorkingModel()], cfg, "wf_i5", "2016-10-01", "2016-11-30",
                       ensemble_members=["working", "market_close"])
    assert res.ensemble_members == ["working"]


class AbstainingModel:
    """A model that abstains on every second match."""

    name = "abstain"
    needs = {"results"}

    def fit(self, history, as_of):
        return None

    def predict(self, fixtures):
        n = len(fixtures)
        P = np.full((n, 3), 1 / 3)
        status = ["ok" if i % 2 == 0 else "abstain" for i in range(n)]
        return pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2],
                             "status": status}, index=fixtures.index)


def test_abstaining_ensemble_member_does_not_fail_the_block(db):
    """A member that abstains on some rows must still leave the block usable.

    The abstained rows are masked with NaN, which used to raise
    "assignment destination is read-only" under pandas 3.
    """
    cfg = load_config()
    res = walk_forward(db, [AbstainingModel()], cfg, "wf_abstain", "2016-10-01",
                       "2016-12-31", ensemble_members=["abstain"])
    assert res.failed == {}
    statuses = [str(r["status"]) for r in db.execute(
        "SELECT DISTINCT status FROM oof_predictions WHERE run_id='wf_abstain' "
        "AND model='abstain'").fetchall()]
    assert "ok" in statuses
    ens = db.execute(
        "SELECT COUNT(*) c FROM oof_predictions WHERE run_id='wf_abstain' "
        "AND model='ensemble'").fetchone()["c"]
    assert ens > 0


@pytest.fixture()
def db(synthetic_db):
    insert_history(synthetic_db, build_history(n_weeks=40))
    return synthetic_db


def test_test_as_fixtures_has_no_result_columns(db):
    matches = load_matches(db)
    fixtures = make_fixtures(matches)
    for col in RESULT_COLUMNS:
        assert col not in fixtures.columns
    assert "odds_close_h" not in fixtures.columns
    assert "odds_close_d" not in fixtures.columns
    assert "odds_close_a" not in fixtures.columns
    assert "home_id" in fixtures.columns and "away_id" in fixtures.columns
    assert "id" in fixtures.columns


def test_week_blocks_are_monday_start():
    blocks = week_blocks("2017-08-01", "2017-08-31", 7)
    assert blocks[0].start == "2017-07-31"
    assert len(blocks) == 5


def test_model_only_sees_data_before_block_start(db):
    cfg = load_config()
    spy = SpyModel()
    res = walk_forward(db, [spy], cfg, "wf_test", "2016-10-01", "2016-12-31")
    assert res.blocks > 0
    for seen, block in zip(spy.seen_max_dates, [b.start for b in week_blocks("2016-10-01", "2016-12-31", 7)]):
        if seen is not None:
            assert seen < block


def test_failed_model_is_recorded_and_run_continues(db):
    cfg = load_config()
    res = walk_forward(db, [BrokenModel(), WorkingModel()], cfg, "wf_fail", "2016-10-01", "2016-12-31")
    assert res.blocks > 0
    assert res.failed.get("broken", 0) > 0
    rows = db.execute(
        "SELECT DISTINCT status FROM oof_predictions WHERE run_id='wf_fail' AND model='broken'").fetchall()
    assert all(str(r["status"]).startswith("failed:") for r in rows)
    ok = db.execute(
        "SELECT COUNT(*) c FROM oof_predictions WHERE run_id='wf_fail' AND model='working' "
        "AND status='ok'").fetchone()["c"]
    assert ok > 0


def test_oof_is_idempotent(db):
    cfg = load_config()
    walk_forward(db, [WorkingModel()], cfg, "wf_idem", "2016-10-01", "2016-11-30")
    first = db.execute(
        "SELECT COUNT(*) c FROM oof_predictions WHERE run_id='wf_idem'").fetchone()["c"]
    walk_forward(db, [WorkingModel()], cfg, "wf_idem", "2016-10-01", "2016-11-30")
    second = db.execute(
        "SELECT COUNT(*) c FROM oof_predictions WHERE run_id='wf_idem'").fetchone()["c"]
    assert first == second
