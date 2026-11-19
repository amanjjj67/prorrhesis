from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from prorrhesis.models.dixoncoles import _grid_from_lambdas
from prorrhesis.sim import season as sim
from prorrhesis.sim import whatif

from conftest import LEAGUE

SEASON = "2425"
PAIRS = [(1, 2), (2, 1), (3, 4), (4, 3), (1, 3), (3, 1), (2, 4), (4, 2),
         (1, 4), (4, 1), (2, 3), (3, 2)]
RESULTS = [(2, 1), (0, 0), (3, 0), (1, 1), (2, 2), (0, 1), (4, 0), (1, 2),
           (0, 3), (2, 0), (1, 1), (3, 1)]


def _insert(conn, pairs, results=None, season=SEASON, start=date(2024, 8, 10)):
    results = results or [(1, 0)] * len(pairs)
    payload = []
    for i, ((home, away), (hg, ag)) in enumerate(zip(pairs, results)):
        d = start + timedelta(days=7 * i)
        payload.append((LEAGUE, season, d.isoformat(), home, away, hg, ag,
                        "H" if hg > ag else "D" if hg == ag else "A"))
    conn.executemany(
        "INSERT INTO matches(league, season, date, home_id, away_id, hg, ag, res) "
        "VALUES (?,?,?,?,?,?,?,?)", payload)
    conn.commit()


def _grid():
    return _grid_from_lambdas(1.4, 1.1, -0.03, 6)


def _triples(n):
    return np.tile(np.array([[0.45, 0.28, 0.27]]), (n, 1))


@pytest.fixture()
def db(synthetic_db):
    return synthetic_db


def test_remaining_fixtures_count_is_n_times_n_minus_one_minus_played(db):
    _insert(db, PAIRS[:4])
    teams = sim.season_teams(db, LEAGUE, SEASON)
    assert teams == [1, 2, 3, 4]
    assert len(sim.remaining_pairs(db, LEAGUE, SEASON)) == 4 * 3 - 4


def test_every_ordered_pair_appears_once_in_a_full_season(db):
    _insert(db, PAIRS)
    assert sim.remaining_pairs(db, LEAGUE, SEASON) == []


def test_the_simulator_is_disabled_until_every_club_has_played(db):
    _insert(db, PAIRS, season="2324", start=date(2023, 8, 12))
    _insert(db, [(1, 2)])
    out = sim.simulate(db, LEAGUE, SEASON, n=100,
                        triples=_triples(11), grids=[_grid()] * 11)
    assert out.disabled
    assert "played" in out.reason


def test_title_probabilities_sum_to_one(db):
    _insert(db, PAIRS[:4])
    pairs = sim.remaining_pairs(db, LEAGUE, SEASON)
    out = sim.simulate(db, LEAGUE, SEASON, n=500,
                       triples=_triples(len(pairs)), grids=[_grid()] * len(pairs))
    assert not out.disabled
    assert sum(r["p_title"] for r in out.rows) == pytest.approx(1.0)


def test_relegation_probabilities_sum_to_one(db):
    _insert(db, PAIRS[:4])
    pairs = sim.remaining_pairs(db, LEAGUE, SEASON)
    out = sim.simulate(db, LEAGUE, SEASON, n=500, relegation=1,
                       triples=_triples(len(pairs)), grids=[_grid()] * len(pairs))
    assert sum(r["p_relegation"] for r in out.rows) == pytest.approx(1.0)


def test_the_simulation_is_deterministic_with_a_fixed_seed(db):
    _insert(db, PAIRS[:4])
    pairs = sim.remaining_pairs(db, LEAGUE, SEASON)
    args = dict(triples=_triples(len(pairs)), grids=[_grid()] * len(pairs))
    a = sim.simulate(db, LEAGUE, SEASON, n=300, seed=7, **args)
    b = sim.simulate(db, LEAGUE, SEASON, n=300, seed=7, **args)
    assert a.rows == b.rows


def test_a_finished_season_reproduces_the_final_table(db):
    """M8: with every result played, the simulated table is the real table."""
    _insert(db, PAIRS, RESULTS)
    out = sim.simulate(db, LEAGUE, SEASON, n=50,
                       triples=_triples(0), grids=[])
    assert not out.disabled
    table = {r["team_id"]: r for r in out.rows}
    assert table[1]["expected_points"] == pytest.approx(8.0)
    assert table[2]["expected_points"] == pytest.approx(8.0)
    assert table[3]["expected_points"] == pytest.approx(9.0)
    assert table[4]["expected_points"] == pytest.approx(7.0)
    assert sum(r["p_title"] for r in out.rows) == pytest.approx(1.0)


def test_expected_points_include_the_points_already_earned(db):
    _insert(db, PAIRS[:4])
    pairs = sim.remaining_pairs(db, LEAGUE, SEASON)
    out = sim.simulate(db, LEAGUE, SEASON, n=500,
                       triples=_triples(len(pairs)), grids=[_grid()] * len(pairs))
    for row in out.rows:
        assert row["expected_points"] >= row["points_now"]


# -- what-if ---------------------------------------------------------------

def test_whatif_scales_the_grid_and_keeps_probabilities_valid():
    base = whatif.apply((1.4, 1.1, -0.03), max_goals=6)
    weaker = whatif.apply((1.4, 1.1, -0.03), max_goals=6, attack=-0.08, side="home")
    assert base["pH"] + base["pD"] + base["pA"] == pytest.approx(1.0)
    assert weaker["pH"] + weaker["pD"] + weaker["pA"] == pytest.approx(1.0)
    assert weaker["pH"] < base["pH"]
    assert weaker["pA"] > base["pA"]


def test_whatif_clamps_the_sliders():
    assert whatif.clamp(0.9) == pytest.approx(0.25)
    assert whatif.clamp(-0.9) == pytest.approx(-0.25)
    out = whatif.apply((1.4, 1.1, -0.03), max_goals=6, attack=5.0)
    assert out["adjustment"]["attack"] == pytest.approx(0.25)


def test_whatif_is_not_stored_in_the_ledger(db):
    from prorrhesis.db import open_db
    from prorrhesis.ledger import append_prediction

    before = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    whatif.apply((1.4, 1.1, -0.03), max_goals=6, attack=-0.08)
    after = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    assert before == after
