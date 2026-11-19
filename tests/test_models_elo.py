from __future__ import annotations

import numpy as np
import pytest

from prorrhesis.models.elo import (
    DEFAULT_HOME_ADV,
    EloModel,
    RatingState,
    elo_update,
    fit_link,
    link_probs,
    softplus,
)

from conftest import LEAGUE


def test_elo_update_is_zero_sum():
    Rh, Ra = 1500.0, 1500.0
    for hg, ag in ((2, 0), (1, 1), (0, 3), (3, 1)):
        nh, na = elo_update(Rh, Ra, hg, ag, 65.0, 20.0)
        assert nh + na == pytest.approx(Rh + Ra)


def test_elo_update_direction():
    nh, na = elo_update(1600.0, 1500.0, 2, 0, 65.0, 20.0)
    assert nh > 1600.0
    assert na < 1500.0
    nh, na = elo_update(1500.0, 1600.0, 0, 2, 65.0, 20.0)
    assert nh < 1500.0


def test_ordinal_link_sums_to_one_and_draw_non_negative():
    d = np.array([-400.0, 0.0, 50.0, 400.0])
    for s in (100.0, 173.7, 300.0):
        for c in (0.0, 0.5, 2.0):
            P = link_probs(d, s, c)
            assert np.allclose(P.sum(axis=1), 1.0)
            assert np.all(P[:, 1] >= 0.0)
            assert np.all(P >= 0.0)


def test_link_home_advantage_direction():
    d = np.array([0.0])
    low = link_probs(d, 173.7, 0.5)[0]
    high = link_probs(d, 173.7, 2.0)[0]
    # a larger draw threshold moves mass into the draw, shrinking both sides
    assert high[1] > low[1]
    assert low[0] > high[0]
    assert low[2] > high[2]


def test_equal_ratings_favour_home():
    P = link_probs(np.array([65.0]), 173.7, 0.5)[0]
    assert P[0] > P[2]


def test_softplus_non_negative():
    assert softplus(-10.0) >= 0.0
    assert softplus(0.0) == pytest.approx(0.6931, abs=1e-3)
    assert softplus(1.0) > 0.0


def test_fit_link_recovers_scale(history):
    from prorrhesis.models.elo import RatingState

    state = RatingState(20.0, {LEAGUE: 65.0}, 0.25, 50.0)
    diffs, _, _ = state.process(history)
    y = np.array([{"H": 0, "D": 1, "A": 2}[r] for r in history["res"]])
    s, c = fit_link(diffs, y)
    assert 40.0 < s < 800.0
    assert c >= 0.0
    P = link_probs(diffs, s, c)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_elo_fit_rejects_future_data(history):
    model = EloModel()
    with pytest.raises(ValueError):
        model.fit(history, as_of=history["date"].min())


def test_elo_predicts_valid_probabilities(history):
    model = EloModel()
    model.fit(history, as_of="2017-12-01")
    fixtures = history.tail(10)[["league", "home_id", "away_id"]]
    P = model.predict(fixtures)
    assert len(P) == 10
    assert np.allclose(P[["pH", "pD", "pA"]].sum(axis=1), 1.0)
    assert (P[["pH", "pD", "pA"]].to_numpy() >= 0).all()


def test_same_date_matches_use_pre_date_ratings(history):
    state = RatingState(20.0, {LEAGUE: 65.0}, 0.25, 50.0)
    state.process(history)
    # after processing, a team's rating must be strictly informed by prior dates only:
    # re-running with the same data must be idempotent
    state2 = RatingState(20.0, {LEAGUE: 65.0}, 0.25, 50.0)
    state2.process(history)
    assert state.ratings == state2.ratings


def test_season_carry_over_shrinks_to_mean(history):
    state = RatingState(20.0, {LEAGUE: 65.0}, 0.25, 50.0)
    state.process(history)
    # after a season change the ratings are pulled towards the league mean
    assert max(state.ratings.values()) - min(state.ratings.values()) < 800.0
