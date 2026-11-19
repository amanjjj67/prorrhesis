from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prorrhesis.models.xgpoisson import XgPoissonModel, _objective_grad

from conftest import LEAGUE, build_history


def _history(n_weeks: int = 80, seed: int = 3) -> pd.DataFrame:
    return build_history(n_weeks=n_weeks, seed=seed)


# ---------------------------------------------------------------- gradient

def test_objective_gradient_matches_finite_difference():
    rng = np.random.default_rng(5)
    n_teams = 5
    n = 200
    hi = rng.integers(0, n_teams, n)
    ai = (hi + 1 + rng.integers(0, n_teams - 1, n)) % n_teams
    yh = rng.poisson(1.4, n).astype(float)
    ya = rng.poisson(1.1, n).astype(float)
    w = rng.uniform(0.3, 1.0, n)
    params = np.concatenate([[0.15, 0.25], rng.normal(0, 0.2, 2 * n_teams)])

    _, grad = _objective_grad(params.copy(), yh, ya, hi, ai, w, n_teams, 0.5)

    eps = 1e-6
    numeric = np.zeros_like(params)
    for i in range(len(params)):
        up = params.copy()
        up[i] += eps
        dn = params.copy()
        dn[i] -= eps
        fp, _ = _objective_grad(up, yh, ya, hi, ai, w, n_teams, 0.5)
        fm, _ = _objective_grad(dn, yh, ya, hi, ai, w, n_teams, 0.5)
        numeric[i] = (fp - fm) / (2 * eps)

    assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------- the target

def test_weight_one_reproduces_the_goals():
    history = _history()
    model = XgPoissonModel(w=1.0)
    proxies = {}
    yh, ya = model._blended_targets(history, proxies)
    assert np.allclose(yh, history["hg"].to_numpy(dtype=float))
    assert np.allclose(ya, history["ag"].to_numpy(dtype=float))


def test_the_target_is_the_blend_of_goals_and_xg():
    history = _history()
    model = XgPoissonModel(w=0.5)
    from prorrhesis.features.xgproxy import build_season_models

    proxies = build_season_models(history)
    yh, ya = model._blended_targets(history, proxies)
    goals = history["hg"].to_numpy(dtype=float)
    ok = np.isfinite(yh)
    assert ok.any(), "the proxy produced no xG at all"
    for i in np.flatnonzero(ok):
        xg = 2.0 * yh[i] - goals[i]
        assert 0.0 <= yh[i] <= max(goals[i], xg)
        assert yh[i] == pytest.approx(0.5 * (goals[i] + xg))


def test_missing_xg_leaves_the_target_missing():
    history = _history()
    model = XgPoissonModel(w=0.5)
    yh, ya = model._blended_targets(history, {})
    assert np.isnan(yh).all()
    assert np.isnan(ya).all()


def test_understat_xg_is_used_when_a_lookup_is_given():
    history = _history()
    model = XgPoissonModel(w=0.0, xg_lookup=lambda row: (2.5, 0.5))
    yh, ya = model._blended_targets(history, {})
    assert np.allclose(yh, 2.5)
    assert np.allclose(ya, 0.5)


# ----------------------------------------------------------------- fitting

def test_fit_rejects_future_rows():
    history = _history(n_weeks=30)
    model = XgPoissonModel()
    with pytest.raises(ValueError):
        model.fit(history, as_of="2016-08-01")


def test_fit_produces_parameters_for_the_league():
    history = _history(n_weeks=80)
    model = XgPoissonModel(w=0.5)
    model.fit(history, as_of="2018-03-01")
    assert LEAGUE in model.params
    params = model.params[LEAGUE]
    assert len(params.teams) == 4
    assert params.rho == 0.0


def test_uncertain_teams_have_no_rating():
    history = _history(n_weeks=12)
    model = XgPoissonModel(w=0.5, min_matches=5)
    model.fit(history, as_of="2017-03-01")
    for league, uncertain in model.uncertain_by_league.items():
        params = model.params[league]
        for t in uncertain:
            i = params.index_of(t)
            assert params.att[i] == 0.0
            assert params.deff[i] == 0.0


# -------------------------------------------------------------- prediction

def test_predict_returns_valid_probabilities():
    history = _history(n_weeks=80)
    model = XgPoissonModel(w=0.5)
    model.fit(history, as_of="2018-03-01")

    fixtures = pd.DataFrame({
        "league": [LEAGUE] * 3,
        "season": ["1819"] * 3,
        "date": ["2018-08-20"] * 3,
        "home_id": [1, 2, 3],
        "away_id": [2, 3, 4],
    })
    out = model.predict(fixtures)
    assert (out["pH"] + out["pD"] + out["pA"] - 1.0).abs().max() < 1e-9
    assert (out[["pH", "pD", "pA"]] > 0).all().all()
    assert (out["status"] == "ok").all()


def test_the_grid_has_no_tau_correction():
    """The xG-Poisson likelihood is quasi-Poisson: rho is zero and tau is one."""
    from prorrhesis.models.dixoncoles import _tau_terms

    x = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    logtau, _, _, _ = _tau_terms(x, y, np.full(4, 1.4), np.full(4, 1.1), 0.0)
    assert np.allclose(logtau, 0.0)


def test_unknown_league_abstains_with_flat_probabilities():
    model = XgPoissonModel()
    fixtures = pd.DataFrame({
        "league": ["XX"],
        "season": ["1819"],
        "date": ["2018-08-20"],
        "home_id": [1],
        "away_id": [2],
    })
    out = model.predict(fixtures)
    assert out["status"].iloc[0] == "abstain"
    assert out["pH"].iloc[0] == pytest.approx(1 / 3)


def test_stronger_attack_gets_a_higher_home_probability():
    history = _history(n_weeks=80)
    model = XgPoissonModel(w=0.5)
    model.fit(history, as_of="2018-03-01")
    params = model.params[LEAGUE]
    best = max(range(len(params.teams)),
               key=lambda i: params.att[i] - params.deff[i])
    worst = min(range(len(params.teams)),
                key=lambda i: params.att[i] - params.deff[i])
    strong = params.teams[best]
    weak = params.teams[worst]

    fixtures = pd.DataFrame({
        "league": [LEAGUE, LEAGUE],
        "season": ["1819", "1819"],
        "date": ["2018-08-20", "2018-08-20"],
        "home_id": [strong, weak],
        "away_id": [weak, strong],
    })
    out = model.predict(fixtures)
    assert out["pH"].iloc[0] > out["pH"].iloc[1]
