from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prorrhesis.models.dixoncoles import (
    DixonColesModel,
    _fit_league,
    _grid_from_lambdas,
    _objective_grad,
    _tau_terms,
)

LEAGUE = "E0"


def _simulate(
    n_matches: int = 4000,
    n_teams: int = 10,
    mu0: float = 0.10,
    home: float = 0.25,
    rho: float = -0.10,
    seed: int = 11,
):
    """Simulate matches from known att/def/home/rho and return (frame, truth)."""
    rng = np.random.default_rng(seed)
    att = rng.normal(0.0, 0.35, n_teams)
    deff = rng.normal(0.0, 0.35, n_teams)
    att = att - att.mean()
    deff = deff - deff.mean()

    rows = []
    for i in range(n_matches):
        h = int(rng.integers(0, n_teams))
        a = int(rng.integers(0, n_teams - 1))
        if a >= h:
            a += 1
        lh = np.exp(mu0 + home + att[h] - deff[a])
        la = np.exp(mu0 + att[a] - deff[h])
        # rejection sampling against the Dixon-Coles tau correction
        while True:
            x = int(rng.poisson(lh))
            y = int(rng.poisson(la))
            if x == 0 and y == 0:
                tau = 1.0 - lh * la * rho
            elif x == 0 and y == 1:
                tau = 1.0 + lh * rho
            elif x == 1 and y == 0:
                tau = 1.0 + la * rho
            elif x == 1 and y == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            if rng.uniform() < tau:
                break
        rows.append({
            "league": LEAGUE,
            "season": "1819",
            "date": "2018-08-10",
            "home_id": h + 1,
            "away_id": a + 1,
            "hg": x,
            "ag": y,
            "res": "H" if x > y else "D" if x == y else "A",
        })
    frame = pd.DataFrame(rows)
    truth = {"att": att, "deff": deff, "mu0": mu0, "home": home, "rho": rho,
             "teams": list(range(1, n_teams + 1))}
    return frame, truth


def _grid_sum_is_one(grid: np.ndarray) -> bool:
    return abs(float(grid.sum()) - 1.0) < 1e-9


# ---------------------------------------------------------------- grid shape

def test_score_grid_sums_to_one():
    grid = _grid_from_lambdas(1.4, 1.1, -0.08, max_goals=10)
    assert grid.shape == (11, 11)
    assert _grid_sum_is_one(grid)


def test_score_grid_sums_to_one_across_parameter_space():
    for lh in (0.4, 1.0, 1.6, 2.8):
        for la in (0.4, 1.0, 1.6, 2.8):
            for rho in (-0.15, -0.05, 0.05, 0.15):
                grid = _grid_from_lambdas(lh, la, rho, max_goals=10)
                assert _grid_sum_is_one(grid), (lh, la, rho)


def test_grid_probabilities_are_non_negative():
    grid = _grid_from_lambdas(1.4, 1.1, -0.08, max_goals=10)
    assert (grid >= 0).all()


def test_grid_home_draw_away_sums_to_one():
    """The three outcomes derived from the grid must also sum to 1."""
    grid = _grid_from_lambdas(1.5, 1.0, -0.08, max_goals=10)
    d = grid.shape[0] - 1
    ii, jj = np.meshgrid(np.arange(d + 1), np.arange(d + 1), indexing="ij")
    pH = float(grid[ii > jj].sum())
    pD = float(np.trace(grid))
    pA = float(grid[ii < jj].sum())
    assert abs((pH + pD + pA) - 1.0) < 1e-9


# ------------------------------------------------------------------- tau

def test_tau_factors_are_positive():
    """tau appears inside a log, so it must stay strictly positive."""
    x = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    lh = np.full(4, 1.5)
    la = np.full(4, 1.2)
    for rho in (-0.2, -0.1, 0.0, 0.1, 0.2):
        logtau, _, _, _ = _tau_terms(x, y, lh, la, rho)
        assert (logtau > -1e8).all(), rho
        assert np.isfinite(logtau).all(), rho


def test_tau_values_match_the_dixon_coles_formula():
    lh, la, rho = 1.5, 1.2, -0.1
    x = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    lhv = np.full(5, lh)
    lav = np.full(5, la)
    logtau, _, _, _ = _tau_terms(x, y, lhv, lav, rho)
    expected = np.array([
        np.log(1 - lh * la * rho),
        np.log(1 + lh * rho),
        np.log(1 + la * rho),
        np.log(1 - rho),
        0.0,
    ])
    assert np.allclose(logtau, expected, atol=1e-12)


def test_tau_derivative_matches_finite_difference():
    x = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    lh = np.full(4, 1.5)
    la = np.full(4, 1.2)
    rho = -0.08
    _, d_rho, _, _ = _tau_terms(x, y, lh, la, rho)
    eps = 1e-7
    up, _, _, _ = _tau_terms(x, y, lh, la, rho + eps)
    dn, _, _, _ = _tau_terms(x, y, lh, la, rho - eps)
    assert np.allclose(d_rho, (up - dn) / (2 * eps), atol=1e-6)


# ------------------------------------------------------- gradient accuracy

def test_objective_gradient_matches_finite_difference():
    rng = np.random.default_rng(5)
    n_teams = 5
    n = 200
    hi = rng.integers(0, n_teams, n)
    ai = (hi + 1 + rng.integers(0, n_teams - 1, n)) % n_teams
    x = rng.poisson(1.4, n).astype(float)
    y = rng.poisson(1.1, n).astype(float)
    w = rng.uniform(0.3, 1.0, n)
    params = np.concatenate([[0.15, 0.25, -0.03], rng.normal(0, 0.2, 2 * n_teams)])

    _, grad = _objective_grad(params.copy(), x, y, hi, ai, w, n_teams, 0.5)

    eps = 1e-6
    numeric = np.zeros_like(params)
    for i in range(len(params)):
        up = params.copy()
        up[i] += eps
        dn = params.copy()
        dn[i] -= eps
        fp, _ = _objective_grad(up, x, y, hi, ai, w, n_teams, 0.5)
        fm, _ = _objective_grad(dn, x, y, hi, ai, w, n_teams, 0.5)
        numeric[i] = (fp - fm) / (2 * eps)

    assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-5)


# ------------------------------------------------- parameter recovery

def test_recovers_known_attack_ratings():
    """Spec: simulate 4,000 matches from known att/def/h/rho, fit, correlation of recovered att > 0.9."""
    frame, truth = _simulate(n_matches=4000, n_teams=10, seed=11)
    model = DixonColesModel(window_seasons=3, half_life_days=3650.0, gamma=0.05)
    model.fit(frame, as_of="2018-08-11")

    params = model.params[LEAGUE]
    recovered = np.array([params.att[params.index_of(t)] for t in truth["teams"]])
    corr = np.corrcoef(recovered, truth["att"])[0, 1]
    assert corr > 0.9, f"attack correlation only {corr:.3f}"


def test_recovers_known_defence_ratings():
    frame, truth = _simulate(n_matches=4000, n_teams=10, seed=12)
    model = DixonColesModel(window_seasons=3, half_life_days=3650.0, gamma=0.05)
    model.fit(frame, as_of="2018-08-11")

    params = model.params[LEAGUE]
    recovered = np.array([params.deff[params.index_of(t)] for t in truth["teams"]])
    corr = np.corrcoef(recovered, truth["deff"])[0, 1]
    assert corr > 0.9, f"defence correlation only {corr:.3f}"


def test_recovers_home_advantage():
    frame, truth = _simulate(n_matches=4000, n_teams=10, home=0.30, seed=13)
    model = DixonColesModel(window_seasons=3, half_life_days=3650.0, gamma=0.05)
    model.fit(frame, as_of="2018-08-11")
    assert model.params[LEAGUE].home == pytest.approx(0.30, abs=0.12)


def test_recovers_rho_sign():
    """A negative rho means 1-1 is over-represented and 0-1 under-represented."""
    frame, truth = _simulate(n_matches=4000, n_teams=10, rho=-0.12, seed=14)
    model = DixonColesModel(window_seasons=3, half_life_days=3650.0, gamma=0.05)
    model.fit(frame, as_of="2018-08-11")
    assert model.params[LEAGUE].rho < 0


def test_fitted_tau_factors_are_positive():
    frame, truth = _simulate(n_matches=4000, n_teams=10, seed=15)
    model = DixonColesModel(window_seasons=3, half_life_days=3650.0, gamma=0.05)
    model.fit(frame, as_of="2018-08-11")

    for h in truth["teams"]:
        for a in truth["teams"]:
            if h == a:
                continue
            lh, la, rho = model.lambdas(LEAGUE, h, a)
            x = np.array([0.0, 0.0, 1.0, 1.0])
            y = np.array([0.0, 1.0, 0.0, 1.0])
            logtau, _, _, _ = _tau_terms(x, y, np.full(4, lh), np.full(4, la), rho)
            assert (logtau > -1e8).all(), (h, a, lh, la, rho)
            assert np.isfinite(logtau).all()


def test_fit_recovers_rho_when_data_is_simulated_from_it():
    """The fitted rho should land near the simulated value."""
    frame, truth = _simulate(n_matches=4000, n_teams=10, rho=-0.10, seed=16)
    model = DixonColesModel(window_seasons=3, half_life_days=3650.0, gamma=0.05)
    model.fit(frame, as_of="2018-08-11")
    assert model.params[LEAGUE].rho == pytest.approx(-0.10, abs=0.08)


# ------------------------------------------------------- model behaviour

def test_predict_returns_valid_probabilities():
    frame, _ = _simulate(n_matches=600, n_teams=6, seed=17)
    model = DixonColesModel(window_seasons=3, half_life_days=3650.0, gamma=0.05)
    model.fit(frame, as_of="2018-08-11")

    fixtures = pd.DataFrame({
        "league": [LEAGUE] * 3,
        "season": ["1819"] * 3,
        "date": ["2018-08-20"] * 3,
        "home_id": [1, 2, 3],
        "away_id": [2, 3, 4],
    })
    out = model.predict(fixtures)
    assert len(out) == 3
    assert (out["pH"] + out["pD"] + out["pA"] - 1.0).abs().max() < 1e-9
    assert (out[["pH", "pD", "pA"]] > 0).all().all()
    assert (out["grid"].iloc[0].sum() - 1.0) < 1e-9


def test_unknown_league_abstains_with_flat_probabilities():
    model = DixonColesModel()
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


def test_fit_rejects_future_rows():
    frame, _ = _simulate(n_matches=100, n_teams=4, seed=18)
    model = DixonColesModel()
    with pytest.raises(ValueError):
        model.fit(frame, as_of="2018-01-01")


def test_known_team_predicts_more_wins_than_underdog():
    """A stronger attack/defence pair must get a higher home probability."""
    frame, _ = _simulate(n_matches=4000, n_teams=10, seed=19)
    model = DixonColesModel(window_seasons=3, half_life_days=3650.0, gamma=0.05)
    model.fit(frame, as_of="2018-08-11")

    params = model.params[LEAGUE]
    strongest = max(
        range(len(params.teams)),
        key=lambda i: params.att[i] - params.deff[i],
    )
    weakest = min(
        range(len(params.teams)),
        key=lambda i: params.att[i] - params.deff[i],
    )
    strong_id = params.teams[strongest]
    weak_id = params.teams[weakest]

    fixtures = pd.DataFrame({
        "league": [LEAGUE, LEAGUE],
        "season": ["1819", "1819"],
        "date": ["2018-08-20", "2018-08-20"],
        "home_id": [strong_id, weak_id],
        "away_id": [weak_id, strong_id],
    })
    out = model.predict(fixtures)
    assert out["pH"].iloc[0] > out["pH"].iloc[1]
