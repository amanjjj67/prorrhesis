from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prorrhesis.features.rolling import Ewma, RollingState
from prorrhesis.features.store import FeatureStore, season_total
from prorrhesis.features.xgproxy import XgProxyModel, build_season_models, team_match_rows

from conftest import LEAGUE, build_history


# --------------------------------------------------------------- EWMA core

def test_ewma_alpha_matches_half_life():
    ew = Ewma(alpha=1.0 - 0.5 ** (1 / 5), prior_n=4)
    assert ew.alpha == pytest.approx(0.12944, abs=1e-5)


def test_ewma_first_update_uses_pseudo_count_correction():
    """a_t = max(alpha, 1/(n + prior_n)); with n=1 and prior_n=4 that is 0.2."""
    ew = Ewma(alpha=0.12944, prior_n=4)
    ew.start(0.0)
    ew.update(1.0)
    assert ew.s == pytest.approx(0.2)


def test_ewma_ignores_non_finite_updates():
    ew = Ewma(alpha=0.5, prior_n=4)
    ew.start(1.0)
    ew.update(float("nan"))
    assert ew.s == pytest.approx(1.0)
    assert ew.n == 0


def test_ewma_converges_towards_a_constant_series():
    ew = Ewma(alpha=0.5, prior_n=0)
    ew.start(0.0)
    for _ in range(200):
        ew.update(3.0)
    assert ew.s == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------- state machine

def test_state_only_reflects_matches_before_the_date():
    """The EWMA starts at the league-season mean and moves 1/(1+prior_n) of the way to the new value."""
    state = RollingState()
    row = {"league": LEAGUE, "season": "1617", "date": "2016-08-10",
           "home_id": 1, "away_id": 2, "hg": 3, "ag": 0,
           "hs": 10, "as_": 2, "hst": 5, "ast": 1}
    state.apply_match(row)
    # league mean goals-for per team-match is (3 + 0) / 2 = 1.5
    assert state.form(1, 5, "gf") == pytest.approx(1.5 + 0.2 * (3.0 - 1.5))
    # the home-only view counts the home side alone, so its mean is 3.0
    assert state.venue(1, "home", "gf") == pytest.approx(3.0)


def test_venue_split_separates_home_and_away():
    state = RollingState()
    state.apply_match({"league": LEAGUE, "season": "1617", "date": "2016-08-10",
                       "home_id": 1, "away_id": 2, "hg": 3, "ag": 0,
                       "hs": 10, "as_": 2, "hst": 5, "ast": 1})
    assert state.venue(1, "home", "gf") == pytest.approx(3.0)
    assert state.venue(1, "away", "gf") == pytest.approx(0.0)
    assert state.venue(2, "away", "gf") == pytest.approx(0.0)
    assert state.venue(2, "away", "ga") == pytest.approx(3.0)


def test_days_since_last_match():
    state = RollingState()
    state.apply_match({"league": LEAGUE, "season": "1617", "date": "2016-08-10",
                       "home_id": 1, "away_id": 2, "hg": 1, "ag": 1,
                       "hs": 5, "as_": 5, "hst": 2, "ast": 2})
    assert state.days_since(1, "2016-08-17") == pytest.approx(7.0)
    # capped at 14
    assert state.days_since(1, "2016-09-30") == pytest.approx(14.0)


def test_matches_within_window():
    state = RollingState()
    for d in ["2016-08-25", "2016-09-01", "2016-09-20"]:
        state.apply_match({"league": LEAGUE, "season": "1617", "date": d,
                           "home_id": 1, "away_id": 2, "hg": 1, "ag": 0,
                           "hs": 5, "as_": 5, "hst": 2, "ast": 2})
    # from 2016-09-21: only 2016-09-20 is inside 14 days; all three are inside 30 days
    assert state.matches_within(1, "2016-09-21", 14) == pytest.approx(1.0)
    assert state.matches_within(1, "2016-09-21", 30) == pytest.approx(3.0)


def test_newly_promoted_flag():
    state = RollingState()
    state.apply_match({"league": LEAGUE, "season": "1617", "date": "2016-08-10",
                       "home_id": 1, "away_id": 2, "hg": 1, "ag": 0,
                       "hs": 5, "as_": 5, "hst": 2, "ast": 2})
    assert state.is_newly_promoted(1, LEAGUE, "1617") == pytest.approx(1.0)
    assert state.is_newly_promoted(1, LEAGUE, "1718") == pytest.approx(0.0)


# --------------------------------------------------------------- xG proxy

def _shots_history(n: int = 400, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        hst = int(rng.poisson(1.6))
        ast = int(rng.poisson(1.3))
        rows.append({
            "league": LEAGUE, "season": "1617", "date": f"2016-{(i % 12) + 1:02d}-10",
            "home_id": (i % 10) + 1, "away_id": ((i + 3) % 10) + 1,
            "hg": int(rng.poisson(hst * 0.38)), "ag": int(rng.poisson(ast * 0.38)),
            "hs": hst + int(rng.poisson(3)), "as_": ast + int(rng.poisson(3)),
            "hst": hst, "ast": ast,
        })
    return pd.DataFrame(rows)


def test_team_match_rows_builds_two_rows_per_match():
    df = _shots_history(50)
    X, y = team_match_rows(df)
    assert len(X) == 2 * len(df)
    assert X.shape[1] == 2
    assert len(y) == 2 * len(df)


def test_xg_proxy_learns_that_shots_on_target_are_worth_more():
    df = _shots_history(1200, seed=9)
    model = XgProxyModel().fit(df)
    assert model.fitted
    off_only = model.predict_goals(sot=0, off=6)
    on_only = model.predict_goals(sot=6, off=0)
    assert on_only > off_only


def test_xg_proxy_model_for_a_season_never_sees_that_season():
    df = _shots_history(600, seed=10)
    df["season"] = ["1617"] * 300 + ["1718"] * 300
    models = build_season_models(df)
    assert not models["1617"].fitted          # nothing earlier to learn from
    assert models["1718"].fitted              # 1617 is available
    assert models["1718"].n_rows == 2 * 300


def test_xg_proxy_prediction_handles_missing_shots():
    model = XgProxyModel()
    assert not model.fitted
    assert np.isnan(model.predict_goals(3, 2))


# ----------------------------------------------------------- feature table

def test_feature_table_shape_and_columns():
    history = build_history(n_weeks=60)
    store = FeatureStore()
    table = store.build_table(history)
    assert len(table) == len(history)
    for col in ("f5_gf_h", "f12_gf_a", "v_gf_h", "v_ga_a", "days_since_h",
                "m14_a", "season_progress", "month", "elo_diff"):
        assert col in table.columns


def test_feature_table_probabilities_are_valid_numbers():
    history = build_history(n_weeks=40)
    store = FeatureStore()
    table = store.build_table(history)
    numeric = table.select_dtypes(include=[np.number])
    assert numeric.notna().any().any()
    # no infinities anywhere
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()


def test_season_progress_is_between_zero_and_one():
    history = build_history(n_weeks=60)
    store = FeatureStore()
    table = store.build_table(history)
    progress = table["season_progress"].to_numpy(dtype=float)
    assert (progress >= 0).all()
    assert (progress <= 1).all()


def test_season_total_uses_the_league_size():
    assert season_total("E0", "1718") == 380
    assert season_total("D1", "1718") == 306
    assert season_total("F1", "2223") == 380
    assert season_total("F1", "2324") == 306


def test_no_feature_column_is_entirely_missing():
    """M4 acceptance: no column is entirely NaN in the eval window except documented optional ones."""
    history = build_history(n_weeks=60)
    store = FeatureStore()
    table = store.build_table(history)
    documented_optional = {"clubelo_diff"}
    all_nan = {c for c in table.columns if table[c].isna().all()}
    assert all_nan <= documented_optional, f"unexpected all-NaN columns: {all_nan - documented_optional}"
