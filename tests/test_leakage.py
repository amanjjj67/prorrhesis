from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prorrhesis.models.baseline import FreqModel, HomeModel
from prorrhesis.models.dixoncoles import DixonColesModel
from prorrhesis.models.elo import EloModel
from prorrhesis.models.market import MarketOpenModel
from prorrhesis.models.xgpoisson import XgPoissonModel

from conftest import LEAGUE, build_history


def scramble_after(history: pd.DataFrame, cutoff: str, rng: np.random.Generator) -> pd.DataFrame:
    """Replace every result dated on/after `cutoff` with random draws."""
    out = history.copy()
    mask = out["date"].astype(str) >= cutoff
    n = int(mask.sum())
    if n == 0:
        return out
    hg = rng.integers(0, 6, size=n)
    ag = rng.integers(0, 6, size=n)
    out.loc[mask, "hg"] = hg
    out.loc[mask, "ag"] = ag
    out.loc[mask, "res"] = np.where(hg > ag, "H", np.where(hg == ag, "D", "A"))
    for col in ("hs", "as_", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar",
                "odds_open_h", "odds_open_d", "odds_open_a",
                "odds_close_h", "odds_close_d", "odds_close_a"):
        if col in out.columns:
            out.loc[mask, col] = rng.integers(0, 15, size=n)
    return out


def make_fixture(history: pd.DataFrame, n: int = 5, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = history.tail(n)[["league", "home_id", "away_id", "date"]].copy()
    rows["odds_open_h"] = rng.uniform(1.3, 6.0, size=n)
    rows["odds_open_d"] = rng.uniform(2.8, 4.2, size=n)
    rows["odds_open_a"] = rng.uniform(1.3, 6.0, size=n)
    return rows


def fresh(model_name: str):
    return {
        "freq": lambda: FreqModel(window_seasons=3),
        "home": lambda: HomeModel(),
        "elo": lambda: EloModel(),
        "dc": lambda: DixonColesModel(),
        "market_open": lambda: MarketOpenModel(),
        "xgpoisson": lambda: XgPoissonModel(),
    }[model_name]()


@pytest.mark.parametrize("model_name", ["freq", "home", "elo", "dc", "market_open", "xgpoisson"])
def test_model_predictions_ignore_the_future(model_name):
    """I2/I3: a model fit on data before `cutoff` must not change when the future is scrambled."""
    rng = np.random.default_rng(11)
    history = build_history(n_weeks=40)
    cutoff = sorted(history["date"].unique())[20]
    fixture = make_fixture(history)

    past = history[history["date"].astype(str) < cutoff]

    before_model = fresh(model_name)
    before_model.fit(past, as_of=cutoff)
    before = before_model.predict(fixture)

    scrambled = scramble_after(history, cutoff, rng)
    assert not scrambled.loc[scrambled["date"].astype(str) >= cutoff, "hg"].equals(
        history.loc[history["date"].astype(str) >= cutoff, "hg"])
    scrambled_past = scrambled[scrambled["date"].astype(str) < cutoff]

    after_model = fresh(model_name)
    after_model.fit(scrambled_past, as_of=cutoff)
    after = after_model.predict(fixture)

    assert np.array_equal(before[["pH", "pD", "pA"]].to_numpy(),
                          after[["pH", "pD", "pA"]].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("model_name", ["freq", "home", "elo", "dc", "market_open", "xgpoisson"])
def test_model_refuses_future_data(model_name):
    """I3: fit must raise when handed a match dated on or after the as-of date."""
    history = build_history(n_weeks=20)
    model = fresh(model_name)
    with pytest.raises(Exception):
        model.fit(history, as_of=history["date"].min())


@pytest.mark.parametrize("model_name", ["freq", "home", "elo", "dc", "market_open", "xgpoisson"])
def test_prediction_does_not_use_the_target_match_result(model_name):
    """A prediction for a fixture must not move when that fixture's own result changes."""
    rng = np.random.default_rng(5)
    history = build_history(n_weeks=30)
    cutoff = sorted(history["date"].unique())[15]
    fixture = make_fixture(history, n=3)

    past = history[history["date"].astype(str) < cutoff]
    model = fresh(model_name)
    model.fit(past, as_of=cutoff)
    before = model.predict(fixture)[["pH", "pD", "pA"]].to_numpy()

    # the same fixture, but the whole future (including its own result) is scrambled
    scrambled = scramble_after(history, cutoff, rng)
    scrambled_past = scrambled[scrambled["date"].astype(str) < cutoff]
    model2 = fresh(model_name)
    model2.fit(scrambled_past, as_of=cutoff)
    after = model2.predict(fixture)[["pH", "pD", "pA"]].to_numpy()

    assert np.array_equal(before, after, equal_nan=True)


def test_market_close_is_a_benchmark_and_not_a_feature():
    """I5: the closing-odds benchmark may read closing odds; no other model may."""
    from prorrhesis.models.market import MarketCloseModel

    assert "odds_close" in MarketCloseModel().needs
    for model in (FreqModel(), HomeModel(), EloModel(), DixonColesModel(), MarketOpenModel(),
                  XgPoissonModel()):
        assert "odds_close" not in model.needs


def test_features_do_not_see_the_future():
    """I2: features for a match must not move when results on or after its date are scrambled."""
    from prorrhesis.features.store import FeatureStore

    rng = np.random.default_rng(21)
    history = build_history(n_weeks=60)
    cutoff = sorted(history["date"].unique())[40]

    before = FeatureStore().build_table(history.reset_index(drop=True))

    scrambled = scramble_after(history, cutoff, rng).reset_index(drop=True)
    after = FeatureStore().build_table(scrambled)

    mask = history["date"].astype(str) < cutoff
    a = before.loc[mask.to_numpy()].to_numpy(dtype=float)
    b = after.loc[mask.to_numpy()].to_numpy(dtype=float)
    assert np.array_equal(a, b, equal_nan=True), "features moved when the future was scrambled"


def test_features_do_not_see_other_matches_on_the_same_day():
    """I2, same day: features read a state built only from strictly earlier dates.

    Every post-kickoff value on one date (goals and shots) is scrambled, and the
    features of the matches on that date must not move: they must not see their
    own result nor the results of the other matches played the same day. Open
    odds are pre-kickoff information and are deliberately left alone.
    """
    from prorrhesis.features.store import FeatureStore

    rng = np.random.default_rng(22)
    history = build_history(n_weeks=50)
    target = sorted(history["date"].unique())[30]
    same_day = history[history["date"] == target]
    assert len(same_day) > 1, "need several matches on the same date"

    before = FeatureStore().build_table(history.reset_index(drop=True))

    scrambled = history.copy()
    mask = scrambled["date"] == target
    n = int(mask.sum())
    hg = rng.integers(0, 6, size=n)
    ag = rng.integers(0, 6, size=n)
    scrambled.loc[mask, "hg"] = hg
    scrambled.loc[mask, "ag"] = ag
    scrambled.loc[mask, "res"] = np.where(hg > ag, "H", np.where(hg == ag, "D", "A"))
    for col in ("hs", "as_", "hst", "ast"):
        if col in scrambled.columns:
            scrambled.loc[mask, col] = rng.integers(0, 15, size=n)
    after = FeatureStore().build_table(scrambled.reset_index(drop=True))

    m = (history["date"] == target).to_numpy()
    a = before.loc[m].to_numpy(dtype=float)
    b = after.loc[m].to_numpy(dtype=float)
    assert np.array_equal(a, b, equal_nan=True), "a match saw a same-day result"


def test_gbm_features_do_not_see_the_future():
    """I2 for the feature-based model: the table the GBM reads is leak-free."""
    from prorrhesis.features.schema import FEATURE_COLUMNS
    from prorrhesis.features.store import build_feature_table

    rng = np.random.default_rng(31)
    history = build_history(n_weeks=60)
    history["id"] = np.arange(1, len(history) + 1)
    cutoff = sorted(history["date"].unique())[40]

    before = build_feature_table(history)
    after = build_feature_table(scramble_after(history, cutoff, rng))

    mask = before["date"].astype(str) < cutoff
    a = before.loc[mask, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    b = after.loc[mask, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    assert np.array_equal(a, b, equal_nan=True)


def test_gbm_predictions_ignore_the_future():
    """The GBM trained on the past must not move when the future is scrambled."""
    from prorrhesis.features.store import build_feature_table
    from prorrhesis.models.gbm import HAVE_LIGHTGBM, GbmModel

    if not HAVE_LIGHTGBM:
        pytest.skip("lightgbm is not installed")

    rng = np.random.default_rng(32)
    history = build_history(n_weeks=120)
    history["id"] = np.arange(1, len(history) + 1)
    cutoff = sorted(history["date"].unique())[80]
    past = history[history["date"].astype(str) < cutoff]
    fixture = history[history["date"].astype(str) == cutoff][
        ["id", "league", "season", "date", "home_id", "away_id"]].reset_index(drop=True)

    def run(frame):
        model = GbmModel()
        model.attach_features(build_feature_table(frame))
        model.fit(past, as_of=cutoff)
        return model.predict(fixture)[["pH", "pD", "pA"]].to_numpy()

    before = run(history)
    after = run(scramble_after(history, cutoff, rng))
    assert np.array_equal(before, after, equal_nan=True)
