from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prorrhesis.models.market import (
    MarketCloseModel,
    MarketOpenModel,
    de_vig,
    proportional,
    shin,
)


def test_proportional_sums_to_one():
    p = proportional(np.array([2.0, 3.5, 4.0]))
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]


def test_shin_sums_to_one():
    p = shin(np.array([2.0, 3.5, 4.0]))
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()


def test_shin_equals_proportional_without_margin():
    odds = np.array([3.0, 3.0, 3.0])
    p = shin(odds)
    assert np.allclose(p, proportional(odds))


def test_shin_removes_more_than_proportional():
    # Shin gives the favourite more probability than the naive proportional book
    odds = np.array([1.5, 4.5, 7.0])
    assert shin(odds)[0] > proportional(odds)[0]


def test_invalid_odds_raise():
    with pytest.raises(ValueError):
        proportional(np.array([0.0, 3.5, 4.0]))
    with pytest.raises(ValueError):
        shin(np.array([np.nan, 3.5, 4.0]))


def test_market_models_abstain_when_odds_missing():
    fixtures = pd.DataFrame({
        "id": [1, 2],
        "league": ["E0", "E0"],
        "home_id": [1, 2],
        "away_id": [2, 3],
        "odds_close_h": [2.0, np.nan],
        "odds_close_d": [3.5, np.nan],
        "odds_close_a": [4.0, np.nan],
    })
    model = MarketCloseModel()
    out = model.predict(fixtures)
    assert out["status"].tolist() == ["ok", "abstain"]
    assert out.loc[0, "pH"] == pytest.approx(float(shin(np.array([2.0, 3.5, 4.0]))[0]))
    assert np.isnan(out.loc[1, "pH"])


def test_market_open_uses_open_odds():
    fixtures = pd.DataFrame({
        "id": [1],
        "league": ["E0"],
        "home_id": [1],
        "away_id": [2],
        "odds_open_h": [1.8],
        "odds_open_d": [3.6],
        "odds_open_a": [4.5],
    })
    out = MarketOpenModel().predict(fixtures)
    assert out["status"].tolist() == ["ok"]
    assert out.loc[0, ["pH", "pD", "pA"]].sum() == pytest.approx(1.0)


def test_de_vig_dispatch():
    odds = np.array([2.0, 3.5, 4.0])
    assert np.allclose(de_vig(odds, "proportional"), proportional(odds))
    assert np.allclose(de_vig(odds, "shin"), shin(odds))
