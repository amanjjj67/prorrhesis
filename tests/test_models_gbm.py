from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prorrhesis.features.store import FEATURE_COLUMNS, build_feature_table
from prorrhesis.models.gbm import GBM_PARAMS, HAVE_LIGHTGBM, GbmModel

from conftest import LEAGUE, build_history

pytestmark = pytest.mark.skipif(not HAVE_LIGHTGBM, reason="lightgbm is not installed")


def _history(n_weeks: int = 200, seed: int = 5) -> pd.DataFrame:
    history = build_history(n_weeks=n_weeks, seed=seed)
    history = history.copy()
    history["id"] = np.arange(1, len(history) + 1)
    return history


def _table(history: pd.DataFrame) -> pd.DataFrame:
    return build_feature_table(history)


def _fixtures(history: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    rows = history.tail(n)[["id", "league", "season", "date", "home_id", "away_id"]].copy()
    return rows.reset_index(drop=True)


# ------------------------------------------------------------ feature table

def test_the_feature_table_has_the_documented_columns():
    table = _table(_history(n_weeks=40))
    assert list(table.columns[:2]) == ["match_id", "date"]
    assert set(FEATURE_COLUMNS).issubset(set(table.columns))
    assert len(FEATURE_COLUMNS) == 52


def test_the_feature_table_covers_every_match_once():
    history = _history(n_weeks=40)
    table = _table(history)
    assert len(table) == len(history)
    assert table["match_id"].is_unique


# ------------------------------------------------------------------ model

def test_fit_without_features_is_an_error():
    model = GbmModel()
    with pytest.raises(ValueError):
        model.fit(_history(n_weeks=40), as_of="2017-01-01")


def test_predict_without_a_model_abstains():
    model = GbmModel()
    out = model.predict(_fixtures(_history(n_weeks=40)))
    assert (out["status"] == "abstain").all()


def test_fit_and_predict_produce_valid_probabilities():
    history = _history()
    table = _table(history)
    model = GbmModel()
    model.attach_features(table)
    model.fit(history, as_of="2020-06-01")

    out = model.predict(_fixtures(history))
    assert (out["status"] == "ok").all()
    assert (out["pH"] + out["pD"] + out["pA"] - 1.0).abs().max() < 1e-6
    assert (out[["pH", "pD", "pA"]] > 0).all().all()


def test_the_market_columns_are_only_in_the_augmented_variant():
    plain = GbmModel()
    augmented = GbmModel(use_market=True)
    assert "mkt_open_h" not in plain.feature_names
    assert "mkt_open_h" in augmented.feature_names
    assert len(augmented.feature_names) == len(plain.feature_names) + 3


def test_the_refit_cadence_is_respected():
    history = _history()
    model = GbmModel(refit_weeks=4)
    model.attach_features(_table(history))
    past = history[history["date"] < "2019-01-01"]

    model.fit(past, as_of="2019-01-01")
    assert model.refits == 1
    model.fit(past, as_of="2019-01-08")
    assert model.refits == 1, "the model must be reused between refits"
    model.fit(past, as_of="2019-02-05")
    assert model.refits == 2


def test_early_stopping_uses_the_most_recent_rows():
    """The validation set is the tail of the training window, never a random split."""
    history = _history()
    table = _table(history)
    model = GbmModel()
    model.attach_features(table)
    model.fit(history, as_of="2020-06-01")
    assert model.booster is not None
    assert model.best_iteration is not None
    assert model.best_iteration <= 500


def test_a_fixture_without_features_is_abstained():
    history = _history()
    model = GbmModel()
    model.attach_features(_table(history))
    model.fit(history, as_of="2020-06-01")

    unknown = pd.DataFrame({
        "id": [999999], "league": [LEAGUE], "season": ["1819"],
        "date": ["2020-01-02"], "home_id": [1], "away_id": [2],
    })
    out = model.predict(unknown)
    assert out["status"].iloc[0] == "abstain"


# ------------------------------------------------------------ importances

def test_importances_add_up_and_are_not_negative():
    history = _history()
    model = GbmModel()
    model.attach_features(_table(history))
    model.fit(history, as_of="2020-06-01")
    assert model.importances
    assert all(v >= 0 for v in model.importances.values())
    assert sum(model.importances.values()) == pytest.approx(1.0, abs=1e-6)


def test_noise_features_do_not_dominate():
    """Spec: Elo and form dominate; random noise features do not."""
    history = _history()
    table = _table(history)
    rng = np.random.default_rng(3)
    table["noise"] = rng.normal(0.0, 1.0, len(table))

    columns = (*FEATURE_COLUMNS, "noise")
    model = GbmModel(feature_columns=columns)
    model.attach_features(table)
    model.fit(history, as_of="2020-06-01")

    top = model.top_features(5)
    names = [n for n, _ in top]
    assert "noise" not in names, f"a noise feature reached the top: {top}"
    assert model.importances.get("noise", 0.0) < 0.2


def test_contributions_have_one_row_per_fixture():
    history = _history()
    table = _table(history)
    model = GbmModel()
    model.attach_features(table)
    model.fit(history, as_of="2020-06-01")
    fixtures = _fixtures(history)
    rows = table.set_index("match_id").reindex(fixtures["id"].tolist())
    contrib = model.contributions(rows)
    assert contrib.shape == (len(fixtures), (len(model.feature_names) + 1) * 3)
