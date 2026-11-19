from __future__ import annotations

import numpy as np
import pytest

from prorrhesis.ensemble.stack import (
    EnsembleStacker,
    available_mask,
    equal_pool,
    fit_log_pool,
    log_pool,
    subset_key,
)

NAMES = ["elo", "freq", "market_open"]


def _triples(n: int = 1200, seed: int = 4, n_models: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(3), size=(n, n_models))


def _labels(P: np.ndarray, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.asarray(P[:, 0, :], dtype=float)
    base = base / base.sum(axis=1, keepdims=True)
    return np.array([rng.choice(3, p=row) for row in base])


def test_stacker_for_block_b_ignores_oof_rows_of_block_b_or_later():
    """I4: corrupting the rows of the current and later blocks must not move
    the stacker for block b."""
    T = _triples()
    y = _labels(T)
    blocks = np.repeat(np.arange(40), 30)
    current = 25

    stacker = EnsembleStacker(NAMES, min_rows=800)
    stacker.fit(T, y, blocks, current_block=current)
    clean = stacker.predict(T[:20]).probs

    rng = np.random.default_rng(11)
    corrupted = T.copy()
    later = blocks >= current
    corrupted[later] = rng.dirichlet(np.ones(3), size=(int(later.sum()), 3))

    stacker2 = EnsembleStacker(NAMES, min_rows=800)
    stacker2.fit(corrupted, y, blocks, current_block=current)

    assert stacker2.method == stacker.method
    assert np.array_equal(clean, stacker2.predict(T[:20]).probs)


def test_log_pool_weights_sum_to_one():
    T = _triples()
    y = _labels(T)
    w = fit_log_pool(T, y)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_missing_models_are_dropped_and_the_rest_renormalised():
    T = _triples(n=5)
    T[0, 0] = np.nan
    T[0, 2] = np.nan
    mask = available_mask(T)
    assert list(mask[0]) == [False, True, False]

    w = np.array([0.5, 0.3, 0.2])
    P = log_pool(T, w)
    assert np.isfinite(P[0]).all()
    assert P[0].sum() == pytest.approx(1.0)
    assert np.allclose(P[0], T[0, 1] / T[0, 1].sum())

    eq = equal_pool(T)
    assert eq[0].sum() == pytest.approx(1.0)


def test_a_subset_without_enough_rows_falls_back_to_the_pool():
    """A subset with fewer than 400 rows must not get its own stacker."""
    T = _triples(n=500)
    y = _labels(T)
    blocks = np.repeat(np.arange(20), 25)
    mask = available_mask(T)
    mask[::2, 1] = False
    T[::2, 1] = np.nan

    stacker = EnsembleStacker(NAMES, min_rows=100)
    stacker.fit(T, y, blocks, current_block=20)
    small = [k for k, v in stacker.subset_rows.items() if v < 400]
    assert small == []
    assert "freq" not in stacker.subset_rows


def test_subset_key_names_the_available_models():
    mask = np.array([True, False, True])
    assert subset_key(mask, NAMES) == "elo|market_open"
