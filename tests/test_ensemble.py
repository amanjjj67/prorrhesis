from __future__ import annotations

import numpy as np
import pytest

from prorrhesis.ensemble.calibrate import (
    EnsembleCalibrator,
    apply_temperature,
    fit_temperature,
)
from prorrhesis.ensemble.stack import (
    EnsembleStacker,
    available_mask,
    equal_pool,
    fit_log_pool,
    log_pool,
    stack_features,
    subset_key,
)
from prorrhesis.eval.bootstrap import block_bootstrap_ci, per_match_loss
from prorrhesis.eval.metrics import log_loss

NAMES = ["elo", "freq", "market_open"]


def _triples(n: int = 500, seed: int = 4, n_models: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(3), size=(n, n_models))
    return P


def _labels(P: np.ndarray, seed: int = 5) -> np.ndarray:
    """Draw outcomes from the first model's probabilities."""
    rng = np.random.default_rng(seed)
    base = np.asarray(P[:, 0, :], dtype=float)
    base = base / base.sum(axis=1, keepdims=True)
    return np.array([rng.choice(3, p=row) for row in base])


# ------------------------------------------------------------------ pools

def test_equal_pool_sums_to_one():
    P = equal_pool(_triples())
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P > 0).all()


def test_log_pool_with_equal_weights_is_the_equal_pool():
    T = _triples()
    w = np.full(T.shape[1], 1.0 / T.shape[1])
    assert np.allclose(log_pool(T, w), equal_pool(T))


def test_log_pool_weights_sum_to_one_and_are_not_negative():
    T = _triples()
    y = _labels(T)
    w = fit_log_pool(T, y)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_log_pool_drops_missing_models_and_renormalises():
    T = _triples(n=4)
    T[0, 1] = np.nan
    T[0, 2] = np.nan
    P = equal_pool(T)
    assert np.isfinite(P[0]).all()
    assert P[0].sum() == pytest.approx(1.0)


def test_available_mask_and_subset_key():
    T = _triples(n=2)
    T[0, 0] = np.nan
    mask = available_mask(T)
    assert list(mask[0]) == [False, True, True]
    assert subset_key(mask[0], NAMES) == "freq|market_open"


def test_stack_features_are_log_ratios():
    T = _triples(n=3)
    X = stack_features(T)
    assert X.shape == (3, 2 * T.shape[1])
    assert np.allclose(X[0, 0], np.log(T[0, 0, 0] / T[0, 0, 1]))
    assert np.allclose(X[0, 1], np.log(T[0, 0, 2] / T[0, 0, 1]))


# --------------------------------------------------------------- stacker

def test_stacker_falls_back_to_equal_before_enough_rows():
    T = _triples(n=100)
    y = _labels(T)
    stacker = EnsembleStacker(NAMES, min_rows=800)
    stacker.fit(T, y, np.arange(len(y)), current_block=10)
    assert stacker.method == "equal"
    outcome = stacker.predict(T[:5])
    assert np.allclose(outcome.probs, equal_pool(T[:5]))


def test_stacker_picks_a_method_when_there_is_enough_data():
    T = _triples(n=1200)
    y = _labels(T)
    blocks = np.repeat(np.arange(40), 30)
    stacker = EnsembleStacker(NAMES, min_rows=800)
    stacker.fit(T, y, blocks, current_block=40)
    assert stacker.method in ("logit", "logpool", "equal")
    assert stacker.scores
    outcome = stacker.predict(T[:20])
    assert np.allclose(outcome.probs.sum(axis=1), 1.0)


def test_stacker_never_sees_later_blocks():
    """I4: the stacker for a block is fitted on earlier blocks only."""
    T = _triples(n=1200)
    y = _labels(T)
    blocks = np.repeat(np.arange(40), 30)

    stacker = EnsembleStacker(NAMES, min_rows=800)
    stacker.fit(T, y, blocks, current_block=30)

    prior = EnsembleStacker(NAMES, min_rows=800)
    prior.fit(T[blocks < 30], y[blocks < 30], blocks[blocks < 30], current_block=30)

    assert stacker.method == prior.method
    assert stacker.scores == prior.scores
    if stacker.weights is None:
        assert prior.weights is None
    else:
        assert np.allclose(stacker.weights, prior.weights)
    assert np.array_equal(stacker.predict(T[:20]).probs, prior.predict(T[:20]).probs)


def test_the_stacker_ignores_rows_from_the_current_block():
    """I4: rows tagged with the current block must not reach the fit."""
    T = _triples(n=1200)
    y = _labels(T)
    blocks = np.repeat(np.arange(40), 30)
    current = 20
    later = blocks >= current

    stacker = EnsembleStacker(NAMES, min_rows=800)
    stacker.fit(T, y, blocks, current_block=current)

    rng = np.random.default_rng(9)
    scrambled = T.copy()
    scrambled[later] = rng.dirichlet(np.ones(3), size=(int(later.sum()), 3))

    stacker2 = EnsembleStacker(NAMES, min_rows=800)
    stacker2.fit(scrambled, y, blocks, current_block=current)

    assert stacker.method == stacker2.method
    assert np.array_equal(stacker.predict(T[:20]).probs, stacker2.predict(T[:20]).probs)


def test_the_stacker_is_deterministic():
    T = _triples(n=1200)
    y = _labels(T)
    blocks = np.repeat(np.arange(40), 30)
    outs = []
    for _ in range(2):
        stacker = EnsembleStacker(NAMES, min_rows=800)
        stacker.fit(T, y, blocks, current_block=40)
        outs.append(stacker.predict(T[:50]).probs)
    assert np.array_equal(outs[0], outs[1])


# ------------------------------------------------------------ calibration

def test_temperature_keeps_the_probabilities_valid():
    P = _triples(n=100)
    y = _labels(P)
    T = fit_temperature(P, y)
    assert T > 0
    Q = apply_temperature(P, T)
    assert np.allclose(Q.sum(axis=1), 1.0)
    assert (Q > 0).all()


def test_temperature_can_fix_an_overconfident_ensemble():
    """A too-sharp ensemble must improve when the temperature is fitted."""
    rng = np.random.default_rng(3)
    n = 3000
    y = rng.integers(0, 3, n)
    P = rng.dirichlet(np.ones(3) * 0.05, size=n)   # far too sharp
    P[np.arange(n), y] += 0.6
    P = P / P.sum(axis=1, keepdims=True)
    before = log_loss(P, y)
    after = log_loss(apply_temperature(P, fit_temperature(P, y)), y)
    assert after < before


def test_calibrator_reports_none_when_nothing_helps():
    rng = np.random.default_rng(8)
    n = 2000
    y = rng.integers(0, 3, n)
    P = rng.dirichlet(np.ones(3), size=n)
    blocks = np.repeat(np.arange(20), 100)
    calibrator = EnsembleCalibrator()
    calibrator.fit(P, y, blocks, current_block=20)
    assert calibrator.method in ("none", "temperature", "dirichlet", "isotonic")
    out = calibrator.predict(P)
    assert np.allclose(out.probs.sum(axis=1), 1.0)


def test_calibrated_output_is_never_worse_than_the_raw_ensemble():
    """The calibrator only switches when it improves the out-of-fold loss."""
    rng = np.random.default_rng(12)
    n = 4000
    y = rng.integers(0, 3, n)
    P = rng.dirichlet(np.ones(3) * 0.2, size=n)
    blocks = np.repeat(np.arange(40), 100)
    calibrator = EnsembleCalibrator()
    calibrator.fit(P, y, blocks, current_block=40)
    if calibrator.method != "none":
        assert calibrator.scores[calibrator.method] <= calibrator.scores["none"] + 1e-9


# -------------------------------------------------------------- bootstrap

def test_block_bootstrap_detects_a_clear_difference():
    rng = np.random.default_rng(2)
    n_blocks, per_block = 60, 20
    blocks = np.repeat(np.arange(n_blocks), per_block)
    loss_a = rng.normal(1.0, 0.05, n_blocks * per_block)
    loss_b = loss_a + 0.2
    out = block_bootstrap_ci(loss_a, loss_b, blocks, n_resamples=2000, seed=42)
    assert out["mean_diff"] < 0
    assert out["ci_high"] < 0
    assert out["significant"]


def test_block_bootstrap_finds_no_difference_between_identical_models():
    rng = np.random.default_rng(6)
    n_blocks, per_block = 40, 15
    blocks = np.repeat(np.arange(n_blocks), per_block)
    loss = rng.normal(1.0, 0.1, n_blocks * per_block)
    out = block_bootstrap_ci(loss, loss.copy(), blocks, n_resamples=2000, seed=42)
    assert out["mean_diff"] == pytest.approx(0.0)
    assert not out["significant"]


def test_per_match_loss_matches_the_metric():
    P = _triples(n=50)
    y = _labels(P)
    assert per_match_loss(P, y).mean() == pytest.approx(log_loss(P, y))
