from __future__ import annotations

import math

import numpy as np
import pytest

from prorrhesis.eval.metrics import accuracy, brier, ece, log_loss, rps, skill_score

P = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
Y = np.array([0, 2])


def test_log_loss_hand_computed():
    expected = -(math.log(0.7) + math.log(0.7)) / 2
    assert log_loss(P, Y) == pytest.approx(expected)


def test_log_loss_uniform_is_ln3():
    U = np.array([[1 / 3, 1 / 3, 1 / 3]] * 4)
    assert log_loss(U, np.array([0, 1, 2, 0])) == pytest.approx(math.log(3))
    assert math.log(3) == pytest.approx(1.0986, abs=1e-4)


def test_brier_hand_computed():
    # first row: (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.14
    # second row: 0.1^2 + 0.2^2 + (0.7-1)^2 = 0.14
    assert brier(P, Y) == pytest.approx(0.14)


def test_rps_hand_computed():
    # row 0: cum P = (0.7, 0.9), cum Y = (1, 1) -> 0.5*((0.3)^2+(0.1)^2) = 0.05
    assert rps(P[:1], Y[:1]) == pytest.approx(0.05)


def test_accuracy():
    assert accuracy(P, Y) == pytest.approx(1.0)
    assert accuracy(P, np.array([1, 1])) == pytest.approx(0.0)


def test_ece_is_zero_for_a_correct_one_hot_forecast():
    n = 20
    P = np.tile([1.0, 0.0, 0.0], (n, 1))
    y = np.zeros(n, dtype=int)
    assert ece(P, y, bins=5) == pytest.approx(0.0, abs=1e-12)


def test_ece_is_positive_for_a_wrong_forecast():
    n = 20
    P = np.tile([1.0, 0.0, 0.0], (n, 1))
    y = np.ones(n, dtype=int)
    assert ece(P, y, bins=5) > 0.0


def test_skill_score():
    assert skill_score(0.9, 1.0) == pytest.approx(0.1)
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)


def test_metrics_on_empty():
    assert math.isnan(log_loss(np.empty((0, 3)), np.empty((0,), dtype=int)))
