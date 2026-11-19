"""Calibration of the ensemble output (spec section 9.5).

Three calibrators, all fitted on prior out-of-fold ensemble rows only:

  temperature  one parameter, p proportional to p ** (1/T);
  dirichlet    multinomial logistic regression on log p, with an L2 penalty;
  isotonic     one-vs-rest isotonic regression, then renormalised.

A "none" option always exists: calibration is used only when it improves the
out-of-fold log loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-9
EVAL_FRACTION = 0.2
DEFAULT_METHODS = ("none", "temperature", "dirichlet", "isotonic")
MIN_ROWS = {"temperature": 50, "dirichlet": 1500, "isotonic": 3000}


def _log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, dtype=float), EPS, 1.0))


def _loss(P: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    ll = -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))
    if sample_weight is None:
        return float(np.mean(ll))
    total = float(np.sum(sample_weight))
    return float(np.sum(sample_weight * ll) / total) if total > 0 else float(np.mean(ll))


def fit_temperature(P: np.ndarray, y: np.ndarray,
                    sample_weight: np.ndarray | None = None) -> float:
    """T that minimises the weighted log loss of p ** (1/T)."""
    logp = _log(P)

    def objective(log_t: np.ndarray) -> float:
        T = float(np.exp(log_t[0]))
        powered = logp / T
        powered = powered - powered.max(axis=1, keepdims=True)
        Q = np.exp(powered)
        Q = Q / Q.sum(axis=1, keepdims=True)
        return _loss(Q, y, sample_weight)

    res = minimize(objective, np.array([0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 500})
    return float(np.exp(res.x[0]))


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    logp = _log(P) / max(float(T), 1e-6)
    logp = logp - logp.max(axis=1, keepdims=True)
    Q = np.exp(logp)
    total = Q.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return Q / total


class DirichletCalibrator:
    """Multinomial logistic regression on the log probabilities."""

    def __init__(self, C: float = 1.0, seed: int = 42):
        self.C = float(C)
        self.seed = int(seed)
        self.model = None

    def fit(self, P: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "DirichletCalibrator":
        self.model = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        self.model.fit(_log(P), y, sample_weight=sample_weight)
        return self

    def predict(self, P: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.asarray(P, dtype=float)
        out = np.zeros((len(P), 3))
        for j, cls in enumerate(self.model.classes_):
            if 0 <= int(cls) < 3:
                out[:, int(cls)] = self.model.predict_proba(_log(P))[:, j]
        total = out.sum(axis=1, keepdims=True)
        total[total <= 0] = 1.0
        return out / total


class IsotonicCalibrator:
    """One-vs-rest isotonic regression, then renormalised."""

    def __init__(self):
        self.models: list[IsotonicRegression] = []

    def fit(self, P: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.models = []
        for k in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(P[:, k], (y == k).astype(float))
            self.models.append(iso)
        return self

    def predict(self, P: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.asarray(P, dtype=float)
        out = np.column_stack([m.predict(P[:, k]) for k, m in enumerate(self.models)])
        out = np.clip(out, 0.0, None)
        total = out.sum(axis=1, keepdims=True)
        total[total <= 0] = 1.0
        return out / total


@dataclass
class CalibrationOutcome:
    probs: np.ndarray
    method: str
    scores: dict[str, float] = field(default_factory=dict)


class EnsembleCalibrator:
    """Picks the calibration method per block using prior rows only."""

    def __init__(self, methods: list[str] | None = None, min_rows: dict | None = None,
                 C: float = 1.0, seed: int = 42, recency_half_life_blocks: int = 40):
        self.methods = list(methods or DEFAULT_METHODS)
        self.min_rows = dict(MIN_ROWS if min_rows is None else min_rows)
        self.C = float(C)
        self.seed = int(seed)
        self.recency_half_life_blocks = int(recency_half_life_blocks)
        self.method: str = "none"
        self.scores: dict[str, float] = {}
        self.temperature: float | None = None
        self.dirichlet: DirichletCalibrator | None = None
        self.isotonic: IsotonicCalibrator | None = None

    # -- fitting -----------------------------------------------------------
    def _fit_method(self, method: str, P: np.ndarray, y: np.ndarray,
                    sample_weight: np.ndarray | None) -> None:
        if method == "temperature":
            self.temperature = fit_temperature(P, y, sample_weight)
        elif method == "dirichlet":
            self.dirichlet = DirichletCalibrator(C=self.C, seed=self.seed).fit(
                P, y, sample_weight)
        elif method == "isotonic":
            self.isotonic = IsotonicCalibrator().fit(P, y)

    def _apply_method(self, method: str, P: np.ndarray) -> np.ndarray:
        if method == "temperature" and self.temperature is not None:
            return apply_temperature(P, self.temperature)
        if method == "dirichlet" and self.dirichlet is not None:
            return self.dirichlet.predict(P)
        if method == "isotonic" and self.isotonic is not None:
            return self.isotonic.predict(P)
        return np.asarray(P, dtype=float)

    def fit(self, P: np.ndarray, y: np.ndarray, blocks: np.ndarray,
            current_block: int) -> None:
        self.method = "none"
        self.scores = {}
        self.temperature = None
        self.dirichlet = None
        self.isotonic = None

        if P is None or len(P) == 0:
            return
        blocks = np.asarray(blocks, dtype=float)
        prior = blocks < float(current_block)
        if not prior.all():
            P = np.asarray(P, dtype=float)[prior]
            y = np.asarray(y)[prior]
            blocks = blocks[prior]
            if len(P) == 0:
                return
        age = np.maximum(current_block - blocks, 0.0)
        sample_weight = 0.5 ** (age / self.recency_half_life_blocks)

        order = np.argsort(np.asarray(blocks, dtype=float))
        cut = max(int(len(order) * (1.0 - EVAL_FRACTION)), 1)
        train_idx, eval_idx = order[:cut], order[cut:]
        if len(eval_idx) < 50 or len(train_idx) < 100:
            train_idx, eval_idx = order, order

        scores: dict[str, float] = {}
        for method in self.methods:
            if method == "none":
                scores["none"] = _loss(P[eval_idx], y[eval_idx])
                continue
            if len(P) < self.min_rows.get(method, 0):
                continue
            probe = EnsembleCalibrator(methods=[method], min_rows=self.min_rows,
                                       C=self.C, seed=self.seed)
            probe._fit_method(method, P[train_idx], y[train_idx], sample_weight[train_idx])
            scores[method] = _loss(probe._apply_method(method, P[eval_idx]), y[eval_idx])

        self.scores = scores
        if scores:
            best = min(scores, key=lambda k: scores[k])
            self.method = best

        if self.method != "none":
            self._fit_method(self.method, P, y, sample_weight)

    def predict(self, P: np.ndarray) -> CalibrationOutcome:
        out = self._apply_method(self.method, P)
        return CalibrationOutcome(probs=out, method=self.method, scores=dict(self.scores))
