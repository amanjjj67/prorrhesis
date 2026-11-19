"""Ensemble stacking (spec section 9).

Two stackers and one baseline:

  A  multinomial logistic regression on the log-probability ratios of the
     base models, one model per available-model subset;
  B  logarithmic opinion pool, weights fitted on prior out-of-fold rows;
  C  equal-weight opinion pool, the fallback before enough rows exist.

Which one is used is re-decided for every block using only rows from earlier
blocks (invariant I4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

EPS = 1e-9
LOGISTIC_MIN_ROWS = 400
RECENCY_HALF_LIFE_BLOCKS = 40
EVAL_FRACTION = 0.2


def _log_probs(triples: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(triples, dtype=float), EPS, 1.0))


def _normalise(logp: np.ndarray) -> np.ndarray:
    shifted = logp - np.nanmax(logp, axis=1, keepdims=True)
    P = np.exp(shifted)
    total = P.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return P / total


def available_mask(triples: np.ndarray) -> np.ndarray:
    """True where a model produced a triple for a match."""
    return np.isfinite(np.asarray(triples, dtype=float)[:, :, 0])


def subset_key(mask: np.ndarray, names: list[str]) -> str:
    """Cache key for the set of models a match actually has."""
    return "|".join(sorted(names[i] for i in range(len(names)) if mask[i]))


def stack_features(triples: np.ndarray) -> np.ndarray:
    """log(pH/pD) and log(pA/pD) per model, concatenated."""
    logp = _log_probs(triples)
    feats = []
    for k in range(triples.shape[1]):
        feats.append(logp[:, k, 0] - logp[:, k, 1])
        feats.append(logp[:, k, 2] - logp[:, k, 1])
    return np.column_stack(feats)


def equal_pool(triples: np.ndarray) -> np.ndarray:
    """Equal-weight logarithmic opinion pool."""
    logp = _log_probs(triples)
    mask = available_mask(triples).astype(float)
    weights = mask / np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return _normalise(np.nansum(weights[:, :, None] * logp, axis=1))


def log_pool(triples: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted logarithmic opinion pool; missing models are dropped and the
    remaining weights renormalised."""
    logp = _log_probs(triples)
    mask = available_mask(triples)
    wmat = np.where(mask, w[None, :], 0.0)
    total = wmat.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    wmat = wmat / total
    return _normalise(np.nansum(wmat[:, :, None] * logp, axis=1))


def fit_log_pool(triples: np.ndarray, y: np.ndarray,
                 sample_weight: np.ndarray | None = None) -> np.ndarray:
    """Fit the pool weights on the simplex by minimising weighted log loss."""
    n_models = triples.shape[1]

    def objective(w: np.ndarray) -> float:
        P = log_pool(triples, w)
        ll = -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))
        if sample_weight is None:
            return float(np.mean(ll))
        total = float(np.sum(sample_weight))
        if total <= 0:
            return float(np.mean(ll))
        return float(np.sum(sample_weight * ll) / total)

    x0 = np.full(n_models, 1.0 / max(n_models, 1))
    res = minimize(
        objective, x0, method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"maxiter": 200, "ftol": 1e-10},
    )
    w = np.clip(res.x, 0.0, 1.0)
    total = w.sum()
    return w / total if total > 0 else x0


class LogisticStacker:
    """Multinomial logistic regression over the log-probability ratios."""

    def __init__(self, C: float = 1.0, seed: int = 42):
        self.C = float(C)
        self.seed = int(seed)
        self.model = None
        self.classes_: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "LogisticStacker":
        self.model = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        self.model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = [int(c) for c in self.model.classes_]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full((len(X), 3), 1 / 3)
        P = self.model.predict_proba(X)
        out = np.zeros((len(X), 3))
        for j, cls in enumerate(self.classes_):
            if 0 <= cls < 3:
                out[:, cls] = P[:, j]
        total = out.sum(axis=1, keepdims=True)
        total[total <= 0] = 1.0
        return out / total


@dataclass
class StackOutcome:
    probs: np.ndarray
    method: str
    weights: dict[str, float] = field(default_factory=dict)
    fallback: str | None = None
    scores: dict[str, float] = field(default_factory=dict)


class EnsembleStacker:
    """Walk-forward ensemble: fits on prior OOF rows, predicts the next block."""

    def __init__(self, names: list[str], C: float = 1.0, min_rows: int = 800,
                 seed: int = 42, methods: list[str] | None = None,
                 recency_half_life_blocks: int = RECENCY_HALF_LIFE_BLOCKS):
        self.names = list(names)
        self.C = float(C)
        self.min_rows = int(min_rows)
        self.seed = int(seed)
        self.methods = list(methods or ["logit", "logpool", "equal"])
        self.recency_half_life_blocks = int(recency_half_life_blocks)
        self.subset_models: dict[str, LogisticStacker] = {}
        self.subset_rows: dict[str, int] = {}
        self.weights: np.ndarray | None = None
        self.method: str = "equal"
        self.scores: dict[str, float] = {}

    # -- helpers -----------------------------------------------------------
    def _fit_subset_models(self, triples: np.ndarray, y: np.ndarray,
                           mask: np.ndarray, sample_weight: np.ndarray | None) -> None:
        self.subset_models = {}
        self.subset_rows = {}
        keys = [subset_key(row, self.names) for row in mask]
        frame = pd.DataFrame({"key": keys, "y": y})
        for key, grp in frame.groupby("key"):
            if len(grp) < LOGISTIC_MIN_ROWS:
                continue
            idx = grp.index.to_numpy()
            X = stack_features(triples[idx])
            w = None if sample_weight is None else sample_weight[idx]
            model = LogisticStacker(C=self.C, seed=self.seed)
            self.subset_models[key] = model.fit(X, y[idx], w)
            self.subset_rows[key] = int(len(idx))

    def _predict_logistic(self, triples: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if not self.subset_models:
            return None
        out = np.full((len(triples), 3), np.nan)
        for i, row in enumerate(mask):
            key = subset_key(row, self.names)
            model = self.subset_models.get(key)
            if model is None:
                return None
            X = stack_features(triples[i:i + 1])
            out[i] = model.predict(X)[0]
        return out

    def _scores(self, triples: np.ndarray, y: np.ndarray, mask: np.ndarray,
                sample_weight: np.ndarray | None) -> dict[str, float]:
        """Log loss of every candidate on the given rows."""
        from ..eval.metrics import log_loss

        out: dict[str, float] = {}
        equal = equal_pool(triples)
        out["equal"] = float(log_loss(equal, y))

        if self.weights is not None:
            pooled = log_pool(triples, self.weights)
            out["logpool"] = float(log_loss(pooled, y))

        logistic = self._predict_logistic(triples, mask)
        if logistic is not None:
            out["logit"] = float(log_loss(logistic, y))
        return out

    # -- public API --------------------------------------------------------
    def fit(self, triples: np.ndarray, y: np.ndarray, blocks: np.ndarray,
            current_block: int) -> None:
        """Fit on prior OOF rows. `blocks` holds each row's block index.

        I4: rows from `current_block` or later are dropped, so the stacker for
        a block can only ever have seen earlier blocks.
        """
        self.subset_models = {}
        self.subset_rows = {}
        self.weights = None
        self.method = "equal"
        self.scores = {}

        if triples is None or len(triples) == 0:
            return
        blocks = np.asarray(blocks, dtype=float)
        prior = blocks < float(current_block)
        if not prior.all():
            triples = np.asarray(triples, dtype=float)[prior]
            y = np.asarray(y)[prior]
            blocks = blocks[prior]
            if len(triples) == 0:
                return
        mask = available_mask(triples)
        age = np.maximum(current_block - blocks, 0.0)
        sample_weight = 0.5 ** (age / self.recency_half_life_blocks)

        if len(triples) < self.min_rows:
            self.scores = self._scores(triples, y, mask, sample_weight)
            return

        # nested split by time: choose the method on the recent part only
        order = np.argsort(np.asarray(blocks, dtype=float))
        cut = max(int(len(order) * (1.0 - EVAL_FRACTION)), 1)
        train_idx, eval_idx = order[:cut], order[cut:]
        if len(eval_idx) < 50 or len(train_idx) < 100:
            train_idx, eval_idx = order, order

        candidates = self.methods
        scores: dict[str, float] = {}

        if "equal" in candidates:
            scores["equal"] = float(self._loss(equal_pool(triples[eval_idx]), y[eval_idx]))
        if "logpool" in candidates:
            w_nested = fit_log_pool(triples[train_idx], y[train_idx], sample_weight[train_idx])
            scores["logpool"] = float(self._loss(log_pool(triples[eval_idx], w_nested),
                                                 y[eval_idx]))
        if "logit" in candidates:
            nested = EnsembleStacker(self.names, C=self.C, min_rows=self.min_rows,
                                     seed=self.seed, methods=self.methods,
                                     recency_half_life_blocks=self.recency_half_life_blocks)
            nested._fit_subset_models(triples[train_idx], y[train_idx], mask[train_idx],
                                      sample_weight[train_idx])
            logistic = nested._predict_logistic(triples[eval_idx], mask[eval_idx])
            if logistic is not None:
                scores["logit"] = float(self._loss(logistic, y[eval_idx]))

        self.scores = scores
        if scores:
            self.method = min(scores, key=lambda k: scores[k])
        else:
            self.method = "equal"

        # refit the winner on every prior row
        self._fit_subset_models(triples, y, mask, sample_weight)
        if self.method == "logpool":
            self.weights = fit_log_pool(triples, y, sample_weight)

    def predict(self, triples: np.ndarray) -> StackOutcome:
        mask = available_mask(triples)
        if self.method == "logit":
            P = self._predict_logistic(triples, mask)
            if P is not None:
                return StackOutcome(probs=P, method="logit", scores=dict(self.scores))
            return StackOutcome(probs=equal_pool(triples), method="equal",
                                fallback="logit_without_subset_model",
                                scores=dict(self.scores))
        if self.method == "logpool" and self.weights is not None:
            return StackOutcome(probs=log_pool(triples, self.weights), method="logpool",
                                weights={n: float(w) for n, w in zip(self.names, self.weights)},
                                scores=dict(self.scores))
        return StackOutcome(probs=equal_pool(triples), method="equal", scores=dict(self.scores))

    @staticmethod
    def _loss(P: np.ndarray, y: np.ndarray) -> float:
        from ..eval.metrics import log_loss

        return float(log_loss(P, y))
