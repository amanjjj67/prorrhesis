from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")
EPS = 1e-6


@dataclass(frozen=True)
class Probs:
    """Home/Draw/Away. Always normalized and clipped. Build only through Probs.make()."""

    h: float
    d: float
    a: float

    @staticmethod
    def make(h: float, d: float, a: float, eps: float = EPS) -> "Probs":
        v = np.clip(np.array([h, d, a], dtype=float), eps, None)
        v = v / v.sum()
        return Probs(*map(float, v))

    def as_array(self) -> np.ndarray:
        return np.array([self.h, self.d, self.a])


class Model(Protocol):
    name: str
    needs: set[str]

    def fit(self, history: pd.DataFrame, as_of: str) -> None: ...

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame: ...


def assert_as_of(history: pd.DataFrame | None, as_of: str) -> None:
    """Invariant I3: a model may only ever see matches strictly before `as_of`."""
    if history is None or len(history) == 0:
        return
    max_date = str(history["date"].max())
    if max_date >= as_of:
        raise ValueError(f"fit saw data on/after as_of ({max_date} >= {as_of})")


def sanitize_probs(P: np.ndarray) -> np.ndarray:
    """Validate a (n,3) probability matrix; raise on anything non-finite or unnormalised."""
    arr = np.asarray(P, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"probability matrix must be (n,3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite probability produced")
    if np.any(arr < 0):
        raise ValueError("negative probability produced")
    sums = arr.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("probability rows do not sum to 1")
    return arr


def probs_to_frame(P: np.ndarray, index: pd.Index) -> pd.DataFrame:
    arr = sanitize_probs(P)
    return pd.DataFrame({"pH": arr[:, 0], "pD": arr[:, 1], "pA": arr[:, 2]}, index=index)


def frame_to_probs(frame: pd.DataFrame) -> np.ndarray:
    return frame[["pH", "pD", "pA"]].to_numpy(dtype=float)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def softplus(x: float) -> float:
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))
