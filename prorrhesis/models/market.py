"""Market-implied probabilities (spec section 7.3).

Two strictly separate uses:
  * `market_close` - benchmark only, never a stacker input (invariant I5).
  * `market_open`  - pre-match odds, may feed the stacker and the GBM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .base import Probs, assert_as_of


def proportional(odds: np.ndarray) -> np.ndarray:
    arr = np.asarray(odds, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("invalid odds")
    pi = 1.0 / arr
    booksum = pi.sum()
    if not np.isfinite(booksum) or booksum <= 0:
        raise ValueError("invalid odds")
    return pi / booksum


def shin(odds: np.ndarray) -> np.ndarray:
    """Shin's method, exactly as written in spec section 7.3."""
    arr = np.asarray(odds, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("invalid odds")
    pi = 1.0 / arr
    Pi = pi.sum()
    if not np.isfinite(Pi) or Pi <= 0:
        raise ValueError("invalid odds")

    def p_of(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4 * (1 - z) * pi**2 / Pi) - z) / (2 * (1 - z))

    def f(z: float) -> float:
        return float(p_of(z).sum() - 1)

    try:
        z = brentq(f, 1e-9, 0.4)
    except (ValueError, RuntimeError):
        return proportional(odds)
    p = p_of(z)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return proportional(odds)
    return p / total


def de_vig(odds: np.ndarray, method: str = "shin") -> np.ndarray:
    if method == "proportional":
        return proportional(odds)
    return shin(odds)


class _MarketBase:
    needs: set[str] = {"odds"}

    def _triples(self, fixtures: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        cols = [f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]
        n = len(fixtures)
        P = np.full((n, 3), np.nan)
        ok = np.zeros(n, dtype=bool)
        for i in range(n):
            vals = [fixtures[cols[k]].iloc[i] if cols[k] in fixtures.columns else np.nan for k in range(3)]
            if any(v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) for v in vals):
                continue
            arr = np.array(vals, dtype=float)
            if not np.all(np.isfinite(arr)) or np.any(arr <= 1.0):
                continue
            try:
                p = de_vig(arr, self.method)
            except ValueError:
                continue
            P[i] = p
            ok[i] = True
        return P, ok


class MarketCloseModel(_MarketBase):
    """Benchmark built from closing odds. Never used as a stacker feature (I5)."""

    name = "market_close"
    needs = {"odds_close"}

    def __init__(self, method: str = "shin"):
        self.method = method

    def fit(self, history: pd.DataFrame, as_of: str) -> None:
        assert_as_of(history, as_of)
        return None

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        P, ok = self._triples(fixtures, "odds_close")
        return self._frame(P, ok, fixtures.index)

    def _frame(self, P, ok, index) -> pd.DataFrame:
        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=index)
        out["status"] = np.where(ok, "ok", "abstain")
        return out


class MarketOpenModel(_MarketBase):
    """Feature model built from pre-match open odds."""

    name = "market_open"
    needs = {"odds_open"}

    def __init__(self, method: str = "shin"):
        self.method = method

    def fit(self, history: pd.DataFrame, as_of: str) -> None:
        assert_as_of(history, as_of)
        return None

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        P, ok = self._triples(fixtures, "odds_open")
        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
        out["status"] = np.where(ok, "ok", "abstain")
        return out


def probs_from_odds(h: float, d: float, a: float, method: str = "shin") -> Probs:
    return Probs.make(*de_vig(np.array([h, d, a], dtype=float), method))
