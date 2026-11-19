"""Baseline models M0 (spec section 7.0). Every plot needs a floor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import assert_as_of

HOME_PRIOR = np.array([0.46, 0.26, 0.28])


class FreqModel:
    """League-specific historical H/D/A frequencies over a trailing window of seasons."""

    name = "freq"
    needs = {"results"}

    def __init__(self, window_seasons: int = 3):
        self.window_seasons = window_seasons
        self.freq: dict[str, np.ndarray] = {}
        self.global_freq = np.array([1 / 3, 1 / 3, 1 / 3])

    def fit(self, history: pd.DataFrame, as_of: str) -> None:
        assert_as_of(history, as_of)
        if len(history) == 0:
            self.freq = {}
            return
        counts = {"H": 0, "D": 0, "A": 0}
        for league, grp in history.groupby("league"):
            seasons = sorted(grp["season"].unique())[-self.window_seasons:]
            sub = grp[grp["season"].isin(seasons)]
            if len(sub) == 0:
                continue
            vc = sub["res"].value_counts()
            total = float(len(sub))
            self.freq[league] = np.array(
                [float(vc.get(k, 0)) / total for k in ("H", "D", "A")], dtype=float
            )
            for k in ("H", "D", "A"):
                counts[k] += int(vc.get(k, 0))
        total = sum(counts.values())
        if total:
            self.global_freq = np.array([counts[k] / total for k in ("H", "D", "A")], dtype=float)

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        n = len(fixtures)
        P = np.zeros((n, 3))
        status = np.empty(n, dtype=object)
        for i, league in enumerate(fixtures["league"].tolist()):
            p = self.freq.get(league, self.global_freq)
            P[i] = p
            status[i] = "ok"
        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
        out["status"] = status
        return out


class HomeModel:
    """Always-home prior, scaled by each league's own frequencies."""

    name = "home"
    needs = {"results"}

    def __init__(self):
        self.freq: dict[str, np.ndarray] = {}
        self.global_freq = np.array([1 / 3, 1 / 3, 1 / 3])

    def fit(self, history: pd.DataFrame, as_of: str) -> None:
        assert_as_of(history, as_of)
        self.freq = {}
        if len(history) == 0:
            return
        counts = {"H": 0, "D": 0, "A": 0}
        for league, grp in history.groupby("league"):
            vc = grp["res"].value_counts()
            total = float(len(grp))
            self.freq[league] = np.array(
                [float(vc.get(k, 0)) / total for k in ("H", "D", "A")], dtype=float
            )
            for k in ("H", "D", "A"):
                counts[k] += int(vc.get(k, 0))
        total = sum(counts.values())
        if total:
            self.global_freq = np.array([counts[k] / total for k in ("H", "D", "A")], dtype=float)

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        n = len(fixtures)
        P = np.zeros((n, 3))
        for i, league in enumerate(fixtures["league"].tolist()):
            freq = self.freq.get(league, self.global_freq)
            scaled = HOME_PRIOR * (freq / self.global_freq)
            P[i] = scaled / scaled.sum()
        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
        out["status"] = "ok"
        return out
