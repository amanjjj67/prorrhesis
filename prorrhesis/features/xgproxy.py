"""Shot-quality proxy for xG (spec section 7.4).

Without Understat we learn a crude xG from football-data's shots columns: for
each team-match, goals scored are regressed on shots on target and shots off
target with a Poisson regression. The regression is fitted separately for each
season-block, using only seasons before the block, so a match never sees a
model trained on its own season or later.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

TEAM_MATCH_COLUMNS = ("sot", "off")


def _clip(value: float) -> float:
    if value is None or not math.isfinite(value):
        return float("nan")
    return float(max(value, 0.0))


def team_match_rows(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with one row per team-match: [shots_on_target, shots_off] -> goals."""
    if matches is None or len(matches) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    hs = matches["hs"].to_numpy(dtype=float) if "hs" in matches else np.full(len(matches), np.nan)
    hst = matches["hst"].to_numpy(dtype=float) if "hst" in matches else np.full(len(matches), np.nan)
    as_ = matches["as_"].to_numpy(dtype=float) if "as_" in matches else np.full(len(matches), np.nan)
    ast = matches["ast"].to_numpy(dtype=float) if "ast" in matches else np.full(len(matches), np.nan)
    hg = matches["hg"].to_numpy(dtype=float)
    ag = matches["ag"].to_numpy(dtype=float)

    sot = np.concatenate([hst, ast])
    off = np.concatenate([hs - hst, as_ - ast])
    goals = np.concatenate([hg, ag])

    ok = np.isfinite(sot) & np.isfinite(off) & np.isfinite(goals)
    ok &= goals >= 0
    X = np.column_stack([np.clip(sot[ok], 0, None), np.clip(off[ok], 0, None)])
    return X, goals[ok]


class XgProxyModel:
    """Poisson regression of goals on shots on target and shots off target."""

    def __init__(self, alpha: float = 1e-3):
        self.alpha = float(alpha)
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=500)
        self.fitted = False
        self.n_rows = 0

    def fit(self, matches: pd.DataFrame) -> "XgProxyModel":
        X, y = team_match_rows(matches)
        if len(y) < 50:
            return self
        self.model.fit(X, y)
        self.fitted = True
        self.n_rows = int(len(y))
        return self

    def predict_goals(self, sot: float, off: float) -> float:
        if not self.fitted:
            return float("nan")
        sot = _clip(sot)
        off = _clip(off)
        if not (math.isfinite(sot) and math.isfinite(off)):
            return float("nan")
        value = float(self.model.predict(np.array([[sot, off]]))[0])
        return max(value, 0.0)

    def predict_match(self, row) -> tuple[float, float]:
        hs = _clip(row.get("hs"))
        hst = _clip(row.get("hst"))
        as_ = _clip(row.get("as_"))
        ast = _clip(row.get("ast"))
        xg_h = self.predict_goals(hst, hs - hst)
        xg_a = self.predict_goals(ast, as_ - ast)
        return xg_h, xg_a


def build_season_models(matches: pd.DataFrame, alpha: float = 1e-3) -> dict[str, XgProxyModel]:
    """One proxy model per season, fitted only on earlier seasons."""
    models: dict[str, XgProxyModel] = {}
    if matches is None or len(matches) == 0:
        return models

    seasons = sorted(str(s) for s in matches["season"].unique())
    for season in seasons:
        earlier = matches[matches["season"].astype(str) < season]
        models[season] = XgProxyModel(alpha=alpha).fit(earlier)
    return models
