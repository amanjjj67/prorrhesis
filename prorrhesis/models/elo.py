"""Elo ratings with an ordinal outcome link (spec section 7.1, model M1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .base import assert_as_of

DEFAULT_HOME_ADV = {"E0": 65, "SP1": 70, "D1": 60, "I1": 65, "F1": 65}
INIT_RATING = 1500.0
HASH_COLS = ["date", "league", "home_id", "away_id", "hg", "ag"]


def elo_update(Rh: float, Ra: float, hg: int, ag: int, H: float, K: float) -> tuple[float, float]:
    d = Rh + H - Ra
    We = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
    W = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    gd = abs(hg - ag)
    G = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
    delta = K * G * (W - We)
    return Rh + delta, Ra - delta


def softplus(x: float) -> float:
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


def _frame_hash(df: pd.DataFrame, cols: list[str]) -> int:
    if len(df) == 0:
        return 0
    h = pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64)
    return int(np.sum(h, dtype=np.uint64))


class RatingState:
    """Elo ratings updated in strict date order; same-date matches use pre-date ratings."""

    def __init__(self, K: float, home_adv: dict[str, float], rho: float, promoted_penalty: float):
        self.K = float(K)
        self.home_adv = dict(home_adv)
        self.rho = float(rho)
        self.promoted_penalty = float(promoted_penalty)
        self.ratings: dict[int, float] = {}
        self.league_teams: dict[str, set] = {}
        self.league_season: dict[str, str] = {}

    def league_mean(self, league: str) -> float:
        teams = self.league_teams.get(league, set())
        if not teams:
            return INIT_RATING
        return float(np.mean([self.ratings[t] for t in teams if t in self.ratings]))

    def prepare(self, matches: pd.DataFrame) -> None:
        """Season carry-over and first-time team ratings, without touching results."""
        if matches is None or matches.empty:
            return
        for league, lg in matches.groupby("league", sort=True):
            season = str(lg["season"].iloc[0])
            prev = self.league_season.get(league)
            if prev is not None and season != prev:
                teams = self.league_teams.get(league, set())
                if teams:
                    mean = float(np.mean([self.ratings[t] for t in teams]))
                    for t in teams:
                        self.ratings[t] = mean + (1.0 - self.rho) * (self.ratings[t] - mean)
            self.league_season[league] = season

        for _, m in matches.iterrows():
            league = str(m["league"])
            mean = self.league_mean(league)
            for tid in (int(m["home_id"]), int(m["away_id"])):
                if tid not in self.ratings:
                    self.ratings[tid] = mean - self.promoted_penalty
                    self.league_teams.setdefault(league, set()).add(tid)

    def apply(self, matches: pd.DataFrame) -> np.ndarray:
        """Apply results and return the pre-date rating differences."""
        if matches is None or matches.empty:
            return np.array([])
        pre = [
            (self.ratings[int(m["home_id"])], self.ratings[int(m["away_id"])])
            for _, m in matches.iterrows()
        ]
        diffs = []
        for (rh, ra), (_, m) in zip(pre, matches.iterrows()):
            H = float(self.home_adv.get(str(m["league"]), 65.0))
            nh, na = elo_update(rh, ra, int(m["hg"]), int(m["ag"]), H, self.K)
            self.ratings[int(m["home_id"])] = nh
            self.ratings[int(m["away_id"])] = na
        for (rh, ra), (_, m) in zip(pre, matches.iterrows()):
            diffs.append(rh + float(self.home_adv.get(str(m["league"]), 65.0)) - ra)
        return np.array(diffs)

    def process(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update ratings for `matches` (already sorted by date). Returns (diffs, home_ids, away_ids)."""
        diffs: list[float] = []
        home_ids: list[int] = []
        away_ids: list[int] = []
        if matches.empty:
            return np.array([]), np.array([], dtype=int), np.array([], dtype=int)

        for date, grp in matches.groupby("date", sort=True):
            self.prepare(grp)
            date_diffs = self.apply(grp)
            diffs.extend(float(d) for d in date_diffs)
            for _, m in grp.iterrows():
                home_ids.append(int(m["home_id"]))
                away_ids.append(int(m["away_id"]))

        return np.array(diffs), np.array(home_ids, dtype=int), np.array(away_ids, dtype=int)


def link_probs(d: np.ndarray, s: float, c: float) -> np.ndarray:
    """P(H)=sigmoid(z-c), P(A)=sigmoid(-z-c), P(D)=1-P(H)-P(A) with z=d/s."""
    z = np.asarray(d, dtype=float) / float(s)
    pH = expit(z - c)
    pA = expit(-z - c)
    pD = np.clip(1.0 - pH - pA, 0.0, None)
    return np.stack([pH, pD, pA], axis=1)


def fit_link(d: np.ndarray, y: np.ndarray, s0: float = 173.7, c0: float = 0.5) -> tuple[float, float]:
    """Maximum likelihood fit of (scale s, draw threshold c >= 0) by scipy minimize."""
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float(s0), float(c0)

    def nll(params: np.ndarray) -> float:
        s = float(np.exp(params[0]))
        c = softplus(float(params[1]))
        P = link_probs(d, s, c)
        P = np.clip(P, 1e-12, 1.0)
        return float(-np.mean(np.log(P[np.arange(len(y)), y])))

    c_raw0 = float(np.log(np.expm1(c0))) if c0 > 0 else -5.0
    res = minimize(nll, x0=np.array([np.log(s0), c_raw0]), method="L-BFGS-B",
                   bounds=[(np.log(40.0), np.log(800.0)), (-8.0, 8.0)])
    return float(np.exp(res.x[0])), float(softplus(float(res.x[1])))


class EloModel:
    """M1: Elo ratings linked to Home/Draw/Away by an ordered logistic."""

    name = "elo"
    needs = {"results"}

    def __init__(
        self,
        K: float = 20.0,
        home_adv: dict[str, float] | None = None,
        rho_season: float = 0.25,
        promoted_penalty: float = 50.0,
        name: str = "elo",
    ):
        self.K = float(K)
        self.home_adv = dict(home_adv or DEFAULT_HOME_ADV)
        self.rho = float(rho_season)
        self.promoted_penalty = float(promoted_penalty)
        self.name = name
        self.state: RatingState | None = None
        self._last_date: str | None = None
        self._n_processed = 0
        self._prefix_hash = 0
        self._diffs = np.array([])
        self.s: float = 173.7
        self.c: float = 0.5

    # -- fitting -----------------------------------------------------------
    def fit(self, history: pd.DataFrame, as_of: str) -> None:
        assert_as_of(history, as_of)
        if history is None or len(history) == 0:
            self.state = None
            self._diffs = np.array([])
            return

        h = history.sort_values(["date", "league", "home_id", "away_id"]).reset_index(drop=True)
        if self.state is None or self._last_date is None:
            self.state = RatingState(self.K, self.home_adv, self.rho, self.promoted_penalty)
            self._diffs = np.array([])
            self._n_processed = 0
            self._prefix_hash = 0
            new = h
        else:
            prefix = h[h["date"] <= self._last_date]
            if len(prefix) == self._n_processed and _frame_hash(prefix, HASH_COLS) == self._prefix_hash:
                new = h[h["date"] > self._last_date]
            else:
                self.state = RatingState(self.K, self.home_adv, self.rho, self.promoted_penalty)
                self._diffs = np.array([])
                self._n_processed = 0
                self._prefix_hash = 0
                new = h

        diffs, _, _ = self.state.process(new)
        if len(diffs):
            self._diffs = np.concatenate([self._diffs, diffs])
        self._n_processed = len(h)
        self._prefix_hash = _frame_hash(h, HASH_COLS)
        self._last_date = str(h["date"].max())

        y = self._results_for(h)
        self.s, self.c = fit_link(self._diffs, y)

    def _results_for(self, h: pd.DataFrame) -> np.ndarray:
        res = h["res"].to_numpy()
        return np.array([{"H": 0, "D": 1, "A": 2}.get(str(r), 1) for r in res], dtype=int)

    # -- prediction --------------------------------------------------------
    def rating_for(self, team_id: int, league: str) -> float:
        if self.state is not None and team_id in self.state.ratings:
            return float(self.state.ratings[team_id])
        if self.state is not None:
            return self.state.league_mean(league)
        return INIT_RATING

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        n = len(fixtures)
        if n == 0:
            return pd.DataFrame({"pH": [], "pD": [], "pA": [], "status": []}, index=fixtures.index)
        d = np.zeros(n)
        for i, (_, row) in enumerate(fixtures.iterrows()):
            league = str(row["league"])
            rh = self.rating_for(int(row["home_id"]), league)
            ra = self.rating_for(int(row["away_id"]), league)
            d[i] = rh + float(self.home_adv.get(league, 65.0)) - ra
        P = link_probs(d, self.s, self.c)
        P = P / P.sum(axis=1, keepdims=True)
        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
        out["status"] = "ok"
        return out


def fit_home_advantage(
    history: pd.DataFrame,
    candidates: list[float],
    block_days: int = 28,
    K: float = 20.0,
    rho: float = 0.25,
    promoted_penalty: float = 50.0,
) -> dict[str, float]:
    """Fit the per-league home advantage by walk-forward log loss on the given history."""
    out: dict[str, float] = {}
    for league, grp in history.groupby("league"):
        grp = grp.sort_values(["date", "home_id", "away_id"]).reset_index(drop=True)
        best_H, best_ll = 65.0, float("inf")
        for H in candidates:
            state = RatingState(K, {league: H}, rho, promoted_penalty)
            diffs, _, _ = state.process(grp)
            y = np.array([{"H": 0, "D": 1, "A": 2}.get(str(r), 1) for r in grp["res"]], dtype=int)
            dates = grp["date"].to_numpy()
            ll = _walk_forward_link_loss(diffs, y, dates, block_days)
            if ll < best_ll:
                best_ll, best_H = ll, float(H)
        out[str(league)] = best_H
    return out


def tuning_path(cfg) -> Path:
    return cfg.data_dir / "tuning" / "elo.json"


def load_tuning(cfg) -> dict:
    p = tuning_path(cfg)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_tuning(cfg, data: dict) -> Path:
    p = tuning_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return p


def tune_elo(
    cfg,
    burn_in: pd.DataFrame,
    k_grid: list[float] | None = None,
    rho_grid: list[float] | None = None,
    h_grid: list[float] | None = None,
    block_days: int = 56,
) -> dict:
    """Tune K, rho and the per-league home advantage on the burn-in window only."""
    k_grid = k_grid or [float(k) for k in range(10, 41, 5)]
    rho_grid = rho_grid or [0.1, 0.2, 0.3, 0.4, 0.5]
    h_grid = h_grid or [float(h) for h in range(30, 111, 10)]
    penalty = float(cfg.elo.get("promoted_penalty", 50))

    by_league = {str(lg): grp.sort_values(["date", "home_id", "away_id"]).reset_index(drop=True)
                 for lg, grp in burn_in.groupby("league")}

    def losses(K: float, rho: float, H: float, league: str) -> float:
        grp = by_league[league]
        state = RatingState(K, {league: H}, rho, penalty)
        diffs, _, _ = state.process(grp)
        y = np.array([{"H": 0, "D": 1, "A": 2}.get(str(r), 1) for r in grp["res"]], dtype=int)
        return _walk_forward_link_loss(diffs, y, grp["date"].to_numpy(), block_days)

    best = {"K": float(cfg.elo.get("K", 20)), "rho": float(cfg.elo.get("rho_season", 0.25)),
            "loss": float("inf")}
    for K in k_grid:
        for rho in rho_grid:
            vals = [losses(K, rho, 65.0, lg) for lg in by_league]
            ll = float(np.mean(vals))
            if ll < best["loss"]:
                best = {"K": float(K), "rho": float(rho), "loss": ll}

    home_adv: dict[str, float] = {}
    for league in by_league:
        best_H, best_ll = 65.0, float("inf")
        for H in h_grid:
            ll = losses(best["K"], best["rho"], H, league)
            if ll < best_ll:
                best_ll, best_H = ll, float(H)
        home_adv[league] = best_H

    return {"K": best["K"], "rho": best["rho"], "H": home_adv, "burn_in_loss": best["loss"]}


def _walk_forward_link_loss(diffs: np.ndarray, y: np.ndarray, dates: np.ndarray, block_days: int) -> float:
    """Fit the link on the past and score the next block; returns mean log loss."""
    blocks = pd.to_datetime(pd.Series(dates)).dt.floor("D")
    starts = sorted(set(blocks))
    if not starts:
        return float("inf")
    first = starts[0]
    group_key = (pd.to_datetime(pd.Series(dates)) - first).dt.days // block_days
    losses = []
    for g in sorted(set(group_key)):
        idx = np.where(group_key == g)[0]
        train = np.where(group_key < g)[0]
        if len(train) < 200 or len(idx) == 0:
            continue
        s, c = fit_link(diffs[train], y[train])
        P = link_probs(diffs[idx], s, c)
        P = np.clip(P, 1e-12, 1.0)
        losses.append(-np.mean(np.log(P[np.arange(len(idx)), y[idx]])))
    return float(np.mean(losses)) if losses else float("inf")
