"""xG-Poisson goal model (spec section 7.4).

Same structure as the Dixon-Coles model, but each team-match is fitted against
a blended target `y* = w * goals + (1 - w) * xg`, where `xg` is Understat xG
when it is enabled and the shot-quality proxy otherwise. The likelihood is
quasi-Poisson (`y* log(lambda) - lambda`): no tau correction and no rho.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..features.xgproxy import XgProxyModel, build_season_models
from .base import assert_as_of
from .dixoncoles import LeagueParams, _grid_from_lambdas

EPS = 1e-10


def _objective_grad(params: np.ndarray, yh: np.ndarray, ya: np.ndarray,
                    hi: np.ndarray, ai: np.ndarray, w: np.ndarray,
                    n_teams: int, gamma: float):
    """Quasi-Poisson negative log-likelihood and its analytic gradient."""
    mu0 = params[0]
    home = params[1]
    att = params[2:2 + n_teams]
    deff = params[2 + n_teams:2 + 2 * n_teams]

    lh = np.exp(mu0 + home + att[hi] - deff[ai])
    la = np.exp(mu0 + att[ai] - deff[hi])

    loglik = yh * np.log(lh) - lh + ya * np.log(la) - la
    obj = float(-np.sum(w * loglik) + gamma * (np.sum(att ** 2) + np.sum(deff ** 2)))

    A = yh - lh
    B = ya - la
    g = np.zeros_like(params)
    g[0] = -np.sum(w * (A + B))
    g[1] = -np.sum(w * A)
    np.add.at(g, 2 + hi, -w * A)
    np.add.at(g, 2 + ai, -w * B)
    np.add.at(g, 2 + n_teams + ai, w * A)     # defence of the away team
    np.add.at(g, 2 + n_teams + hi, w * B)     # defence of the home team
    g[2:2 + n_teams] += 2.0 * gamma * att
    g[2 + n_teams:] += 2.0 * gamma * deff
    return obj, g


def _fit_league(yh, ya, hi, ai, w, teams, gamma, start=None):
    n_teams = len(teams)
    if start is None:
        params = np.zeros(2 + 2 * n_teams)
        params[0] = 0.15
        params[1] = 0.25
    else:
        params = start.copy()

    res = minimize(
        _objective_grad, params, args=(yh, ya, hi, ai, w, n_teams, gamma),
        method="L-BFGS-B", jac=True,
        bounds=[(None, None), (None, None)] + [(None, None)] * (2 * n_teams),
        options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-8},
    )
    params = res.x
    att = params[2:2 + n_teams]
    deff = params[2 + n_teams:]
    m_a = float(att.mean()) if n_teams else 0.0
    m_d = float(deff.mean()) if n_teams else 0.0
    params[0] = params[0] + m_a - m_d
    params[2:2 + n_teams] = att - m_a
    params[2 + n_teams:] = deff - m_d
    return params, {"success": bool(res.success), "message": str(res.message),
                    "nll": float(res.fun)}


class XgPoissonModel:
    """M3: goal model fitted against blended goals/xG targets."""

    name = "xgpoisson"
    needs = {"results"}

    def __init__(
        self,
        w: float = 0.5,
        window_seasons: int = 3,
        half_life_days: float = 365.0,
        gamma: float = 0.5,
        max_goals: int = 10,
        min_matches: int = 5,
        name: str = "xgpoisson",
        xg_lookup=None,
    ):
        self.w = float(w)
        self.window_seasons = int(window_seasons)
        self.half_life_days = float(half_life_days)
        self.gamma = float(gamma)
        self.max_goals = int(max_goals)
        self.min_matches = int(min_matches)
        self.name = name
        self.xg_lookup = xg_lookup
        self.params: dict[str, LeagueParams] = {}
        self.info: dict[str, dict] = {}
        self.uncertain_teams: set[int] = set()
        self.uncertain_by_league: dict[str, set[int]] = {}
        self._prev: dict[str, LeagueParams] = {}

    # -- xG target ---------------------------------------------------------
    def _xg_for_row(self, row, proxy: XgProxyModel | None) -> tuple[float, float]:
        if self.xg_lookup is not None:
            found = self.xg_lookup(row)
            if found is not None:
                return float(found[0]), float(found[1])
        if proxy is None or not proxy.fitted:
            return float("nan"), float("nan")
        return proxy.predict_match(row)

    def _blended_targets(self, sub: pd.DataFrame,
                         proxies: dict[str, XgProxyModel]) -> tuple[np.ndarray, np.ndarray]:
        goals_h = sub["hg"].to_numpy(dtype=float)
        goals_a = sub["ag"].to_numpy(dtype=float)
        xg_h = np.full(len(sub), np.nan)
        xg_a = np.full(len(sub), np.nan)
        for i, (_, row) in enumerate(sub.iterrows()):
            proxy = proxies.get(str(row["season"]))
            xh, xa = self._xg_for_row(row, proxy)
            xg_h[i] = xh
            xg_a[i] = xa
        if self.w >= 1.0:
            return goals_h, goals_a
        if self.w <= 0.0:
            return xg_h, xg_a
        yh = self.w * goals_h + (1.0 - self.w) * xg_h
        ya = self.w * goals_a + (1.0 - self.w) * xg_a
        return yh, ya

    # -- fitting -----------------------------------------------------------
    def fit(self, history: pd.DataFrame, as_of: str) -> None:
        assert_as_of(history, as_of)
        self._prev = self.params
        self.params = {}
        self.info = {}
        self.uncertain_teams = set()
        self.uncertain_by_league = {}
        if history is None or len(history) == 0:
            return

        proxies = build_season_models(history)
        xi = math.log(2.0) / max(self.half_life_days, 1.0)
        as_of_ts = pd.Timestamp(as_of)

        for league, grp in history.groupby("league"):
            grp = grp.sort_values(["date", "home_id", "away_id"])
            seasons = sorted(grp["season"].unique())[-self.window_seasons:]
            sub = grp[grp["season"].isin(seasons)]
            if len(sub) < 20:
                continue

            yh, ya = self._blended_targets(sub, proxies)
            ok = np.isfinite(yh) & np.isfinite(ya)
            if ok.sum() < 20:
                continue
            sub = sub[ok]
            yh, ya = yh[ok], ya[ok]

            counts = pd.concat([sub["home_id"], sub["away_id"]]).value_counts()
            teams = sorted(int(t) for t in counts.index)
            idx = {t: i for i, t in enumerate(teams)}
            hi = np.array([idx[int(t)] for t in sub["home_id"]], dtype=int)
            ai = np.array([idx[int(t)] for t in sub["away_id"]], dtype=int)
            age_days = (as_of_ts - pd.to_datetime(sub["date"])).dt.days.to_numpy(dtype=float)
            w = np.exp(-xi * np.clip(age_days, 0, None))

            params, info = _fit_league(yh, ya, hi, ai, w, teams, self.gamma,
                                        self._warm_start(league, teams))
            self.params[league] = LeagueParams(
                teams=teams, att=params[2:2 + len(teams)],
                deff=params[2 + len(teams):], mu0=float(params[0]),
                home=float(params[1]), rho=0.0,
                counts={int(k): int(v) for k, v in counts.items()},
            )
            self.info[league] = info

            uncertain = {t for t, c in self.params[league].counts.items()
                         if c < self.min_matches}
            if uncertain:
                lp = self.params[league]
                for i, t in enumerate(lp.teams):
                    if t in uncertain:
                        lp.att[i] = 0.0
                        lp.deff[i] = 0.0
            self.uncertain_by_league[league] = uncertain
            self.uncertain_teams |= uncertain

    def _warm_start(self, league: str, teams: list[int]) -> np.ndarray | None:
        prev = self._prev.get(league)
        if prev is None:
            return None
        n = len(teams)
        params = np.zeros(2 + 2 * n)
        params[0] = prev.mu0
        params[1] = prev.home
        for i, t in enumerate(teams):
            j = prev.index_of(t)
            if j is not None:
                params[2 + i] = prev.att[j]
                params[2 + n + i] = prev.deff[j]
        return params

    # -- prediction --------------------------------------------------------
    def lambdas(self, league: str, home_id: int, away_id: int) -> tuple[float, float]:
        p = self.params.get(league)
        if p is None:
            return 1.3, 1.1
        hi = p.index_of(home_id)
        ai = p.index_of(away_id)
        att_h = float(p.att[hi]) if hi is not None else 0.0
        def_a = float(p.deff[ai]) if ai is not None else 0.0
        att_a = float(p.att[ai]) if ai is not None else 0.0
        def_h = float(p.deff[hi]) if hi is not None else 0.0
        lh = float(np.exp(p.mu0 + p.home + att_h - def_a))
        la = float(np.exp(p.mu0 + att_a - def_h))
        return lh, la

    def grid_for(self, league: str, home_id: int, away_id: int) -> np.ndarray:
        lh, la = self.lambdas(league, home_id, away_id)
        return _grid_from_lambdas(lh, la, 0.0, self.max_goals)

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        n = len(fixtures)
        P = np.zeros((n, 3))
        grids: list[np.ndarray] = []
        status = np.empty(n, dtype=object)
        d = self.max_goals
        ii, jj = np.meshgrid(np.arange(d + 1), np.arange(d + 1), indexing="ij")
        for i, (_, row) in enumerate(fixtures.iterrows()):
            league = str(row["league"])
            if league not in self.params:
                P[i] = (1 / 3, 1 / 3, 1 / 3)
                grids.append(None)
                status[i] = "abstain"
                continue
            g = self.grid_for(league, int(row["home_id"]), int(row["away_id"]))
            grids.append(g)
            P[i] = (float(g[ii > jj].sum()), float(np.trace(g)), float(g[ii < jj].sum()))
            status[i] = "ok"
        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
        out["status"] = status
        out["grid"] = grids
        return out
