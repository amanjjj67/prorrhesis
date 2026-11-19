"""Dixon-Coles goal model with time decay (spec section 7.2, model M2).

Fitted per league on a trailing window of seasons, with a ridge penalty, an
analytic gradient, and warm starts between blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .base import assert_as_of

EPS = 1e-10


@dataclass
class LeagueParams:
    teams: list[int]
    att: np.ndarray
    deff: np.ndarray
    mu0: float
    home: float
    rho: float
    counts: dict[int, int] = field(default_factory=dict)

    def index_of(self, team_id: int) -> int | None:
        try:
            return self.teams.index(int(team_id))
        except ValueError:
            return None


def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _tau_terms(x: np.ndarray, y: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float):
    """log tau plus its partial derivatives wrt rho, lh and la."""
    logtau = np.zeros(len(x))
    d_rho = np.zeros(len(x))
    d_lh = np.zeros(len(x))
    d_la = np.zeros(len(x))

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        if m00.any():
            t = 1.0 - lh[m00] * la[m00] * rho
            ok = t > EPS
            ts = np.where(ok, t, 1.0)
            logtau[m00] = np.where(ok, np.log(ts), -50.0)
            d_rho[m00] = np.where(ok, -lh[m00] * la[m00] / ts, 0.0)
            d_lh[m00] = np.where(ok, -la[m00] * rho / ts, 0.0)
            d_la[m00] = np.where(ok, -lh[m00] * rho / ts, 0.0)
        if m01.any():
            t = 1.0 + lh[m01] * rho
            ok = t > EPS
            ts = np.where(ok, t, 1.0)
            logtau[m01] = np.where(ok, np.log(ts), -50.0)
            d_rho[m01] = np.where(ok, lh[m01] / ts, 0.0)
            d_lh[m01] = np.where(ok, rho / ts, 0.0)
        if m10.any():
            t = 1.0 + la[m10] * rho
            ok = t > EPS
            ts = np.where(ok, t, 1.0)
            logtau[m10] = np.where(ok, np.log(ts), -50.0)
            d_rho[m10] = np.where(ok, la[m10] / ts, 0.0)
            d_la[m10] = np.where(ok, rho / ts, 0.0)
        if m11.any():
            t = 1.0 - rho
            ok = t > EPS
            ts = np.where(ok, t, 1.0)
            logtau[m11] = np.where(ok, np.log(ts), -50.0)
            d_rho[m11] = np.where(ok, -1.0 / ts, 0.0)
    return logtau, d_rho, d_lh, d_la


def _objective_grad(params: np.ndarray, x, y, hi, ai, w, n_teams, gamma):
    mu0 = params[0]
    h = params[1]
    rho = params[2]
    att = params[3:3 + n_teams]
    deff = params[3 + n_teams:3 + 2 * n_teams]

    lh = np.exp(mu0 + h + att[hi] - deff[ai])
    la = np.exp(mu0 + att[ai] - deff[hi])

    logp = _poisson_logpmf(x, lh) + _poisson_logpmf(y, la)
    logtau, d_rho, d_lh, d_la = _tau_terms(x, y, lh, la, rho)
    ll = logp + logtau

    obj = float(-np.sum(w * ll) + gamma * (np.sum(att ** 2) + np.sum(deff ** 2)))

    A = x - lh
    B = y - la
    tl = d_lh * lh      # d loglik / d log(lambda_h) contribution from tau
    ta = d_la * la      # d loglik / d log(lambda_a) contribution from tau

    g = np.zeros_like(params)
    g[0] = -np.sum(w * (A + B + tl + ta))
    g[1] = -np.sum(w * (A + tl))
    g[2] = -np.sum(w * d_rho)
    np.add.at(g, 3 + hi, -w * (A + tl))
    np.add.at(g, 3 + ai, -w * (B + ta))
    np.add.at(g, 3 + n_teams + ai, w * (A + tl))   # def of the away team
    np.add.at(g, 3 + n_teams + hi, w * (B + ta))   # def of the home team
    g[3:3 + n_teams] += 2.0 * gamma * att
    g[3 + n_teams:] += 2.0 * gamma * deff
    return obj, g


def _fit_league(
    x: np.ndarray, y: np.ndarray, hi: np.ndarray, ai: np.ndarray, w: np.ndarray,
    teams: list[int], gamma: float, rho_bounds: tuple[float, float],
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    n_teams = len(teams)
    if start is None:
        params = np.zeros(3 + 2 * n_teams)
        params[0] = 0.15   # mu0 ~ exp(0.15) = 1.16 goals
        params[1] = 0.25   # home advantage
        params[2] = -0.03  # rho
    else:
        params = start.copy()

    res = minimize(
        _objective_grad, params, args=(x, y, hi, ai, w, n_teams, gamma),
        method="L-BFGS-B", jac=True,
        bounds=[(None, None), (None, None), rho_bounds] + [(None, None)] * (2 * n_teams),
        options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-8},
    )
    params = res.x
    att = params[3:3 + n_teams]
    deff = params[3 + n_teams:]
    m_a = float(att.mean()) if n_teams else 0.0
    m_d = float(deff.mean()) if n_teams else 0.0
    params[0] = params[0] + m_a - m_d
    att = att - m_a
    deff = deff - m_d
    params[3:3 + n_teams] = att
    params[3 + n_teams:] = deff
    return params, {"success": bool(res.success), "message": str(res.message),
                    "nll": float(res.fun)}


def _grid_from_lambdas(lh: float, la: float, rho: float, max_goals: int) -> np.ndarray:
    k = np.arange(max_goals + 1)
    lx = _poisson_logpmf(k, lh)
    ly = _poisson_logpmf(k, la)
    X, Y = np.meshgrid(k, k, indexing="ij")
    logtau, _, _, _ = _tau_terms(X.ravel(), Y.ravel(), np.full(X.size, lh), np.full(X.size, la), rho)
    P = np.exp(lx[:, None] + ly[None, :] + logtau.reshape(max_goals + 1, max_goals + 1))
    return P / P.sum()


class DixonColesModel:
    """M2: per-league Dixon-Coles goal model on a trailing window."""

    name = "dc"
    needs = {"results"}

    def __init__(
        self,
        window_seasons: int = 3,
        half_life_days: float = 365.0,
        gamma: float = 0.5,
        max_goals: int = 10,
        min_matches: int = 5,
        name: str = "dc",
    ):
        self.window_seasons = int(window_seasons)
        self.half_life_days = float(half_life_days)
        self.gamma = float(gamma)
        self.max_goals = int(max_goals)
        self.min_matches = int(min_matches)
        self.name = name
        self.params: dict[str, LeagueParams] = {}
        self.info: dict[str, dict] = {}
        self.uncertain_teams: set[int] = set()
        self.uncertain_by_league: dict[str, set[int]] = {}
        self._prev: dict[str, LeagueParams] = {}

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

        xi = math.log(2.0) / max(self.half_life_days, 1.0)
        as_of_ts = pd.Timestamp(as_of)

        for league, grp in history.groupby("league"):
            grp = grp.sort_values(["date", "home_id", "away_id"])
            seasons = sorted(grp["season"].unique())[-self.window_seasons:]
            sub = grp[grp["season"].isin(seasons)]
            if len(sub) < 20:
                continue
            counts = pd.concat([sub["home_id"], sub["away_id"]]).value_counts()
            teams = sorted(int(t) for t in counts.index)
            idx = {t: i for i, t in enumerate(teams)}
            hi = np.array([idx[int(t)] for t in sub["home_id"]], dtype=int)
            ai = np.array([idx[int(t)] for t in sub["away_id"]], dtype=int)
            x = sub["hg"].to_numpy(dtype=float)
            y = sub["ag"].to_numpy(dtype=float)
            age_days = (as_of_ts - pd.to_datetime(sub["date"])).dt.days.to_numpy(dtype=float)
            w = np.exp(-xi * np.clip(age_days, 0, None))

            start = self._warm_start(league, teams)
            params, info = _fit_league(x, y, hi, ai, w, teams, self.gamma,
                                       (-0.2, 0.2), start)
            self.params[league] = LeagueParams(
                teams=teams, att=params[3:3 + len(teams)],
                deff=params[3 + len(teams):], mu0=float(params[0]),
                home=float(params[1]), rho=float(params[2]),
                counts={int(k): int(v) for k, v in counts.items()},
            )
            self.info[league] = info
            uncertain = set()
            for t, c in self.params[league].counts.items():
                if c < self.min_matches:
                    uncertain.add(t)
            if uncertain:
                lp = self.params[league]
                for idx, t in enumerate(lp.teams):
                    if t in uncertain:
                        lp.att[idx] = 0.0
                        lp.deff[idx] = 0.0
            self.uncertain_by_league[league] = uncertain
            self.uncertain_teams |= uncertain

    def _warm_start(self, league: str, teams: list[int]) -> np.ndarray | None:
        prev = self._prev.get(league)
        if prev is None:
            return None
        n = len(teams)
        params = np.zeros(3 + 2 * n)
        params[0] = prev.mu0
        params[1] = prev.home
        params[2] = prev.rho
        for i, t in enumerate(teams):
            j = prev.index_of(t)
            if j is not None:
                params[3 + i] = prev.att[j]
                params[3 + n + i] = prev.deff[j]
        return params

    # -- prediction --------------------------------------------------------
    def lambdas(self, league: str, home_id: int, away_id: int) -> tuple[float, float, float]:
        p = self.params.get(league)
        if p is None:
            return 1.3, 1.1, 0.0
        hi = p.index_of(home_id)
        ai = p.index_of(away_id)
        att_h = float(p.att[hi]) if hi is not None else 0.0
        def_a = float(p.deff[ai]) if ai is not None else 0.0
        att_a = float(p.att[ai]) if ai is not None else 0.0
        def_h = float(p.deff[hi]) if hi is not None else 0.0
        lh = float(np.exp(p.mu0 + p.home + att_h - def_a))
        la = float(np.exp(p.mu0 + att_a - def_h))
        rho = p.rho
        lo = max(-1.0 / lh, -1.0 / la)
        hi_b = min(1.0, 1.0 / (lh * la))
        if not (lo < rho < hi_b):
            rho = float(np.clip(rho, lo + 1e-6, hi_b - 1e-6))
        return lh, la, rho

    def grid_for(self, league: str, home_id: int, away_id: int) -> np.ndarray:
        lh, la, rho = self.lambdas(league, home_id, away_id)
        return _grid_from_lambdas(lh, la, rho, self.max_goals)

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
            pH = float(g[ii > jj].sum())
            pD = float(np.trace(g))
            pA = float(g[ii < jj].sum())
            P[i] = (pH, pD, pA)
            status[i] = "ok"
        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
        out["status"] = status
        out["grid"] = grids
        return out
