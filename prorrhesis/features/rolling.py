"""Chronological EWMA state machine (spec section 8.1).

All team state is built in one pass over matches sorted by date. For each
calendar date the features are read from the current state, which contains
only matches strictly before that date; only then are that date's results
applied. Same-day leakage is therefore impossible by construction.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

FORM_STATS = ("gf", "ga", "shots", "sot", "xgf", "xga", "pts")
VENUE_STATS = ("gf", "ga", "xgf", "xga")
VIEWS = ("overall", "home", "away")
RECENT_WINDOW_DAYS = 60


@dataclass
class Ewma:
    """EWMA with the spec's pseudo-count bias correction.

    alpha = 1 - 0.5 ** (1 / H_matches); n += 1; a_t = max(alpha, 1/(n + prior_n));
    s <- s + a_t * (x - s). The series starts at the league-season mean.
    """

    alpha: float
    prior_n: float
    s: float = float("nan")
    n: int = 0
    started: bool = False

    def start(self, value: float) -> None:
        if math.isfinite(value):
            self.s = float(value)
            self.started = True

    def update(self, x: float) -> None:
        if x is None or not math.isfinite(x):
            return
        self.n += 1
        a = max(self.alpha, 1.0 / (self.n + self.prior_n))
        if not self.started:
            self.s = float(x)
            self.started = True
            return
        self.s += a * (x - self.s)


@dataclass
class TeamState:
    form: dict[int, dict[str, Ewma]] = field(default_factory=dict)
    venue: dict[str, dict[str, Ewma]] = field(default_factory=dict)
    last_date: str | None = None
    recent_dates: deque = field(default_factory=lambda: deque(maxlen=64))
    first_season: dict[str, str] = field(default_factory=dict)

    def touch(self, date: str) -> None:
        self.last_date = date
        self.recent_dates.append(date)

    def matches_within(self, as_of: str, days: int) -> int:
        cutoff = (pd_ts(as_of) - _days(days)).strftime("%Y-%m-%d")
        return sum(1 for d in self.recent_dates if d > cutoff)


def pd_ts(value: str):
    import pandas as pd

    return pd.Timestamp(value)


def _days(n: int):
    import pandas as pd

    return pd.Timedelta(days=n)


class RollingState:
    """Per-team rolling state, updated strictly in date order."""

    def __init__(self, half_lives=(5, 12), venue_half_life: int = 8, prior_n: int = 4):
        self.half_lives = [int(h) for h in half_lives]
        self.venue_half_life = int(venue_half_life)
        self.prior_n = int(prior_n)
        self.teams: dict[int, TeamState] = {}
        self._sum: dict[tuple[str, str, str], dict[str, float]] = {}
        self._cnt: dict[tuple[str, str, str], dict[str, int]] = {}

    # -- means -------------------------------------------------------------
    def _ewma(self, half_life: int) -> Ewma:
        return Ewma(alpha=1.0 - 0.5 ** (1.0 / half_life), prior_n=self.prior_n)

    def _mean(self, league: str, season: str, view: str, stat: str) -> float:
        key = (league, season, view)
        total = self._sum.get(key, {}).get(stat, 0.0)
        n = self._cnt.get(key, {}).get(stat, 0)
        return total / n if n else float("nan")

    def _accumulate(self, league: str, season: str, view: str, values: dict[str, float]) -> None:
        key = (league, season, view)
        sums = self._sum.setdefault(key, {})
        cnts = self._cnt.setdefault(key, {})
        for stat, value in values.items():
            if value is None or not math.isfinite(value):
                continue
            sums[stat] = sums.get(stat, 0.0) + float(value)
            cnts[stat] = cnts.get(stat, 0) + 1

    # -- team lifecycle ----------------------------------------------------
    def _new_team(self, league: str, season: str) -> TeamState:
        state = TeamState()
        for hl in self.half_lives:
            state.form[hl] = {stat: self._ewma(hl) for stat in FORM_STATS}
        for view in ("home", "away"):
            state.venue[view] = {stat: self._ewma(self.venue_half_life) for stat in VENUE_STATS}
        for hl in self.half_lives:
            for stat in FORM_STATS:
                state.form[hl][stat].start(self._mean(league, season, "overall", stat))
        for view in ("home", "away"):
            for stat in VENUE_STATS:
                state.venue[view][stat].start(self._mean(league, season, view, stat))
        return state

    def team(self, team_id: int) -> TeamState | None:
        return self.teams.get(int(team_id))

    # -- updates -----------------------------------------------------------
    def apply_match(self, row, xg_h: float = float("nan"), xg_a: float = float("nan")) -> None:
        """Apply one match's result to the rolling state."""
        league = str(row["league"])
        season = str(row["season"])
        date = str(row["date"])
        h = int(row["home_id"])
        a = int(row["away_id"])
        hg = float(row["hg"])
        ag = float(row["ag"])

        hs = _num(row.get("hs"))
        sast = _num(row.get("as_"))
        hst = _num(row.get("hst"))
        ast = _num(row.get("ast"))

        pts_h = 3.0 if hg > ag else 1.0 if hg == ag else 0.0
        pts_a = 3.0 if ag > hg else 1.0 if ag == hg else 0.0

        home_values = {"gf": hg, "ga": ag, "shots": hs, "sot": hst, "xgf": xg_h, "xga": xg_a, "pts": pts_h}
        away_values = {"gf": ag, "ga": hg, "shots": sast, "sot": ast, "xgf": xg_a, "xga": xg_h, "pts": pts_a}

        self._accumulate(league, season, "overall", home_values)
        self._accumulate(league, season, "overall", away_values)
        self._accumulate(league, season, "home", home_values)
        self._accumulate(league, season, "away", away_values)

        for team_id, values, view in ((h, home_values, "home"), (a, away_values, "away")):
            state = self.teams.get(team_id)
            if state is None:
                state = self._new_team(league, season)
                self.teams[team_id] = state
            state.first_season.setdefault(league, season)
            for hl in self.half_lives:
                for stat in FORM_STATS:
                    state.form[hl][stat].update(values[stat])
            for stat in VENUE_STATS:
                state.venue[view][stat].update(values[stat])
            state.touch(date)

    # -- reads -------------------------------------------------------------
    def form(self, team_id: int, half_life: int, stat: str) -> float:
        state = self.teams.get(int(team_id))
        if state is None:
            return float("nan")
        ew = state.form.get(half_life, {}).get(stat)
        return ew.s if ew is not None else float("nan")

    def venue(self, team_id: int, view: str, stat: str) -> float:
        state = self.teams.get(int(team_id))
        if state is None:
            return float("nan")
        ew = state.venue.get(view, {}).get(stat)
        return ew.s if ew is not None else float("nan")

    def days_since(self, team_id: int, as_of: str, cap: int = 14) -> float:
        state = self.teams.get(int(team_id))
        if state is None or state.last_date is None:
            return float("nan")
        delta = (pd_ts(as_of) - pd_ts(state.last_date)).days
        return float(min(delta, cap))

    def matches_within(self, team_id: int, as_of: str, days: int) -> float:
        state = self.teams.get(int(team_id))
        if state is None:
            return float("nan")
        cutoff = (pd_ts(as_of) - _days(days)).strftime("%Y-%m-%d")
        return float(sum(1 for d in state.recent_dates if d > cutoff))

    def is_newly_promoted(self, team_id: int, league: str, season: str) -> float:
        """First season this club appears in this league in the data."""
        state = self.teams.get(int(team_id))
        if state is None:
            return float("nan")
        first = state.first_season.get(league)
        if first is None:
            return float("nan")
        return 1.0 if first == season else 0.0


def _num(value) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")
