"""The only feature entry point (spec sections 8.2 and 8.3).

`build_features(match_row, as_of)` is the single function allowed to produce
features. It reads only from the chronological state machine and from tables
filtered by `date < as_of`. No model reads the `matches` table directly for
features.
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd

from ..config import Config
from ..models.elo import DEFAULT_HOME_ADV, RatingState, load_tuning
from ..models.market import shin
from .rolling import RollingState
from .schema import (
    FEATURE_COLUMNS,
    FORM_HALF_LIVES,
    FORM_STATS,
    MARKET_FEATURE_COLUMNS,
    PRIOR_N,
    VENUE_HALF_LIFE,
    VENUE_STATS,
)
from .xgproxy import XgProxyModel, build_season_models

# Nominal number of matches in a full league-season; used for season_progress.
SEASON_TOTALS = {"E0": 380, "SP1": 380, "I1": 380, "D1": 306, "F1": 380}
LEAGUE_CODES = ("D1", "E0", "F1", "I1", "SP1")


def season_total(league: str, season: str) -> int:
    if league == "F1" and str(season) >= "2324":
        return 306
    return SEASON_TOTALS.get(league, 380)


def elo_params(cfg: Config) -> dict:
    t = load_tuning(cfg)
    return {
        "K": float(t.get("K", cfg.elo.get("K", 20))),
        "H": dict(t.get("H") or cfg.elo.get("H", DEFAULT_HOME_ADV)),
        "rho": float(t.get("rho", cfg.elo.get("rho_season", 0.25))),
        "promoted_penalty": float(cfg.elo.get("promoted_penalty", 50)),
    }


def _finite(value) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


class FeatureStore:
    """Builds features for matches in strict date order.

    The state is built in one pass: for each date the features are read from
    the current state, which holds only matches strictly before that date, and
    only then are that date's results applied.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        elo: dict | None = None,
        clubelo: dict[str, dict[int, float]] | None = None,
        half_lives: tuple[int, ...] = FORM_HALF_LIVES,
        venue_half_life: int = VENUE_HALF_LIFE,
        prior_n: int = PRIOR_N,
    ):
        self.cfg = cfg
        self.elo = elo or {}
        self.clubelo = clubelo or {}
        self.half_lives = tuple(half_lives)
        self.venue_half_life = int(venue_half_life)
        self.prior_n = int(prior_n)
        self.state = RollingState(half_lives=self.half_lives,
                                  venue_half_life=self.venue_half_life,
                                  prior_n=self.prior_n)
        self.rating_state = RatingState(
            K=float(self.elo.get("K", 20)),
            home_adv=dict(self.elo.get("H", DEFAULT_HOME_ADV)),
            rho=float(self.elo.get("rho", 0.25)),
            promoted_penalty=float(self.elo.get("promoted_penalty", 50)),
        )
        self.proxy_models: dict[str, XgProxyModel] = {}
        self._season_count: dict[tuple[str, str], int] = {}
        self._clubelo_dates: list[str] = sorted(self.clubelo)

    # -- helpers -----------------------------------------------------------
    def _xg_for(self, row) -> tuple[float, float]:
        model = self.proxy_models.get(str(row["season"]))
        if model is None or not model.fitted:
            return float("nan"), float("nan")
        return model.predict_match(row)

    def _clubelo_elo(self, team_id: int, as_of: str) -> float:
        """Rating from the nearest snapshot dated strictly before the match."""
        chosen = None
        for d in self._clubelo_dates:
            if d < as_of:
                chosen = d
            else:
                break
        if chosen is None:
            return float("nan")
        return _finite(self.clubelo[chosen].get(int(team_id)))

    def _market_open(self, row) -> tuple[float, float, float]:
        h = _finite(row.get("odds_open_h"))
        d = _finite(row.get("odds_open_d"))
        a = _finite(row.get("odds_open_a"))
        if not all(math.isfinite(v) and v > 0 for v in (h, d, a)):
            return float("nan"), float("nan"), float("nan")
        try:
            p = shin(np.array([h, d, a], dtype=float))
        except Exception:
            return float("nan"), float("nan"), float("nan")
        return float(p[0]), float(p[1]), float(p[2])

    def _elo_diff(self, row, as_of: str) -> float:
        h = int(row["home_id"])
        a = int(row["away_id"])
        rh = self.rating_state.ratings.get(h)
        ra = self.rating_state.ratings.get(a)
        if rh is None or ra is None:
            return float("nan")
        H = float(self.elo.get("H", DEFAULT_HOME_ADV).get(str(row["league"]), 65.0))
        return float(rh + H - ra)

    # -- the feature read --------------------------------------------------
    def _features_for(self, row, as_of: str, season_rank: int = 0) -> dict:
        league = str(row["league"])
        season = str(row["season"])
        h = int(row["home_id"])
        a = int(row["away_id"])
        out: dict[str, float] = {}

        for hl in self.half_lives:
            for stat in FORM_STATS:
                out[f"f{hl}_{stat}_h"] = self.state.form(h, hl, stat)
                out[f"f{hl}_{stat}_a"] = self.state.form(a, hl, stat)
        for stat in VENUE_STATS:
            out[f"v_{stat}_h"] = self.state.venue(h, "home", stat)
            out[f"v_{stat}_a"] = self.state.venue(a, "away", stat)

        out["days_since_h"] = self.state.days_since(h, as_of, cap=14)
        out["days_since_a"] = self.state.days_since(a, as_of, cap=14)
        out["m14_h"] = self.state.matches_within(h, as_of, 14)
        out["m14_a"] = self.state.matches_within(a, as_of, 14)
        out["m30_h"] = self.state.matches_within(h, as_of, 30)
        out["m30_a"] = self.state.matches_within(a, as_of, 30)

        played = self._season_count.get((league, season), 0) + season_rank
        out["season_progress"] = float(played) / float(season_total(league, season))
        out["league"] = float(LEAGUE_CODES.index(league)) if league in LEAGUE_CODES else float("nan")
        out["month"] = float(pd.Timestamp(as_of).month)
        out["new_promoted_h"] = self.state.is_newly_promoted(h, league, season)
        out["new_promoted_a"] = self.state.is_newly_promoted(a, league, season)

        out["elo_diff"] = self._elo_diff(row, as_of)
        out["clubelo_diff"] = (self._clubelo_elo(h, as_of) - self._clubelo_elo(a, as_of)
                               if self.clubelo else float("nan"))

        mh, md, ma = self._market_open(row)
        out["mkt_open_h"] = mh
        out["mkt_open_d"] = md
        out["mkt_open_a"] = ma
        return out

    # -- public API --------------------------------------------------------
    def build_table(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Features for every match in `matches`, one chronological pass."""
        if matches is None or len(matches) == 0:
            return pd.DataFrame()
        matches = matches.sort_values(["date", "league", "home_id", "away_id"])
        self.proxy_models = build_season_models(matches)

        rows: list[dict] = []
        for date, grp in matches.groupby("date", sort=True):
            grp = grp.sort_values(["league", "home_id", "away_id"])
            self.rating_state.prepare(grp)
            ranks = {}
            feats = []
            for _, row in grp.iterrows():
                key = (str(row["league"]), str(row["season"]))
                ranks[key] = ranks.get(key, 0) + 1
                feats.append(self._features_for(row, str(date), ranks[key]))
            rows.extend(feats)
            for _, row in grp.iterrows():
                self.state.apply_match(row, *self._xg_for(row))
                key = (str(row["league"]), str(row["season"]))
                self._season_count[key] = self._season_count.get(key, 0) + 1
            self.rating_state.apply(grp)
        return pd.DataFrame(rows, index=matches.index)

    def build_features(self, match_row, as_of: str) -> dict:
        """Features for a single match row, using state built only from `date < as_of`."""
        return self._features_for(match_row, str(as_of))


def load_clubelo(conn: sqlite3.Connection) -> dict[str, dict[int, float]]:
    """Snapshots from the elo_ext table, keyed by as-of date."""
    try:
        rows = pd.read_sql_query("SELECT team_id, asof, elo FROM elo_ext ORDER BY asof", conn)
    except Exception:
        return {}
    out: dict[str, dict[int, float]] = {}
    for team_id, asof, elo in zip(rows["team_id"], rows["asof"], rows["elo"]):
        out.setdefault(str(asof), {})[int(team_id)] = float(elo)
    return out


def build_store(cfg: Config, conn: sqlite3.Connection, as_of: str | None = None,
                matches: pd.DataFrame | None = None) -> FeatureStore:
    """A store whose state covers every match strictly before `as_of`."""
    from ..eval.walkforward import load_matches

    store = FeatureStore(cfg=cfg, elo=elo_params(cfg), clubelo=load_clubelo(conn))
    if matches is None:
        matches = load_matches(conn, end=None if as_of is None else _day_before(as_of))
    if matches is not None and len(matches):
        store.build_table(matches)
    return store


def _day_before(date_str: str) -> str:
    return (pd.Timestamp(date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def build_feature_table(
    matches: pd.DataFrame,
    cfg: Config | None = None,
    elo: dict | None = None,
    clubelo: dict[str, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """Features for every match, one row per match, carrying match_id and date.

    One chronological pass over `matches`: each row sees only matches that
    happened strictly earlier. This is the table the GBM trains on.
    """
    if matches is None or len(matches) == 0:
        return pd.DataFrame(columns=["match_id", "date", *FEATURE_COLUMNS])

    ordered_input = matches.reset_index(drop=True)
    store = FeatureStore(cfg=cfg, elo=elo, clubelo=clubelo)
    feats = store.build_table(ordered_input)
    aligned = ordered_input.loc[feats.index]

    out = feats.reset_index(drop=True)
    out.insert(0, "date", aligned["date"].to_numpy())
    out.insert(0, "match_id", aligned["id"].to_numpy() if "id" in aligned.columns
               else aligned.index.to_numpy())
    return out


def build_features(match_row, as_of: str, db: sqlite3.Connection | None = None) -> dict:
    """The single entry point for features (spec section 8.3)."""
    if db is None:
        raise ValueError("build_features needs a database connection")
    from ..config import load_config

    cfg = load_config()
    store = build_store(cfg, db, as_of=str(as_of))
    return store.build_features(match_row, str(as_of))
