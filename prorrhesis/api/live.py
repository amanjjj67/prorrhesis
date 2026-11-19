"""The live prediction service shared by the API endpoints.

Fitting the models takes seconds, so one fitted set is kept in memory and
reused. Every call is serialised by a lock: the models are not thread-safe and
the sqlite connection is shared between the request threads.

The features of an upcoming fixture are computed from the same chronological
state machine that trained the models, so a fixture that has not been played
yet is described by exactly the information available before its kick-off.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..config import Config
from ..db import connect
from ..predict import LivePredictor

# Upcoming fixtures have no row in `matches`, so they get a synthetic id that
# cannot collide with a match id. The feature table is keyed by this id.
SYNTHETIC_ID_BASE = -1_000_000

MODEL_ORDER = ["freq", "home", "elo", "dc", "xgpoisson", "market_open",
               "market_close", "gbm", "gbm_market_aug"]


def _day_before(date_str: str) -> str:
    return (pd.Timestamp(date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def display_names(conn: sqlite3.Connection) -> dict[int, str]:
    """Club names to show: the ingested name, or a readable form of the slug."""
    rows = conn.execute(
        "SELECT t.id AS id, t.canon AS canon, "
        "(SELECT alias FROM team_aliases a WHERE a.team_id = t.id "
        " AND a.source = 'footballdata' ORDER BY a.alias LIMIT 1) AS alias "
        "FROM teams t"
    ).fetchall()
    out: dict[int, str] = {}
    for r in rows:
        name = r["alias"] or str(r["canon"]).replace("-", " ").title()
        out[int(r["id"])] = str(name)
    return out


class LiveService:
    """Fits the models once and predicts with them."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._predictor: LivePredictor | None = None
        self._store = None
        self._feature_table: pd.DataFrame | None = None
        self._as_of: str = ""
        self._names: dict[int, str] = {}

    # -- plumbing ----------------------------------------------------------
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.cfg.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def names(self) -> dict[int, str]:
        with self._lock:
            if not self._names:
                self._names = display_names(self.connection())
            return self._names

    # -- fitting -----------------------------------------------------------
    def _build(self) -> None:
        from ..eval.tuning import load_dc_tuning
        from ..eval.walkforward import load_matches
        from ..features.store import FeatureStore, elo_params, load_clubelo
        from ..models import build_models
        from ..models.elo import load_tuning as load_elo_tuning

        conn = self.connection()
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._as_of = as_of

        matches = load_matches(conn, end=_day_before(as_of))
        store = FeatureStore(cfg=self.cfg, elo=elo_params(self.cfg),
                             clubelo=load_clubelo(conn))
        feats = store.build_table(matches.reset_index(drop=True))
        aligned = matches.reset_index(drop=True).loc[feats.index]
        table = feats.reset_index(drop=True)
        table.insert(0, "date", aligned["date"].to_numpy())
        table.insert(0, "match_id", aligned["id"].to_numpy())
        self._store = store
        self._feature_table = table

        predictor = LivePredictor(
            cfg=self.cfg, conn=conn, as_of=as_of,
            model_names=["freq", "elo", "dc", "xgpoisson", "market_open",
                         "market_close", "gbm"],
        )
        predictor.history = matches
        predictor.models = build_models(
            self.cfg, predictor.model_names,
            tuning=load_elo_tuning(self.cfg), dc_tuning=load_dc_tuning(self.cfg))
        for m in predictor.models:
            if getattr(m, "needs_features", False):
                m.attach_features(table)
            m.fit(matches, as_of=as_of)
        predictor._fit_ensemble()
        self._predictor = predictor

    def predictor(self, refresh: bool = False) -> LivePredictor | None:
        with self._lock:
            if refresh or self._predictor is None:
                self._build()
            return self._predictor

    # -- fixtures ----------------------------------------------------------
    def _fixtures_from_table(self, league: str, start: str, end: str) -> pd.DataFrame:
        conn = self.connection()
        rows = conn.execute(
            "SELECT id, league, season, date, kickoff_utc, home_id, away_id, "
            "odds_open_h, odds_open_d, odds_open_a FROM fixtures "
            "WHERE league = ? AND date >= ? AND date <= ? ORDER BY date, home_id",
            (league, start, end),
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame([dict(r) for r in rows])
        frame["estimated"] = False
        return frame

    def _fixtures_derived(self, league: str) -> pd.DataFrame:
        from ..sim.schedule import remaining_schedule
        from ..sim.season import current_season

        conn = self.connection()
        season = current_season(conn, league)
        if season is None:
            return pd.DataFrame()
        rows = remaining_schedule(conn, league, season)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        for col in ("kickoff_utc", "odds_open_h", "odds_open_d", "odds_open_a"):
            frame[col] = None
        return frame

    def fixtures(self, league: str, days: int = 7) -> pd.DataFrame:
        """Upcoming fixtures of a league, from the feed or derived from the season."""
        today = pd.Timestamp(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        start = today.strftime("%Y-%m-%d")
        end = (today + timedelta(days=int(days))).strftime("%Y-%m-%d")

        frame = self._fixtures_from_table(league, start, end)
        if len(frame) == 0:
            derived = self._fixtures_derived(league)
            if len(derived):
                frame = derived[(derived["date"] >= start) & (derived["date"] <= end)]
        if len(frame) == 0:
            return pd.DataFrame()

        frame = frame.reset_index(drop=True)
        frame["id"] = [SYNTHETIC_ID_BASE - i for i in range(len(frame))]
        frame["kickoff_utc"] = frame["kickoff_utc"].astype("string")
        for col in ("odds_open_h", "odds_open_d", "odds_open_a"):
            frame[col] = frame[col].astype(float)
        return frame

    # -- prediction --------------------------------------------------------
    def _feature_rows(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Features of the upcoming fixtures, from the state before kick-off."""
        from ..features.schema import FEATURE_COLUMNS

        if self._store is None or len(fixtures) == 0:
            return pd.DataFrame(columns=["match_id", "date", *FEATURE_COLUMNS])
        rows = []
        for _, fixture in fixtures.iterrows():
            row = {
                "league": str(fixture["league"]),
                "season": str(fixture.get("season", "")),
                "home_id": int(fixture["home_id"]),
                "away_id": int(fixture["away_id"]),
                "id": int(fixture["id"]),
                "odds_open_h": fixture.get("odds_open_h"),
                "odds_open_d": fixture.get("odds_open_d"),
                "odds_open_a": fixture.get("odds_open_a"),
            }
            feats = self._store.build_features(row, self._as_of)
            feats["match_id"] = int(fixture["id"])
            feats["date"] = str(fixture["date"])
            rows.append(feats)
        return pd.DataFrame(rows)

    def predict(self, fixtures: pd.DataFrame) -> dict | None:
        """Ensemble, calibrated ensemble, per-model triples and the DC grids."""
        predictor = self.predictor()
        if predictor is None or len(fixtures) == 0:
            return None

        # the GBM reads its features by id, so the fixture rows are added to
        # the historical table before predicting
        if self._feature_table is not None:
            extra = self._feature_rows(fixtures)
            combined = self._feature_table
            if len(extra):
                combined = pd.concat([self._feature_table, extra], ignore_index=True)
            for m in predictor.models:
                if getattr(m, "needs_features", False):
                    m.attach_features(combined)

        model_probs = predictor.predict(fixtures)
        ensemble, calibrated = predictor.predict_ensemble(fixtures, model_probs)

        grids: list[np.ndarray | None] = []
        dc = next((m for m in predictor.models if m.name == "dc"), None)
        for _, row in fixtures.iterrows():
            if dc is None:
                grids.append(None)
                continue
            grids.append(dc.grid_for(str(row["league"]), int(row["home_id"]),
                                     int(row["away_id"])))

        per_model: dict[str, list] = {}
        for name, frame in model_probs.items():
            per_model[name] = [
                {"pH": float(frame["pH"].iloc[i]),
                 "pD": float(frame["pD"].iloc[i]),
                 "pA": float(frame["pA"].iloc[i]),
                 "status": str(frame["status"].iloc[i])}
                for i in range(len(frame))
            ]

        ens = [
            {"pH": float(ensemble["pH"].iloc[i]), "pD": float(ensemble["pD"].iloc[i]),
             "pA": float(ensemble["pA"].iloc[i]),
             "status": str(ensemble["status"].iloc[i])}
            for i in range(len(ensemble))
        ]
        cal = [
            {"pH": float(calibrated["pH"].iloc[i]), "pD": float(calibrated["pD"].iloc[i]),
             "pA": float(calibrated["pA"].iloc[i]),
             "status": str(calibrated["status"].iloc[i])}
            for i in range(len(calibrated))
        ]
        return {
            "models": per_model,
            "ensemble": ens,
            "ensemble_cal": cal,
            "grids": grids,
            "method": predictor.ensemble_method,
            "calibration_method": predictor.calibration_method,
        }

    # -- helpers used by the endpoints -------------------------------------
    def lambdas(self, league: str, home_id: int, away_id: int) -> tuple[float, float, float]:
        predictor = self.predictor()
        if predictor is None:
            return 1.3, 1.1, 0.0
        dc = next((m for m in predictor.models if m.name == "dc"), None)
        if dc is None:
            return 1.3, 1.1, 0.0
        return dc.lambdas(str(league), int(home_id), int(away_id))

    def uncertain_teams(self, league: str) -> set[int]:
        predictor = self.predictor()
        if predictor is None:
            return set()
        dc = next((m for m in predictor.models if m.name == "dc"), None)
        if dc is None:
            return set()
        return set(dc.uncertain_by_league.get(str(league), set()))

    def league_mean_goals(self, league: str) -> tuple[float, float] | None:
        conn = self.connection()
        row = conn.execute(
            "SELECT AVG(hg) h, AVG(ag) a FROM matches WHERE league = ?", (league,)
        ).fetchone()
        if row is None or row["h"] is None:
            return None
        return float(row["h"]), float(row["a"])
