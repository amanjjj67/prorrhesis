"""Live predictions for the upcoming fixtures.

The backtest measures models on history; this module puts the same models to
work on the next matches. The ensemble is fitted on the out-of-fold rows of the
last evaluation run, which are historical predictions, so no future
information can reach the live numbers.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Config, load_config
from .ensemble.calibrate import EnsembleCalibrator
from .ensemble.stack import EnsembleStacker
from .eval.walkforward import load_matches, oof_frame

DEFAULT_LIVE_MODELS = ["freq", "elo", "dc", "market_close", "market_open"]


def latest_run(conn: sqlite3.Connection, kind: str | None = None) -> str | None:
    sql = "SELECT run_id FROM runs"
    params: list = []
    if kind:
        sql += " WHERE kind = ?"
        params.append(kind)
    sql += " ORDER BY created_at DESC LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    return str(row["run_id"]) if row else None


def oof_triples(conn: sqlite3.Connection, run_id: str,
                members: list[str]) -> tuple[list[int], np.ndarray]:
    """Out-of-fold triples of the given members, aligned on match id."""
    oof = oof_frame(conn, run_id)
    oof = oof[oof["status"] == "ok"]
    ids = sorted({int(m) for m in oof["match_id"]})
    index = {mid: i for i, mid in enumerate(ids)}
    triples = np.full((len(ids), len(members), 3), np.nan)
    for k, name in enumerate(members):
        sub = oof[oof["model"] == name]
        for r in sub.itertuples():
            i = index.get(int(r.match_id))
            if i is None:
                continue
            triples[i, k, 0] = float(r.pH)
            triples[i, k, 1] = float(r.pD)
            triples[i, k, 2] = float(r.pA)
    return ids, triples


@dataclass
class LivePredictor:
    """Fits the models on history and predicts the upcoming fixtures."""

    cfg: Config
    conn: sqlite3.Connection
    model_names: list[str] = field(default_factory=lambda: list(DEFAULT_LIVE_MODELS))
    run_id: str | None = None
    as_of: str = ""
    models: list = field(default_factory=list)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    stacker: EnsembleStacker | None = None
    calibrator: EnsembleCalibrator | None = None
    ensemble_method: str = "equal"
    calibration_method: str = "none"

    def __post_init__(self) -> None:
        if self.run_id is None:
            self.run_id = latest_run(self.conn)
        self.members = [n for n in self.model_names if n != "market_close"]

    # -- setup -------------------------------------------------------------
    def prepare(self, as_of: str | None = None) -> "LivePredictor":
        from .eval.tuning import load_dc_tuning
        from .eval.walkforward import build_feature_table
        from .models import build_models
        from .models.elo import load_tuning as load_elo_tuning

        self.as_of = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.history = load_matches(self.conn)
        self.history = self.history[self.history["date"] < self.as_of]

        self.models = build_models(
            self.cfg, self.model_names,
            tuning=load_elo_tuning(self.cfg), dc_tuning=load_dc_tuning(self.cfg))

        feature_models = [m for m in self.models if getattr(m, "needs_features", False)]
        if feature_models:
            table = build_feature_table(self.conn, self.cfg)
            for m in feature_models:
                m.attach_features(table)

        for m in self.models:
            m.fit(self.history, as_of=self.as_of)

        self._fit_ensemble()
        return self

    def _fit_ensemble(self) -> None:
        """Fit the stacker and the calibrator on the last run's out-of-fold rows."""
        if not self.members or not self.run_id:
            return
        stack_cfg = self.cfg.stack or {}
        cal_cfg = self.cfg.calibrate or {}
        ids, triples = oof_triples(self.conn, self.run_id, self.members)
        if len(ids) == 0:
            return

        oof = oof_frame(self.conn, self.run_id)
        block_of = dict(zip(oof["match_id"].astype(int), oof["block_start"].astype(str)))
        starts = sorted({block_of.get(m, "") for m in ids})
        block_index = {s: i for i, s in enumerate(starts)}
        blocks = np.array([block_index.get(block_of.get(m, ""), 0) for m in ids], dtype=float)

        matches = load_matches(self.conn)
        res_of = dict(zip(matches["id"].astype(int), matches["res"].astype(str)))
        labels = np.array([{"H": 0, "D": 1, "A": 2}.get(res_of.get(int(m), ""), -1)
                           for m in ids], dtype=int)

        self.stacker = EnsembleStacker(
            self.members,
            C=float(stack_cfg.get("C", 1.0)),
            min_rows=int(stack_cfg.get("min_rows", 800)),
            methods=list(stack_cfg.get("methods", ["logit", "logpool", "equal"])),
            recency_half_life_blocks=int(stack_cfg.get("recency_half_life_blocks", 40)),
        )
        self.stacker.fit(triples, labels, blocks, current_block=int(blocks.max()) + 1)

        ens = oof_frame(self.conn, self.run_id, "ensemble")
        ens = ens[ens["status"] == "ok"]
        if len(ens):
            P = ens[["pH", "pD", "pA"]].to_numpy(dtype=float)
            y = np.array([{"H": 0, "D": 1, "A": 2}.get(res_of.get(int(m), ""), -1)
                          for m in ens["match_id"]], dtype=int)
            keep = y >= 0
            P, y = P[keep], y[keep]
            cal_blocks = np.zeros(len(P), dtype=float)
            self.calibrator = EnsembleCalibrator(
                methods=list(cal_cfg.get("methods",
                                        ["none", "temperature", "dirichlet", "isotonic"])),
                min_rows=dict(cal_cfg.get("min_rows", {})),
                recency_half_life_blocks=int(stack_cfg.get("recency_half_life_blocks", 40)),
            )
            self.calibrator.fit(P, y, cal_blocks, current_block=1)

    # -- prediction --------------------------------------------------------
    def predict(self, fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Per-model triples for the given fixtures."""
        out: dict[str, pd.DataFrame] = {}
        for m in self.models:
            out[m.name] = m.predict(fixtures)
        return out

    def predict_ensemble(self, fixtures: pd.DataFrame,
                         model_probs: dict[str, pd.DataFrame] | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The ensemble and the calibrated ensemble for the given fixtures."""
        n = len(fixtures)
        if model_probs is None:
            model_probs = self.predict(fixtures)
        triples = np.full((n, len(self.members), 3), np.nan)
        for k, name in enumerate(self.members):
            frame = model_probs.get(name)
            if frame is None:
                continue
            for i in range(n):
                if "status" in frame.columns and str(frame["status"].iloc[i]) != "ok":
                    continue
                triples[i, k, :] = [frame["pH"].iloc[i], frame["pD"].iloc[i], frame["pA"].iloc[i]]

        if self.stacker is None:
            P = np.full((n, 3), 1 / 3)
            self.ensemble_method = "equal"
        else:
            outcome = self.stacker.predict(triples)
            P = outcome.probs
            self.ensemble_method = outcome.method

        if self.calibrator is None:
            Q = P
            self.calibration_method = "none"
        else:
            cal = self.calibrator.predict(P)
            Q = cal.probs
            self.calibration_method = cal.method

        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
        out["status"] = "ok"
        out["method"] = self.ensemble_method
        cal_frame = pd.DataFrame({"pH": Q[:, 0], "pD": Q[:, 1], "pA": Q[:, 2]}, index=fixtures.index)
        cal_frame["status"] = "ok"
        cal_frame["method"] = self.calibration_method
        return out, cal_frame


def upcoming_fixtures(conn: sqlite3.Connection, league: str, days: int = 7,
                      today: str | None = None) -> pd.DataFrame:
    """The upcoming fixtures of a league, from the feed or derived from the season."""
    from .sim.schedule import remaining_schedule
    from .sim.season import current_season

    start = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end = (pd.Timestamp(start) + pd.Timedelta(days=int(days))).strftime("%Y-%m-%d")

    rows = conn.execute(
        "SELECT id, league, season, date, kickoff_utc, home_id, away_id, "
        "odds_open_h, odds_open_d, odds_open_a FROM fixtures "
        "WHERE league = ? AND date >= ? AND date <= ? ORDER BY date, home_id",
        (league, start, end),
    ).fetchall()
    if rows:
        frame = pd.DataFrame([dict(r) for r in rows])
    else:
        season = current_season(conn, league)
        derived = remaining_schedule(conn, league, season) if season else []
        if not derived:
            return pd.DataFrame()
        frame = pd.DataFrame(derived)
        for col in ("kickoff_utc", "odds_open_h", "odds_open_d", "odds_open_a"):
            frame[col] = None
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if len(frame) == 0:
        return frame
    frame = frame.reset_index(drop=True)
    frame["kickoff_utc"] = frame["kickoff_utc"].astype("string")
    for col in ("odds_open_h", "odds_open_d", "odds_open_a"):
        frame[col] = frame[col].astype(float)
    return frame


def run_live_predictions(cfg: Config, conn: sqlite3.Connection, days: int = 7,
                         as_of: str | None = None) -> dict:
    """Predict the upcoming fixtures and append every triple to the ledger.

    One row per base model that produced a triple, plus `ensemble` and
    `ensemble_cal`. The grid is stored for the goal models.
    """
    from .ledger import append_prediction
    from .sim.season import current_season

    predictor = LivePredictor(cfg=cfg, conn=conn)
    predictor.prepare(as_of)

    report = {"appended": 0, "skipped": 0, "late": 0, "fixtures": 0,
              "errors": [], "leagues": {}}

    for league in list(cfg.leagues.keys()):
        fixtures = upcoming_fixtures(conn, league, days)
        if len(fixtures) == 0:
            report["leagues"][league] = {"fixtures": 0, "appended": 0}
            continue
        season = str(fixtures["season"].iloc[0]) if "season" in fixtures.columns else ""
        model_probs = predictor.predict(fixtures)
        ensemble, calibrated = predictor.predict_ensemble(fixtures, model_probs)

        dc = next((m for m in predictor.models if m.name == "dc"), None)
        grids = []
        for _, row in fixtures.iterrows():
            if dc is None:
                grids.append(None)
                continue
            grids.append(dc.grid_for(str(row["league"]), int(row["home_id"]),
                                     int(row["away_id"])))

        league_report = {"fixtures": len(fixtures), "appended": 0, "skipped": 0,
                         "late": 0}
        for i, (_, fixture) in enumerate(fixtures.iterrows()):
            report["fixtures"] += 1
            common = {
                "league": league,
                "date": str(fixture["date"]),
                "home_id": int(fixture["home_id"]),
                "away_id": int(fixture["away_id"]),
                "config_hash": cfg.config_hash,
                "kickoff_utc": (None if fixture.get("kickoff_utc") is None
                                or pd.isna(fixture.get("kickoff_utc"))
                                else str(fixture["kickoff_utc"])),
            }
            grid = grids[i]
            grid_json = None
            if grid is not None:
                grid_json = json.dumps([[round(float(v), 8) for v in row]
                                        for row in grid.tolist()])

            for model_name, frame in list(model_probs.items()) + [
                    ("ensemble", ensemble), ("ensemble_cal", calibrated)]:
                if "status" in frame.columns and str(frame["status"].iloc[i]) != "ok":
                    continue
                digest = append_prediction(
                    conn,
                    pH=float(frame["pH"].iloc[i]), pD=float(frame["pD"].iloc[i]),
                    pA=float(frame["pA"].iloc[i]), model=model_name,
                    grid_json=grid_json if model_name in ("dc", "xgpoisson") else None,
                    **common,
                )
                if digest is None:
                    if _is_late(conn, common, model_name):
                        league_report["late"] += 1
                        report["late"] += 1
                    else:
                        league_report["skipped"] += 1
                        report["skipped"] += 1
                else:
                    league_report["appended"] += 1
                    report["appended"] += 1
        report["leagues"][league] = league_report

    return report


def _is_late(conn: sqlite3.Connection, common: dict, model: str) -> bool:
    """True when the row was rejected because kick-off had already passed."""
    row = conn.execute(
        "SELECT 1 FROM ingest_log WHERE source = 'ledger' AND kind = 'late_prediction' "
        "AND note LIKE ? LIMIT 1",
        (f"%{common['date']}%{model}%",),
    ).fetchone()
    return row is not None
