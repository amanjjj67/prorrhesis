"""Walk-forward evaluation driver (spec section 10.1)."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..config import Config
from ..ensemble.calibrate import EnsembleCalibrator
from ..ensemble.stack import EnsembleStacker
from .metrics import accuracy, brier, ece, log_loss, rps, skill_score

RESULT_COLUMNS = ["hg", "ag", "res", "hthg", "htag", "hs", "as_", "hst", "ast",
                  "hc", "ac", "hf", "af", "hy", "ay", "hr", "ar"]
CLOSING_COLUMNS = ["odds_close_h", "odds_close_d", "odds_close_a", "odds_close_src"]
FIXTURE_COLUMNS = ["id", "league", "season", "date", "kickoff_utc", "home_id", "away_id",
                    "odds_open_h", "odds_open_d", "odds_open_a", "odds_open_src"]

MATCH_COLUMNS = [
    "id", "league", "season", "date", "kickoff_utc", "home_id", "away_id", "hg", "ag", "res",
    "hthg", "htag", "hs", "as_", "hst", "ast", "hc", "ac", "hf", "af", "hy", "ay", "hr", "ar",
    "odds_open_h", "odds_open_d", "odds_open_a", "odds_open_src",
    "odds_close_h", "odds_close_d", "odds_close_a", "odds_close_src",
]


@dataclass
class Block:
    start: str
    end: str


@dataclass
class WalkForwardResult:
    run_id: str
    blocks: int = 0
    matches: int = 0
    failed: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    ensemble: dict = field(default_factory=dict)
    ensemble_members: list[str] = field(default_factory=list)


def load_matches(conn: sqlite3.Connection, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    sql = f"SELECT {', '.join(MATCH_COLUMNS)} FROM matches"
    params: list = []
    clauses = []
    if start:
        clauses.append("date >= ?")
        params.append(start)
    if end:
        clauses.append("date <= ?")
        params.append(end)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY date, league, home_id, away_id"
    df = pd.read_sql_query(sql, conn, params=params)
    for col in ("date", "kickoff_utc"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    if "res" in df.columns:
        df["res"] = df["res"].astype("string")
    return df


def test_as_fixtures(test: pd.DataFrame) -> pd.DataFrame:
    """Strip everything that is only known after kickoff."""
    drop = [c for c in (RESULT_COLUMNS + CLOSING_COLUMNS) if c in test.columns]
    out = test.drop(columns=drop)
    missing = [c for c in FIXTURE_COLUMNS if c not in out.columns and c in test.columns]
    for c in missing:
        out[c] = test[c]
    keep = [c for c in FIXTURE_COLUMNS if c in out.columns]
    return out[keep].copy()


def week_blocks(start: str, end: str, block_days: int = 7) -> list[Block]:
    """Monday-start blocks covering [start, end]."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    monday = s - pd.Timedelta(days=int(s.weekday()))
    blocks = []
    cur = monday
    while cur <= e:
        nxt = cur + pd.Timedelta(days=block_days - 1)
        blocks.append(Block(cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = cur + pd.Timedelta(days=block_days)
    return blocks


def save_oof(conn: sqlite3.Connection, run_id: str, model: str, block_start: str,
             match_ids: list[int], frame: pd.DataFrame | None, status: str) -> None:
    rows = []
    if frame is not None and status == "ok":
        for mid, (_, r) in zip(match_ids, frame.iterrows()):
            st = str(r.get("status", "ok"))
            if st != "ok":
                rows.append((run_id, model, int(mid), block_start, None, None, None, st))
            else:
                rows.append((run_id, model, int(mid), block_start, float(r["pH"]), float(r["pD"]),
                             float(r["pA"]), "ok"))
    else:
        for mid in match_ids:
            rows.append((run_id, model, int(mid), block_start, None, None, None, status))
    conn.executemany(
        "INSERT INTO oof_predictions(run_id, model, match_id, block_start, pH, pD, pA, status) "
        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(run_id, model, match_id) DO UPDATE SET "
        "block_start=excluded.block_start, pH=excluded.pH, pD=excluded.pD, pA=excluded.pA, "
        "status=excluded.status",
        rows,
    )


def save_diagnostics(conn: sqlite3.Connection, run_id: str, block_start: str, stage: str,
                     scores: dict[str, float], selected: str) -> None:
    """Record every candidate method with its score and which one was picked."""
    rows = [
        (run_id, block_start, stage, str(method), float(score),
         1 if str(method) == selected else 0)
        for method, score in (scores or {}).items()
        if score is not None and np.isfinite(score)
    ]
    conn.executemany(
        "INSERT INTO ensemble_diagnostics(run_id, block_start, stage, method, score, selected) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(run_id, block_start, stage, method) DO UPDATE SET "
        "score=excluded.score, selected=excluded.selected",
        rows,
    )


LABEL_MAP = {"H": 0, "D": 1, "A": 2}


class EnsembleHistory:
    """Per-model triples of earlier blocks, used to fit the ensemble (I4)."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        self.triples: list[np.ndarray] = []
        self.blocks: list[float] = []
        self.labels: list[int] = []
        self.ens: list[np.ndarray] = []
        self.ens_blocks: list[float] = []
        self.ens_labels: list[int] = []

    def add_block(self, match_ids: list[int], triples_by_model: dict[str, np.ndarray],
                  block_index: int, labels: dict[int, int],
                  ensemble_probs: np.ndarray | None = None) -> None:
        for i, mid in enumerate(match_ids):
            row = np.full((len(self.names), 3), np.nan)
            for k, name in enumerate(self.names):
                block = triples_by_model.get(name)
                if block is not None and i < len(block):
                    row[k] = block[i]
            self.triples.append(row)
            self.blocks.append(float(block_index))
            self.labels.append(int(labels.get(int(mid), -1)))
        if ensemble_probs is not None:
            for i, mid in enumerate(match_ids):
                self.ens.append(np.asarray(ensemble_probs[i], dtype=float))
                self.ens_blocks.append(float(block_index))
                self.ens_labels.append(int(labels.get(int(mid), -1)))

    def matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.triples:
            return (np.zeros((0, len(self.names), 3)), np.zeros(0), np.zeros(0, dtype=int))
        return (np.stack(self.triples), np.asarray(self.blocks),
                np.asarray(self.labels, dtype=int))

    def ensemble_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.ens:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
        return (np.stack(self.ens), np.asarray(self.ens_blocks),
                np.asarray(self.ens_labels, dtype=int))


def probs_frame(P: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(P, dtype=float)
    return pd.DataFrame({"pH": arr[:, 0], "pD": arr[:, 1], "pA": arr[:, 2]})


def build_feature_table(conn: sqlite3.Connection, cfg: Config) -> pd.DataFrame:
    """Feature table for every match in the database, one chronological pass."""
    from ..features.store import build_feature_table as _build
    from ..features.store import elo_params, load_clubelo

    matches = load_matches(conn)
    if len(matches) == 0:
        return pd.DataFrame()
    return _build(matches, cfg=cfg, elo=elo_params(cfg), clubelo=load_clubelo(conn))


def walk_forward(
    conn: sqlite3.Connection,
    models: list,
    cfg: Config,
    run_id: str,
    start: str,
    end: str,
    progress: bool = False,
    ensemble_members: list[str] | None = None,
) -> WalkForwardResult:
    blocks = week_blocks(start, end, int(cfg.eval.get("block_days", 7)))
    matches = load_matches(conn)
    result = WalkForwardResult(run_id=run_id)
    t0 = time.time()

    feature_models = [m for m in models if getattr(m, "needs_features", False)]
    if feature_models:
        table = build_feature_table(conn, cfg)
        for m in feature_models:
            m.attach_features(table)

    # I5: closing odds are a benchmark only and never enter the stack
    members = [m for m in (ensemble_members or []) if m != "market_close"]
    result.ensemble_members = list(members)
    history = EnsembleHistory(members) if members else None
    labels = {int(mid): LABEL_MAP.get(str(res), -1)
              for mid, res in zip(matches["id"], matches["res"])}
    stack_cfg = cfg.stack or {}
    cal_cfg = cfg.calibrate or {}
    methods_used: dict[str, int] = {}

    for i, b in enumerate(blocks):
        train = matches[matches["date"] < b.start]
        test = matches[(matches["date"] >= max(b.start, start)) &
                       (matches["date"] <= min(b.end, end))]
        if len(test) == 0:
            continue
        result.blocks += 1
        result.matches += len(test)
        fixtures = test_as_fixtures(test)
        bench = test[[c for c in test.columns if c not in RESULT_COLUMNS]]
        ids = [int(x) for x in test["id"].tolist()]

        block_triples: dict[str, np.ndarray] = {}
        for model in models:
            try:
                model.fit(train, as_of=b.start)
                # only the closing-odds benchmark may see closing odds (I5)
                frame = bench if "odds_close" in getattr(model, "needs", set()) else fixtures
                P = model.predict(frame)
                save_oof(conn, run_id, model.name, b.start, ids, P, "ok")
                if model.name in members:
                    # pandas 3 hands back a read-only view, so the copy is needed
                    arr = P[["pH", "pD", "pA"]].to_numpy(dtype=float, copy=True)
                    if "status" in P.columns:
                        arr[P["status"].astype(str).to_numpy() != "ok"] = np.nan
                    block_triples[model.name] = arr
            except Exception as exc:  # I10: a failing model must not stop the run
                save_oof(conn, run_id, model.name, b.start, ids, None, f"failed:{type(exc).__name__}")
                result.failed[model.name] = result.failed.get(model.name, 0) + 1

        if history is not None and block_triples:
            block_matrix = np.full((len(ids), len(members), 3), np.nan)
            for k, name in enumerate(members):
                if name in block_triples:
                    block_matrix[:, k, :] = block_triples[name]

            X, row_blocks, y = history.matrix()
            stacker = EnsembleStacker(
                members,
                C=float(stack_cfg.get("C", 1.0)),
                min_rows=int(stack_cfg.get("min_rows", 800)),
                methods=list(stack_cfg.get("methods", ["logit", "logpool", "equal"])),
                recency_half_life_blocks=int(stack_cfg.get("recency_half_life_blocks", 40)),
            )
            stacker.fit(X, y, row_blocks, result.blocks)
            outcome = stacker.predict(block_matrix)
            save_oof(conn, run_id, "ensemble", b.start, ids, probs_frame(outcome.probs), "ok")
            save_diagnostics(conn, run_id, b.start, "stacker", outcome.scores, outcome.method)
            methods_used["stacker:" + outcome.method] = methods_used.get(
                "stacker:" + outcome.method, 0) + 1

            P_prior, cal_blocks, cal_y = history.ensemble_matrix()
            calibrator = EnsembleCalibrator(
                methods=list(cal_cfg.get("methods", ["none", "temperature", "dirichlet", "isotonic"])),
                min_rows=dict(cal_cfg.get("min_rows", {})),
                recency_half_life_blocks=int(stack_cfg.get("recency_half_life_blocks", 40)),
            )
            calibrator.fit(P_prior, cal_y, cal_blocks, result.blocks)
            cal = calibrator.predict(outcome.probs)
            save_oof(conn, run_id, "ensemble_cal", b.start, ids, probs_frame(cal.probs), "ok")
            save_diagnostics(conn, run_id, b.start, "calibrator", cal.scores, cal.method)
            methods_used["calibrator:" + cal.method] = methods_used.get(
                "calibrator:" + cal.method, 0) + 1

            history.add_block(ids, block_triples, result.blocks, labels, outcome.probs)

        conn.commit()
        if progress and (i % 20 == 0 or i == len(blocks) - 1):
            print(f"  block {i + 1}/{len(blocks)} {b.start} n={len(test)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    result.seconds = time.time() - t0
    result.ensemble = methods_used
    return result


def create_run(conn: sqlite3.Connection, run_id: str, kind: str, cfg: Config, notes: str = "") -> None:
    conn.execute(
        "INSERT INTO runs(run_id, created_at, config_hash, kind, notes) VALUES (?,?,?,?,?) "
        "ON CONFLICT(run_id) DO UPDATE SET created_at=excluded.created_at, "
        "config_hash=excluded.config_hash, kind=excluded.kind, notes=excluded.notes",
        (run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), cfg.config_hash, kind, notes),
    )
    conn.commit()


def oof_frame(conn: sqlite3.Connection, run_id: str, model: str | None = None) -> pd.DataFrame:
    sql = "SELECT model, match_id, block_start, pH, pD, pA, status FROM oof_predictions WHERE run_id = ?"
    params: list = [run_id]
    if model:
        sql += " AND model = ?"
        params.append(model)
    return pd.read_sql_query(sql, conn, params=params)


def arena_diagnostics(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Every ensemble candidate method with its mean score and selection count."""
    rows = pd.read_sql_query(
        "SELECT stage, method, score, selected FROM ensemble_diagnostics WHERE run_id = ?",
        conn, params=[run_id])
    if rows.empty:
        return []
    out = []
    for (stage, method), grp in rows.groupby(["stage", "method"]):
        out.append({
            "stage": str(stage),
            "method": str(method),
            "blocks": int(len(grp)),
            "mean_score": float(grp["score"].mean()),
            "selected_blocks": int(grp["selected"].sum()),
        })
    out.sort(key=lambda r: (r["stage"], r["mean_score"]))
    return out


def compare_models(conn: sqlite3.Connection, run_id: str, model_a: str,
                   model_b: str) -> dict:
    """Paired block bootstrap between two models of the same run."""
    from .bootstrap import block_bootstrap_ci, per_match_loss

    a = oof_frame(conn, run_id, model_a)
    b = oof_frame(conn, run_id, model_b)
    if a.empty or b.empty:
        return {"a": model_a, "b": model_b, "n": 0, "error": "no out-of-fold rows"}
    a = a[a["status"] == "ok"][["match_id", "block_start", "pH", "pD", "pA"]]
    b = b[b["status"] == "ok"][["match_id", "block_start", "pH", "pD", "pA"]]
    merged = a.merge(b, on="match_id", suffixes=("_a", "_b"))
    if merged.empty:
        return {"a": model_a, "b": model_b, "n": 0, "error": "no common matches"}

    matches = load_matches(conn)
    ymap = dict(zip(matches["id"], matches["res"]))
    y = np.array([{"H": 0, "D": 1, "A": 2}.get(str(ymap[int(m)]), -1)
                  for m in merged["match_id"]])
    keep = y >= 0
    merged = merged[keep].reset_index(drop=True)
    y = y[keep]

    Pa = merged[["pH_a", "pD_a", "pA_a"]].to_numpy(dtype=float)
    Pb = merged[["pH_b", "pD_b", "pA_b"]].to_numpy(dtype=float)
    block_col = "block_start_a" if "block_start_a" in merged.columns else "block_start"
    blocks, unique = np.unique(merged[block_col].to_numpy(), return_inverse=True)
    out = block_bootstrap_ci(per_match_loss(Pa, y), per_match_loss(Pb, y), unique)
    out.update({"a": model_a, "b": model_b, "n": int(len(y)),
                "blocks": int(len(blocks))})
    return out


def arena_table(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Per-model metrics over the run's out-of-fold predictions."""
    oof = oof_frame(conn, run_id)
    if oof.empty:
        return []
    matches = load_matches(conn)
    ymap = dict(zip(matches["id"], zip(matches["league"], matches["season"], matches["res"])))
    out = []
    for model, grp in oof.groupby("model"):
        ok = grp[grp["status"] == "ok"]
        if len(ok) == 0:
            out.append({"model": model, "n": 0, "log_loss": float("nan"), "status": "no predictions"})
            continue
        P = ok[["pH", "pD", "pA"]].to_numpy(dtype=float)
        y = np.array([{"H": 0, "D": 1, "A": 2}[ymap[int(m)][2]] for m in ok["match_id"]])
        ll = log_loss(P, y)
        entry = {
            "model": model,
            "n": len(ok),
            "log_loss": ll,
            "brier": brier(P, y),
            "rps": rps(P, y),
            "accuracy": accuracy(P, y),
            "ece": ece(P, y),
            "abstentions": int((grp["status"] != "ok").sum()),
        }
        out.append(entry)
    out.sort(key=lambda r: (np.isnan(r.get("log_loss", np.nan)), r.get("log_loss", 9e9)))
    base = next((r for r in out if r["model"] == "freq"), None)
    for r in out:
        if base and base.get("n"):
            r["skill_vs_freq"] = skill_score(r["log_loss"], base["log_loss"])
        else:
            r["skill_vs_freq"] = float("nan")
    return out
