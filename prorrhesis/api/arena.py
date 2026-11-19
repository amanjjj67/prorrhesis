"""Arena and calibration (spec section 14.1).

The arena shows the models side by side over the out-of-fold rows of a run:
log loss with a bootstrap interval, the other metrics, and a paired comparison
against the closing-odds benchmark.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from ..config import Config
from ..eval.bootstrap import block_bootstrap_mean_ci, per_match_loss
from ..eval.holdout import holdout_broken, read_lock
from ..eval.metrics import accuracy, brier, ece, log_loss, rps
from ..eval.walkforward import compare_models, oof_frame
from ..explain import reliability_bins

BENCHMARK = "market_close"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def _pick_run(conn: sqlite3.Connection, run: str | None) -> str | None:
    if run:
        row = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (run,)).fetchone()
        if row:
            return str(row["run_id"])
        return None
    row = conn.execute(
        "SELECT run_id FROM runs WHERE kind IN ('headline','tuning') "
        "ORDER BY CASE kind WHEN 'headline' THEN 0 ELSE 1 END, created_at DESC LIMIT 1"
    ).fetchone()
    return str(row["run_id"]) if row else None


def _model_metrics(P: np.ndarray, y: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(P, y),
        "rps": rps(P, y),
        "brier": brier(P, y),
        "accuracy": accuracy(P, y),
        "ece": ece(P, y),
    }


def build_arena(cfg: Config, conn: sqlite3.Connection, run: str | None = None) -> dict:
    run_id = _pick_run(conn, run)
    if run_id is None:
        return {"run_id": None, "models": [], "note": "no evaluation run yet",
                "holdout_broken": holdout_broken(cfg, conn)}

    oof = oof_frame(conn, run_id)
    if oof.empty:
        return {"run_id": run_id, "models": [], "note": "this run has no predictions",
                "holdout_broken": holdout_broken(cfg, conn)}

    matches = pd.read_sql_query(
        "SELECT id, league, season, res FROM matches", conn)
    league_of = dict(zip(matches["id"], matches["league"]))
    season_of = dict(zip(matches["id"], matches["season"]))
    res_of = dict(zip(matches["id"], matches["res"]))

    models = []
    for model, grp in oof.groupby("model"):
        ok = grp[grp["status"] == "ok"]
        if len(ok) == 0:
            models.append({"model": model, "n": 0, "abstentions": int(len(grp)),
                           "log_loss": None, "note": "no predictions"})
            continue
        P = ok[["pH", "pD", "pA"]].to_numpy(dtype=float)
        y = np.array([LABEL_MAP.get(str(res_of[int(m)]), -1) for m in ok["match_id"]])
        keep = y >= 0
        P, y = P[keep], y[keep]
        blocks = ok["block_start"].to_numpy()[keep]
        unique, inverse = np.unique(blocks, return_inverse=True)

        entry = {"model": model, "n": int(len(y)), "blocks": int(len(unique)),
                 "abstentions": int((grp["status"] != "ok").sum())}
        entry.update(_model_metrics(P, y))
        ci = block_bootstrap_mean_ci(per_match_loss(P, y), inverse)
        entry["ci_low"] = ci["ci_low"]
        entry["ci_high"] = ci["ci_high"]
        models.append(entry)

    models.sort(key=lambda r: (r.get("log_loss") is None, r.get("log_loss", 9e9)))
    base = next((r for r in models if r["model"] == "freq"), None)
    for r in models:
        if base and base.get("n") and r.get("log_loss") is not None:
            r["skill_vs_freq"] = float(1.0 - r["log_loss"] / base["log_loss"])
        else:
            r["skill_vs_freq"] = None

    paired = None
    if any(r["model"] == BENCHMARK for r in models):
        target = "ensemble" if any(r["model"] == "ensemble" for r in models) else \
            next((r["model"] for r in models if r["model"] not in ("freq", "home")), None)
        if target:
            paired = compare_models(conn, run_id, target, BENCHMARK)

    return {
        "run_id": run_id,
        "run_kind": _kind(conn, run_id),
        "window": _window(conn, run_id),
        "models": models,
        "benchmark": BENCHMARK,
        "paired": paired,
        "per_league": _per_league(conn, run_id, league_of, season_of, res_of),
        "holdout_broken": holdout_broken(cfg, conn),
        "holdout_lock": read_lock(cfg),
        "note": "",
    }


def _kind(conn: sqlite3.Connection, run_id: str) -> str:
    row = conn.execute("SELECT kind FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return str(row["kind"]) if row else ""


def _window(conn: sqlite3.Connection, run_id: str) -> dict:
    row = conn.execute("SELECT notes, created_at FROM runs WHERE run_id = ?",
                       (run_id,)).fetchone()
    if row is None:
        return {}
    parts = str(row["notes"] or "").split("..")
    if len(parts) == 2:
        return {"start": parts[0], "end": parts[1]}
    return {"created_at": str(row["created_at"] or "")}


def _per_league(conn: sqlite3.Connection, run_id: str, league_of: dict,
                season_of: dict, res_of: dict) -> list[dict]:
    """The same table, split by league."""
    oof = oof_frame(conn, run_id)
    if oof.empty:
        return []
    out = []
    for model, grp in oof.groupby("model"):
        ok = grp[grp["status"] == "ok"]
        if len(ok) == 0:
            continue
        P = ok[["pH", "pD", "pA"]].to_numpy(dtype=float)
        y = np.array([LABEL_MAP.get(str(res_of[int(m)]), -1) for m in ok["match_id"]])
        keep = y >= 0
        P, y = P[keep], y[keep]
        leagues = ok["match_id"].map(lambda m: league_of.get(int(m), ""))
        for league in sorted(set(leagues)):
            sel = (leagues == league).to_numpy()
            if sel.sum() < 30:
                continue
            entry = {"model": model, "league": league, "n": int(sel.sum())}
            entry.update(_model_metrics(P[sel], y[sel]))
            out.append(entry)
    out.sort(key=lambda r: (r["league"], r["log_loss"]))
    return out


def build_time_machine(conn: sqlite3.Connection, date: str,
                       run: str | None = None) -> dict:
    """Replay a past matchweek: the models as they were, and what happened.

    The rows come from the out-of-fold table of the latest run, so the numbers
    are exactly the ones the arena measured - not a fresh fit.
    """
    run_id = _pick_run(conn, run)
    if run_id is None:
        return {"run_id": None, "date": date, "matches": [],
                "note": "no evaluation run yet"}

    oof = oof_frame(conn, run_id, "ensemble")
    if oof.empty:
        oof = oof_frame(conn, run_id, "market_close")
    if oof.empty:
        return {"run_id": run_id, "date": date, "matches": [],
                "note": "this run has no predictions"}

    matches = pd.read_sql_query(
        "SELECT id, league, date, home_id, away_id, hg, ag, res FROM matches WHERE date = ?",
        conn, params=[date])
    if matches.empty:
        return {"run_id": run_id, "date": date, "matches": [],
                "note": "no matches on that date"}

    names = _names(conn)
    rows = []
    for record in matches.to_dict("records"):
        hit = oof[oof["match_id"] == int(record["id"])]
        if hit.empty:
            continue
        hit = hit[hit["status"] == "ok"]
        if hit.empty:
            continue
        triple = hit.iloc[0]
        idx = LABEL_MAP.get(str(record["res"]), -1)
        loss = None
        if idx >= 0:
            p = (float(triple["pH"]), float(triple["pD"]), float(triple["pA"]))[idx]
            loss = float(-np.log(min(max(p, 1e-15), 1.0)))
        rows.append({
            "match_id": int(record["id"]),
            "league": str(record["league"]),
            "home": names.get(int(record["home_id"]), str(record["home_id"])),
            "away": names.get(int(record["away_id"]), str(record["away_id"])),
            "pH": float(triple["pH"]),
            "pD": float(triple["pD"]),
            "pA": float(triple["pA"]),
            "result": str(record["res"]),
            "score": f"{int(record['hg'])}-{int(record['ag'])}",
            "loss": loss,
        })
    return {"run_id": run_id, "date": date, "matches": rows, "note": ""}


def _names(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute(
        "SELECT t.id AS id, "
        "(SELECT alias FROM team_aliases a WHERE a.team_id = t.id "
        " AND a.source = 'footballdata' ORDER BY a.alias LIMIT 1) AS alias, "
        "t.canon AS canon FROM teams t"
    ).fetchall()
    return {int(r["id"]): str(r["alias"] or r["canon"]).replace("-", " ").title()
            for r in rows}


def build_calibration(conn: sqlite3.Connection, model: str = "ensemble_cal",
                      run: str | None = None) -> dict:
    """Reliability bins per outcome for one model of a run."""
    run_id = _pick_run(conn, run)
    if run_id is None:
        return {"run_id": None, "model": model, "outcomes": [],
                "note": "no evaluation run yet"}

    oof = oof_frame(conn, run_id, model)
    if oof.empty:
        return {"run_id": run_id, "model": model, "outcomes": [],
                "note": "this model has no predictions in this run"}

    matches = pd.read_sql_query("SELECT id, res FROM matches", conn)
    res_of = dict(zip(matches["id"], matches["res"]))
    ok = oof[oof["status"] == "ok"]
    y = np.array([LABEL_MAP.get(str(res_of[int(m)]), -1) for m in ok["match_id"]])
    keep = y >= 0
    P = ok[["pH", "pD", "pA"]].to_numpy(dtype=float)[keep]
    y = y[keep]

    outcomes = []
    for k, key in enumerate(("H", "D", "A")):
        predicted = P[:, k]
        observed = (y == k).astype(float)
        outcomes.append({
            "outcome": key,
            "bins": reliability_bins(predicted, observed),
            "ece": ece(P[:, [k]], np.where(y == k, 0, 1)),
            "n": int(len(y)),
        })
    return {
        "run_id": run_id,
        "model": model,
        "outcomes": outcomes,
        "note": "",
    }
