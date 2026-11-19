"""The ledger screen (spec section 13 and 14.1)."""

from __future__ import annotations

import sqlite3

import numpy as np

from ..eval.bootstrap import block_bootstrap_mean_ci
from ..ledger import rows as ledger_rows
from ..ledger import verify

LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def _loss(pH: float, pD: float, pA: float, res: str) -> float | None:
    idx = LABEL_MAP.get(str(res).upper())
    if idx is None:
        return None
    p = (pH, pD, pA)[idx]
    if p is None or not np.isfinite(p) or p <= 0:
        return None
    return float(-np.log(min(max(p, 1e-15), 1.0)))


def build_ledger(conn: sqlite3.Connection, limit: int = 200) -> dict:
    """Recent ledger rows, the chain state and the running log loss."""
    chain = verify(conn)
    rows = ledger_rows(conn, limit=limit)
    for row in rows:
        row["loss"] = (_loss(row["pH"], row["pD"], row["pA"], row["result"])
                       if row.get("result") else None)

    settled = [r for r in rows if r.get("result")]
    summary = _running_log_loss(conn)
    return {
        "rows": rows,
        "count": len(rows),
        "chain_ok": bool(chain.get("ok")),
        "chain": chain,
        "settled": len(settled),
        "open": len(rows) - len(settled),
        "running_log_loss": summary,
        "note": "",
    }


def _running_log_loss(conn: sqlite3.Connection, threshold: int = 100) -> dict:
    """Rolling log loss of the live predictions against the closing-odds benchmark."""
    rows = conn.execute(
        "SELECT p.model, p.pH, p.pD, p.pA, p.date, s.res, "
        "m.odds_close_h, m.odds_close_d, m.odds_close_a "
        "FROM predictions p "
        "JOIN matches m ON m.league = p.league AND m.date = p.date "
        " AND m.home_id = p.home_id AND m.away_id = p.away_id "
        "JOIN settlements s ON s.league = p.league AND s.date = p.date "
        " AND s.home_id = p.home_id AND s.away_id = p.away_id"
    ).fetchall()

    def collect(models: tuple[str, ...]) -> list[float]:
        out = []
        for r in rows:
            if str(r["model"]) not in models:
                continue
            loss = _loss(r["pH"], r["pD"], r["pA"], r["res"])
            if loss is not None:
                out.append(loss)
        return out

    def summarise(losses: list[float]) -> dict:
        if len(losses) == 0:
            return {"n": 0, "log_loss": None, "ci_low": None, "ci_high": None,
                    "judgement": "too few to judge"}
        ci = block_bootstrap_mean_ci(np.asarray(losses), np.arange(len(losses)))
        out = {"n": len(losses), "log_loss": float(np.mean(losses)),
               "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
        out["judgement"] = ("too few to judge" if len(losses) < threshold
                            else "measured")
        return out

    ensemble = summarise(collect(("ensemble", "ensemble_cal")))
    benchmark = summarise(collect(("market_close",)))
    return {
        "ensemble": ensemble,
        "market_close": benchmark,
        "threshold": threshold,
        "note": "",
    }
