"""Hyperparameter tuning on the tuning window (spec section 10.5).

Tuning happens on the tuning window only; the headline window stays frozen.
The scorer here keeps predictions in memory so a grid search does not fill
the oof_predictions table with throwaway rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..models.dixoncoles import DixonColesModel
from .walkforward import RESULT_COLUMNS, test_as_fixtures, week_blocks

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def walk_forward_scores(
    models: list,
    matches: pd.DataFrame,
    start: str,
    end: str,
    block_days: int = 7,
) -> dict[str, tuple[float, int]]:
    """In-memory walk-forward log loss per model; returns {name: (log_loss, n)}."""
    blocks = week_blocks(start, end, block_days)
    losses: dict[str, list[np.ndarray]] = {m.name: [] for m in models}

    for b in blocks:
        train = matches[matches["date"] < b.start]
        test = matches[(matches["date"] >= b.start) & (matches["date"] <= b.end)]
        if len(test) == 0:
            continue
        fixtures = test_as_fixtures(test)
        bench = test[[c for c in test.columns if c not in RESULT_COLUMNS]]
        y = test["res"].map(OUTCOME_INDEX).to_numpy(dtype=int)

        for model in models:
            try:
                model.fit(train, as_of=b.start)
                frame = bench if "odds_close" in getattr(model, "needs", set()) else fixtures
                P = model.predict(frame)
                probs = P[["pH", "pD", "pA"]].to_numpy(dtype=float)
                if "status" in P.columns:
                    keep = (P["status"].astype(str) == "ok").to_numpy()
                else:
                    keep = np.ones(len(P), dtype=bool)
                if not keep.any():
                    continue
                ll = -np.log(np.clip(probs[np.arange(len(P)), y], 1e-12, 1.0))
                losses[model.name].append(ll[keep])
            except Exception:
                continue

    out = {}
    for name, chunks in losses.items():
        if not chunks:
            out[name] = (float("nan"), 0)
            continue
        stacked = np.concatenate(chunks)
        out[name] = (float(stacked.mean()), int(len(stacked)))
    return out


def tune_dc(
    cfg: Config,
    matches: pd.DataFrame,
    start: str,
    end: str,
    half_lives: list[float] | None = None,
    gammas: list[float] | None = None,
    windows: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """Coordinate search over the Dixon-Coles hyperparameters the spec says to tune.

    half_life_days is searched in the spec's 120-700 day range and gamma around
    its 0.5 default; window_seasons is searched last.
    """
    half_lives = half_lives or [180.0, 365.0, 550.0, 700.0]
    gammas = gammas or [0.1, 0.5, 1.0, 2.0]
    windows = windows or [2, 3, 4]
    block_days = int(cfg.eval.get("block_days", 7))

    trials: list[dict] = []

    def score(hl: float, gamma: float, window: int) -> float:
        model = DixonColesModel(window_seasons=window, half_life_days=hl, gamma=gamma)
        ll, n = walk_forward_scores([model], matches, start, end, block_days)[model.name]
        trials.append({"half_life_days": hl, "gamma": gamma, "window_seasons": window,
                       "log_loss": ll, "n": n})
        if verbose:
            print(f"  dc hl={hl:.0f} gamma={gamma} window={window} -> logloss {ll:.4f} (n={n})",
                  flush=True)
        return ll

    best_hl, best_gamma, best_window = 365.0, 0.5, 3
    best_ll = score(best_hl, best_gamma, best_window)

    for hl in half_lives:
        ll = score(hl, best_gamma, best_window)
        if ll < best_ll:
            best_ll, best_hl = ll, hl
    for gamma in gammas:
        ll = score(best_hl, gamma, best_window)
        if ll < best_ll:
            best_ll, best_gamma = ll, gamma
    for window in windows:
        ll = score(best_hl, best_gamma, window)
        if ll < best_ll:
            best_ll, best_window = ll, window

    return {
        "half_life_days": float(best_hl),
        "gamma": float(best_gamma),
        "window_seasons": int(best_window),
        "log_loss": float(best_ll),
        "trials": trials,
    }


def tuning_path(cfg: Config) -> Path:
    return cfg.data_dir / "tuning" / "dc.json"


def load_dc_tuning(cfg: Config) -> dict:
    p = tuning_path(cfg)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_dc_tuning(cfg: Config, data: dict) -> Path:
    p = tuning_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return p
