"""Simulator and what-if (spec sections 12.1, 12.2 and 14.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..sim.season import current_season, remaining_pairs, simulate
from ..sim.whatif import SLIDER_LIMIT, apply
from .live import SYNTHETIC_ID_BASE, LiveService


def build_simulation(cfg: Config, service: LiveService, league: str,
                     n: int = 10000) -> dict:
    """Season simulation for a league: expected points and the title chances."""
    conn = service.connection()
    season = current_season(conn, league)
    if season is None:
        return {"league": league, "season": None, "n": int(n), "disabled": True,
                "reason": "no matches for this league yet", "teams": []}

    pairs = remaining_pairs(conn, league, season)
    if not pairs:
        return {"league": league, "season": season, "n": int(n), "disabled": True,
                "reason": "the season is complete - there is nothing left to simulate",
                "teams": []}

    frame = pd.DataFrame([
        {
            "id": SYNTHETIC_ID_BASE - i,
            "league": league,
            "season": season,
            "date": season,
            "kickoff_utc": None,
            "home_id": int(home),
            "away_id": int(away),
            "odds_open_h": np.nan, "odds_open_d": np.nan, "odds_open_a": np.nan,
        }
        for i, (home, away) in enumerate(pairs)
    ])

    prediction = service.predict(frame)
    if prediction is None:
        return {"league": league, "season": season, "n": int(n), "disabled": True,
                "reason": "no predictions available", "teams": []}

    triples = np.array([[row["pH"], row["pD"], row["pA"]]
                        for row in prediction["ensemble"]], dtype=float)
    grids = list(prediction["grids"])

    result = simulate(conn, league, season, n=int(n), triples=triples, grids=grids)
    names = service.names()
    payload = result.as_dict()
    for row in payload.get("teams", []):
        row["team"] = names.get(int(row["team_id"]), str(row["team_id"]))
    payload["remaining_fixtures"] = len(pairs)
    payload["note"] = ""
    return payload


def build_whatif(cfg: Config, service: LiveService, payload: dict) -> dict:
    """Adjusted H/D/A and grid for a manually nudged match. Nothing is stored."""
    from .predictions import _resolve_team

    league = str(payload.get("league") or "")
    if not league:
        return {"error": "league is required"}

    home_id = payload.get("home_id")
    away_id = payload.get("away_id")
    if home_id is None and payload.get("home") is not None:
        home_id = _resolve_team(service, payload["home"])
    if away_id is None and payload.get("away") is not None:
        away_id = _resolve_team(service, payload["away"])
    if home_id is None or away_id is None:
        return {"error": "home and away are required"}

    attack = float(payload.get("attack", 0.0) or 0.0)
    defence = float(payload.get("defence", 0.0) or 0.0)
    side = str(payload.get("side", "home") or "home").lower()
    if side not in ("home", "away"):
        side = "home"

    lambdas = service.lambdas(league, int(home_id), int(away_id))
    max_goals = int(cfg.dc.get("max_goals", 10))
    result = apply(lambdas, max_goals=max_goals, attack=attack, defence=defence,
                   side=side)
    names = service.names()
    result["league"] = league
    result["home_id"] = int(home_id)
    result["away_id"] = int(away_id)
    result["home"] = names.get(int(home_id), str(home_id))
    result["away"] = names.get(int(away_id), str(away_id))
    result["slider_limit"] = SLIDER_LIMIT
    result["stored"] = False
    return result
