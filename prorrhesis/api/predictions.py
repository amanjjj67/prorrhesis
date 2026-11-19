"""Fixtures and match detail (spec section 14.1)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import Config
from ..explain import disagreement, dc_reason
from .live import LiveService


def top_scoreline(grid: np.ndarray | None) -> dict | None:
    """The most likely scoreline of a Dixon-Coles grid."""
    if grid is None or not len(grid):
        return None
    flat = int(np.argmax(grid))
    d = grid.shape[0] - 1
    home_goals, away_goals = divmod(flat, d + 1)
    return {"home_goals": int(home_goals), "away_goals": int(away_goals),
            "probability": float(grid[home_goals, away_goals])}


def _favourite(triple: dict) -> dict:
    best = max(("H", "D", "A"), key=lambda k: triple.get(f"p{k}", 0.0))
    return {"outcome": best, "probability": float(triple.get(f"p{best}", 0.0))}


def build_fixtures(cfg: Config, service: LiveService, league: str,
                   days: int = 7) -> dict:
    """Upcoming fixtures with the ensemble triple, top scoreline and disagreement."""
    fixtures = service.fixtures(league, days)
    if len(fixtures) == 0:
        return {"league": league, "days": int(days), "as_of": _today(),
                "fixtures": [], "count": 0, "note": "no upcoming fixtures found"}

    result = service.predict(fixtures)
    names = service.names()
    rows = []
    for i, (_, fixture) in enumerate(fixtures.iterrows()):
        triple = result["ensemble"][i] if result else {}
        model_triples = {}
        for name, values in (result["models"] if result else {}).items():
            model_triples[name] = values[i]
        flag = disagreement(model_triples)
        grid = result["grids"][i] if result else None
        rows.append({
            "id": int(fixture["id"]),
            "date": str(fixture["date"]),
            "kickoff_utc": fixture.get("kickoff_utc"),
            "home_id": int(fixture["home_id"]),
            "away_id": int(fixture["away_id"]),
            "home": names.get(int(fixture["home_id"]), str(fixture["home_id"])),
            "away": names.get(int(fixture["away_id"]), str(fixture["away_id"])),
            "estimated": bool(fixture.get("estimated", False)),
            "pH": triple.get("pH"),
            "pD": triple.get("pD"),
            "pA": triple.get("pA"),
            "status": triple.get("status", "abstain"),
            "favourite": _favourite(triple) if triple else None,
            "top_scoreline": top_scoreline(grid),
            "disagree": bool(flag),
            "disagreement": flag,
            "odds_open": _open_odds(fixture),
        })
    return {
        "league": league,
        "days": int(days),
        "as_of": _today(),
        "method": result.get("method") if result else None,
        "calibration_method": result.get("calibration_method") if result else None,
        "fixtures": rows,
        "count": len(rows),
        "note": "",
    }


def _open_odds(fixture: pd.Series) -> dict | None:
    h = fixture.get("odds_open_h")
    d = fixture.get("odds_open_d")
    a = fixture.get("odds_open_a")
    try:
        vals = [float(h), float(d), float(a)]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) and v > 0 for v in vals):
        return None
    return {"h": vals[0], "d": vals[1], "a": vals[2]}


def _resolve_team(service: LiveService, value: str) -> int | None:
    """A team id or a team name."""
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    names = service.names()
    lowered = text.lower()
    for team_id, name in names.items():
        if name.lower() == lowered:
            return team_id
    for team_id, name in names.items():
        if lowered in name.lower():
            return team_id
    return None


def build_match(cfg: Config, service: LiveService, league: str, date: str,
                home: str, away: str) -> dict:
    """Every model's triple for one match, with the grid, reasons and uncertainty."""
    home_id = _resolve_team(service, home)
    away_id = _resolve_team(service, away)
    if home_id is None or away_id is None:
        return {"error": "unknown team", "league": league, "date": date,
                "home": home, "away": away}

    frame = pd.DataFrame([{
        "id": -1,
        "league": league,
        "season": _season_of(service, league),
        "date": date,
        "kickoff_utc": None,
        "home_id": home_id,
        "away_id": away_id,
        "odds_open_h": np.nan, "odds_open_d": np.nan, "odds_open_a": np.nan,
    }])
    result = service.predict(frame)
    names = service.names()

    models = {}
    for name, values in (result["models"] if result else {}).items():
        models[name] = values[0]
    triple = result["ensemble"][0] if result else {}
    cal = result["ensemble_cal"][0] if result else {}
    grid = result["grids"][0] if result else None

    lambdas = service.lambdas(league, home_id, away_id)
    reasons = [dc_reason(lambdas, service.league_mean_goals(league))]

    flag = disagreement(models)
    if flag:
        reasons.append(flag["text"])

    uncertain = service.uncertain_teams(league)
    uncertainty = None
    if home_id in uncertain or away_id in uncertain:
        thin = [names.get(t, str(t)) for t in (home_id, away_id) if t in uncertain]
        uncertainty = (f"Little data for {', '.join(thin)} - the estimate is "
                       f"mostly the league average.")

    return {
        "league": league,
        "date": date,
        "home_id": home_id,
        "away_id": away_id,
        "home": names.get(home_id, str(home_id)),
        "away": names.get(away_id, str(away_id)),
        "models": models,
        "ensemble": triple,
        "ensemble_cal": cal,
        "method": result.get("method") if result else None,
        "calibration_method": result.get("calibration_method") if result else None,
        "grid": grid.tolist() if grid is not None else None,
        "top_scoreline": top_scoreline(grid),
        "market_open": _market_open(models),
        "reasons": reasons,
        "disagreement": flag,
        "uncertainty": uncertainty,
        "lambdas": [float(lambdas[0]), float(lambdas[1]), float(lambdas[2])],
    }


def _market_open(models: dict) -> dict | None:
    entry = models.get("market_open")
    if not entry or entry.get("status") != "ok":
        return None
    return {"pH": entry["pH"], "pD": entry["pD"], "pA": entry["pA"]}


def _season_of(service: LiveService, league: str) -> str:
    from ..sim.season import current_season

    conn = service.connection()
    season = current_season(conn, league)
    return season or ""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
