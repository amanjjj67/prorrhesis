"""Season simulation (spec section 12.1).

The remaining fixtures of a season need no schedule feed: in each of the five
leagues every ordered pair (home, away) plays exactly once per season. So the
missing fixtures are simply "all ordered pairs of this season's clubs minus the
pairs that have already been played".

For every remaining fixture the ensemble gives H/D/A and the Dixon-Coles grid
gives the scoreline distribution. Each simulation draws the outcome first and
then a scoreline from the part of the grid that matches that outcome.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY = 0, 1, 2


@dataclass
class SimulationResult:
    league: str
    season: str
    n: int
    teams: list[int] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    disabled: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "league": self.league,
            "season": self.season,
            "n": self.n,
            "disabled": self.disabled,
            "reason": self.reason,
            "teams": self.rows,
        }


def current_season(conn: sqlite3.Connection, league: str) -> str | None:
    row = conn.execute(
        "SELECT season FROM matches WHERE league = ? ORDER BY date DESC LIMIT 1",
        (league,)).fetchone()
    return str(row["season"]) if row else None


def season_teams(conn: sqlite3.Connection, league: str, season: str) -> list[int]:
    """Clubs that have played at least once in the season."""
    rows = conn.execute(
        "SELECT home_id FROM matches WHERE league = ? AND season = ? "
        "UNION SELECT away_id FROM matches WHERE league = ? AND season = ?",
        (league, season, league, season)).fetchall()
    return sorted(int(r["home_id"]) for r in rows)


def league_clubs(conn: sqlite3.Connection, league: str) -> list[int]:
    """Every club that has ever appeared in the league."""
    rows = conn.execute(
        "SELECT home_id FROM matches WHERE league = ? "
        "UNION SELECT away_id FROM matches WHERE league = ?",
        (league, league)).fetchall()
    return sorted(int(r["home_id"]) for r in rows)


def played_pairs(conn: sqlite3.Connection, league: str, season: str) -> set[tuple[int, int]]:
    rows = conn.execute(
        "SELECT home_id, away_id FROM matches WHERE league = ? AND season = ?",
        (league, season)).fetchall()
    return {(int(r["home_id"]), int(r["away_id"])) for r in rows}


def remaining_pairs(conn: sqlite3.Connection, league: str, season: str) -> list[tuple[int, int]]:
    """Ordered pairs of this season's clubs that have not played each other yet."""
    teams = season_teams(conn, league, season)
    done = played_pairs(conn, league, season)
    out = [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]
    return out


def _conditional_grids(grid: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each outcome: the cumulative distribution and the scoreline cells."""
    d = grid.shape[0] - 1
    ii, jj = np.meshgrid(np.arange(d + 1), np.arange(d + 1), indexing="ij")
    masks = [ii > jj, ii == jj, ii < jj]
    out = []
    for mask in masks:
        cells = grid[mask]
        flat = cells / cells.sum()
        coords = np.column_stack([ii[mask], jj[mask]])
        out.append((np.cumsum(flat), coords))
    return out


def season_roster_size(conn: sqlite3.Connection, league: str, season: str) -> int | None:
    """How many clubs the league had in the season before `season`."""
    row = conn.execute(
        "SELECT season FROM matches WHERE league = ? AND season < ? "
        "ORDER BY season DESC LIMIT 1", (league, season)).fetchone()
    if row is None:
        return None
    previous = str(row["season"])
    count = conn.execute(
        "SELECT COUNT(*) c FROM (SELECT home_id FROM matches WHERE league = ? AND season = ? "
        "UNION SELECT away_id FROM matches WHERE league = ? AND season = ?)",
        (league, previous, league, previous)).fetchone()["c"]
    return int(count)


def current_table(conn: sqlite3.Connection, league: str, season: str) -> dict[int, dict]:
    """Points, goal difference and goals for of the matches already played."""
    rows = conn.execute(
        "SELECT home_id, away_id, hg, ag FROM matches WHERE league = ? AND season = ?",
        (league, season)).fetchall()
    table: dict[int, dict] = {}

    def slot(team: int) -> dict:
        return table.setdefault(int(team), {"points": 0.0, "gd": 0.0, "gf": 0.0})

    for r in rows:
        home, away = int(r["home_id"]), int(r["away_id"])
        hg, ag = int(r["hg"]), int(r["ag"])
        hs, as_ = slot(home), slot(away)
        hs["gf"] += hg
        as_["gf"] += ag
        hs["gd"] += hg - ag
        as_["gd"] += ag - hg
        if hg > ag:
            hs["points"] += 3
        elif hg < ag:
            as_["points"] += 3
        else:
            hs["points"] += 1
            as_["points"] += 1
    return table


def simulate(
    conn: sqlite3.Connection,
    league: str,
    season: str | None = None,
    n: int = 10000,
    triples: np.ndarray | None = None,
    grids: list[np.ndarray] | None = None,
    teams: list[int] | None = None,
    relegation: int = 3,
    top: int = 4,
    seed: int = 42,
) -> SimulationResult:
    """Run the season simulation.

    `triples` holds the ensemble H/D/A for each remaining fixture and `grids`
    the matching Dixon-Coles grids. Both are supplied by the caller so the
    simulation can be tested without fitting any model.
    """
    season = season or current_season(conn, league)
    if season is None:
        return SimulationResult(league, "", n, disabled=True,
                                reason="no matches for this league yet")

    season_clubs = season_teams(conn, league, season)
    expected = season_roster_size(conn, league, season)
    if expected is not None and len(season_clubs) < expected:
        return SimulationResult(league, season, n, disabled=True,
                                reason="every club must have played at least once")

    pairs = remaining_pairs(conn, league, season)
    team_list = teams if teams is not None else season_clubs
    if triples is None or grids is None:
        return SimulationResult(league, season, n, teams=team_list, disabled=True,
                                reason="no predictions available")

    n_fixtures = len(pairs)
    m = len(pairs)

    rng = np.random.default_rng(seed)
    P = np.asarray(triples, dtype=float)

    index = {t: i for i, t in enumerate(team_list)}
    n_teams = len(team_list)
    played = current_table(conn, league, season)

    pts = np.zeros((n, n_teams))
    gd = np.zeros((n, n_teams))
    gf = np.zeros((n, n_teams))
    for i, team in enumerate(team_list):
        row = played.get(int(team))
        if row:
            pts[:, i] = row["points"]
            gd[:, i] = row["gd"]
            gf[:, i] = row["gf"]

    if n_fixtures:
        u = rng.random((n, m))
        outcomes = (u > P[None, :, 0]).astype(int) + (u > (P[None, :, 0] + P[None, :, 1])).astype(int)

        goals_h = np.zeros((n, m), dtype=int)
        goals_a = np.zeros((n, m), dtype=int)
        for j in range(m):
            cond = _conditional_grids(grids[j])
            v = rng.random(n)
            out_j = outcomes[:, j]
            home_goals = np.zeros(n, dtype=int)
            away_goals = np.zeros(n, dtype=int)
            for k in range(3):
                sel = out_j == k
                if not sel.any():
                    continue
                cdf, coords = cond[k]
                pos = np.searchsorted(cdf, v[sel], side="right")
                pos = np.clip(pos, 0, len(cdf) - 1)
                home_goals[sel] = coords[pos, 0]
                away_goals[sel] = coords[pos, 1]
            goals_h[:, j] = home_goals
            goals_a[:, j] = away_goals

        for j, (home, away) in enumerate(pairs):
            hi, ai = index[home], index[away]
            out_j = outcomes[:, j]
            home_pts = np.where(out_j == OUTCOME_HOME, 3.0,
                                np.where(out_j == OUTCOME_DRAW, 1.0, 0.0))
            away_pts = np.where(out_j == OUTCOME_AWAY, 3.0,
                                np.where(out_j == OUTCOME_DRAW, 1.0, 0.0))
            pts[:, hi] += home_pts
            pts[:, ai] += away_pts
            gf[:, hi] += goals_h[:, j]
            gf[:, ai] += goals_a[:, j]
            gd[:, hi] += goals_h[:, j] - goals_a[:, j]
            gd[:, ai] += goals_a[:, j] - goals_h[:, j]

    tie = rng.random((n, n_teams))
    order = np.lexsort((tie, gf, gd, pts), axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.tile(np.arange(n_teams), (n, 1)), axis=1)

    rows = []
    for i, team in enumerate(team_list):
        rows.append({
            "team_id": int(team),
            "expected_points": float(pts[:, i].mean()),
            "points_now": float(played.get(int(team), {}).get("points", 0.0)),
            "p_title": float((ranks[:, i] == 0).mean()),
            "p_top4": float((ranks[:, i] < top).mean()),
            "p_top6": float((ranks[:, i] < 6).mean()),
            "p_relegation": float((ranks[:, i] >= n_teams - relegation).mean()),
        })
    rows.sort(key=lambda r: -r["expected_points"])
    return SimulationResult(league, season, n, teams=team_list, rows=rows)
