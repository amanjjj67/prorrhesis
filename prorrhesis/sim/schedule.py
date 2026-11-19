"""Estimated remaining schedule.

The fixtures feed of football-data.co.uk does not always cover the five
leagues, so the remaining fixtures are derived from the season itself: in
every one of the leagues each ordered pair (home, away) plays exactly once per
season. The round-robin draw is rebuilt with the circle method, the pairs that
have already been played are removed, and the rest is spread over the
matchdays that are still to come.

The dates of the remaining matchdays are estimates: the average gap between
the played matchdays is continued. The screen labels them as estimates.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pandas as pd


def round_robin_rounds(teams: list[int]) -> list[list[tuple[int, int]]]:
    """Circle method: n-1 rounds in which every ordered pair meets once."""
    n = len(teams)
    if n < 2:
        return []
    arr = list(teams)
    half = n // 2
    rounds: list[list[tuple[int, int]]] = []
    for r in range(n - 1):
        pairs = []
        for i in range(half):
            a, b = arr[i], arr[n - 1 - i]
            pairs.append((b, a) if r % 2 else (a, b))
        rounds.append(pairs)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]
    return rounds


def played_matchdays(conn: sqlite3.Connection, league: str, season: str) -> list[str]:
    """The matchday dates of the season, oldest first."""
    rows = conn.execute(
        "SELECT DISTINCT date FROM matches WHERE league = ? AND season = ? ORDER BY date",
        (league, season)).fetchall()
    return [str(r["date"]) for r in rows]


def remaining_schedule(conn: sqlite3.Connection, league: str, season: str) -> list[dict]:
    """The matches still to be played, with estimated dates."""
    from .season import played_pairs, season_teams

    teams = season_teams(conn, league, season)
    if len(teams) < 2:
        return []
    done = played_pairs(conn, league, season)
    rounds = round_robin_rounds(teams)

    matchdays = played_matchdays(conn, league, season)
    if len(matchdays) >= 2:
        gaps = [(pd.Timestamp(b) - pd.Timestamp(a)).days
                for a, b in zip(matchdays, matchdays[1:]) if b > a]
        step = max(1, int(round(sum(gaps) / len(gaps)))) if gaps else 7
        last = pd.Timestamp(matchdays[-1])
    else:
        step = 7
        last = pd.Timestamp(matchdays[-1]) if matchdays else pd.Timestamp("2024-08-01")

    out: list[dict] = []
    played_rounds = len(matchdays)
    for r, pairs in enumerate(rounds):
        left = [(h, a) for h, a in pairs if (h, a) not in done]
        if not left:
            continue
        matchday = played_rounds + (r - played_rounds) + 1
        date = (last + timedelta(days=step * (r - played_rounds + 1))).strftime("%Y-%m-%d")
        for home, away in left:
            out.append({
                "league": league,
                "season": season,
                "date": date,
                "home_id": int(home),
                "away_id": int(away),
                "matchday": matchday,
                "estimated": True,
            })
    return out
