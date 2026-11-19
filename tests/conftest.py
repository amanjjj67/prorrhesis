from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from prorrhesis.db import open_db
from prorrhesis.names import SOURCE_FOOTBALLDATA, add_alias, ensure_team

LEAGUE = "E0"
TEAMS = ["Alpha FC", "Beta FC", "Gamma FC", "Delta FC"]


def build_history(n_weeks: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2016, 8, 6)
    rows = []
    for w in range(n_weeks):
        d = start + timedelta(days=7 * w)
        pairs = [(0, 1), (2, 3), (1, 2), (3, 0)]
        for h, a in pairs:
            hg = int(rng.poisson(1.4))
            ag = int(rng.poisson(1.1))
            hst = int(rng.poisson(1.4))
            ast = int(rng.poisson(1.1))
            rows.append({
                "league": LEAGUE,
                "season": "1617" if d < date(2017, 6, 1) else "1718",
                "date": d.isoformat(),
                "home_id": h + 1,
                "away_id": a + 1,
                "hg": hg,
                "ag": ag,
                "res": "H" if hg > ag else "D" if hg == ag else "A",
                "hs": hst + int(rng.poisson(3)),
                "as_": ast + int(rng.poisson(3)),
                "hst": hst,
                "ast": ast,
                "odds_open_h": float(rng.uniform(1.4, 6.0)),
                "odds_open_d": float(rng.uniform(2.9, 4.2)),
                "odds_open_a": float(rng.uniform(1.4, 6.0)),
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def history() -> pd.DataFrame:
    return build_history()


@pytest.fixture()
def synthetic_db(tmp_path):
    conn = open_db(tmp_path / "syn.db")
    for i, name in enumerate(TEAMS, start=1):
        tid = ensure_team(conn, name.lower().replace(" ", "-"))
        add_alias(conn, name, SOURCE_FOOTBALLDATA, tid)
    conn.commit()
    yield conn
    conn.close()
