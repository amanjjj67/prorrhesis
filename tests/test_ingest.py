from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from prorrhesis.db import open_db
from prorrhesis.ingest import footballdata as fd
from prorrhesis.names import SOURCE_FOOTBALLDATA, add_alias, ensure_team

HEADER = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,"
    "B365H,B365D,B365A,PSH,PSD,PSA,AvgH,AvgD,AvgA,MaxH,MaxD,MaxA,"
    "PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA,B365CH,B365CD,B365CA"
)


def write_csv(path: Path, rows: list[str], encoding: str = "utf-8") -> Path:
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding=encoding)
    return path


ROW_FULL = ("E0,16/08/24,15:00,Alpha FC,Beta FC,2,1,H,1,0,10,8,5,3,10,9,4,3,1,2,0,0,"
            "1.80,3.50,4.50,1.85,3.40,4.40,1.90,3.45,4.30,2.00,3.60,4.60,"
            "1.70,3.30,4.80,1.75,3.35,4.70,1.72,3.33,4.75")


def test_parses_both_date_formats(tmp_path):
    p = write_csv(tmp_path / "a.csv", [ROW_FULL, ROW_FULL.replace("16/08/24", "16/08/2024")])
    frame, report = fd.parse_season_file(p, "E0", "2425")
    assert report.rows_kept == 2
    assert set(frame["date"]) == {"2024-08-16"}
    assert frame["hg"].tolist() == [2, 2]
    assert frame["res"].tolist() == ["H", "H"]


def test_latin1_fallback(tmp_path):
    path = tmp_path / "b.csv"
    path.write_bytes((HEADER + "\n" + ROW_FULL + "\n").encode("latin-1"))
    frame, report = fd.parse_season_file(path, "E0", "2425")
    assert report.rows_kept == 1


def test_missing_columns_tolerated(tmp_path):
    short = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
             "E0,16/08/24,Alpha FC,Beta FC,2,1,H\n")
    p = tmp_path / "c.csv"
    p.write_text(short, encoding="utf-8")
    frame, report = fd.parse_season_file(p, "E0", "2425")
    assert report.rows_kept == 1
    assert frame["odds_open_h"].isna().all()
    assert frame["kickoff_utc"].isna().all()
    assert pd.isna(frame["hs"].iloc[0])


def test_unplayed_rows_dropped(tmp_path):
    row = ROW_FULL.replace(",2,1,H,", ",,,,")
    p = write_csv(tmp_path / "d.csv", [row])
    frame, report = fd.parse_season_file(p, "E0", "2425")
    assert report.dropped_unplayed == 1
    assert report.rows_kept == 0


def test_odds_chains_pick_first_complete_family(tmp_path):
    # only B365 open odds present -> open source must be B365, close source must be B365C
    row = ("," * 0)
    parts = ROW_FULL.split(",")
    # blank out Avg*, PS* and PSC* / AvgC* leaving B365 families
    for idx in list(range(25, 31)) + list(range(34, 40)):
        parts[idx] = ""
    p = write_csv(tmp_path / "e.csv", [",".join(parts)])
    frame, report = fd.parse_season_file(p, "E0", "2425")
    assert frame["odds_open_src"].iloc[0] == "B365H"
    assert frame["odds_close_src"].iloc[0] == "B365CH"
    assert frame["odds_open_h"].iloc[0] == pytest.approx(1.80)


def test_res_disagreement_trusts_goals(tmp_path):
    row = ROW_FULL.replace(",2,1,H,", ",3,1,D,")
    p = write_csv(tmp_path / "f.csv", [row])
    frame, report = fd.parse_season_file(p, "E0", "2425")
    assert report.res_disagreements == 1
    assert frame["res"].iloc[0] == "H"


def test_upsert_is_idempotent(tmp_path):
    conn = open_db(tmp_path / "i.db")
    a = ensure_team(conn, "alpha-fc")
    b = ensure_team(conn, "beta-fc")
    add_alias(conn, "Alpha FC", SOURCE_FOOTBALLDATA, a)
    add_alias(conn, "Beta FC", SOURCE_FOOTBALLDATA, b)
    conn.commit()

    p = write_csv(tmp_path / "g.csv", [ROW_FULL])
    frame, _ = fd.parse_season_file(p, "E0", "2425")
    first = fd.upsert_matches(conn, frame)
    second = fd.upsert_matches(conn, frame)
    count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    assert first["inserted"] == 1
    assert second["inserted"] == 0
    assert count == 1
    conn.close()


def test_unresolved_name_rejects_only_its_rows(tmp_path):
    conn = open_db(tmp_path / "j.db")
    a = ensure_team(conn, "alpha-fc")
    b = ensure_team(conn, "beta-fc")
    add_alias(conn, "Alpha FC", SOURCE_FOOTBALLDATA, a)
    add_alias(conn, "Beta FC", SOURCE_FOOTBALLDATA, b)
    conn.commit()

    other = ROW_FULL.replace("Beta FC", "Unknown FC")
    p = write_csv(tmp_path / "h.csv", [ROW_FULL, other])
    frame, _ = fd.parse_season_file(p, "E0", "2425")
    stats = fd.upsert_matches(conn, frame)
    assert stats["rejected"] == 1
    assert stats["inserted"] == 1
    assert "Unknown FC" in stats["unresolved"]
    conn.close()


def test_current_season_code():
    from datetime import datetime

    assert fd.current_season_code(datetime(2026, 11, 19)) == "2627"
    assert fd.current_season_code(datetime(2026, 3, 1)) == "2526"
    assert fd.current_season_code(datetime(2026, 7, 15)) == "2627"
