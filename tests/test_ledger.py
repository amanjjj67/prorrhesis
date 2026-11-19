from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from prorrhesis.db import open_db
from prorrhesis.ledger import (
    ZERO_HASH,
    append_prediction,
    is_duplicate,
    last_hash,
    row_hash,
    settle,
    verify,
)


FUTURE = "2030-08-10"
FUTURE_2 = "2030-08-11"


def _day(offset: int) -> str:
    base = datetime.fromisoformat(FUTURE + "T00:00:00+00:00")
    return (base + timedelta(days=offset)).date().isoformat()


@pytest.fixture()
def conn(tmp_path):
    c = open_db(tmp_path / "ledger.db")
    yield c
    c.close()


def _append(conn, model="elo", date=FUTURE, config_hash="cfg1", **kw):
    kwargs = {"kickoff_utc": FUTURE + "T15:00:00+00:00"}
    kwargs.update(kw)
    return append_prediction(
        conn, league=kwargs.pop("league", "E0"), date=date,
        home_id=kwargs.pop("home_id", 1), away_id=kwargs.pop("away_id", 2),
        model=model, pH=kwargs.pop("pH", 0.5), pD=kwargs.pop("pD", 0.3),
        pA=kwargs.pop("pA", 0.2), config_hash=config_hash, **kwargs)


def test_the_first_row_chains_onto_the_zero_hash(conn):
    digest = _append(conn)
    row = conn.execute("SELECT prev_hash, hash FROM predictions").fetchone()
    assert row["prev_hash"] == ZERO_HASH
    assert row["hash"] == digest


def test_rows_chain_onto_each_other(conn):
    first = _append(conn, model="elo", date=FUTURE)
    second = _append(conn, model="dc", date=FUTURE_2)
    assert second != first
    row = conn.execute("SELECT prev_hash FROM predictions WHERE id = 2").fetchone()
    assert row["prev_hash"] == first


def test_row_hash_is_stable_and_order_independent():
    row = {"b": 2, "a": 1}
    assert row_hash("0" * 64, row) == row_hash("0" * 64, {"a": 1, "b": 2})
    assert row_hash("0" * 64, row) != row_hash("0" * 64, {"b": 3, "a": 1})


def test_the_chain_verifies(conn):
    for i in range(5):
        _append(conn, model="elo", date=_day(i))
    out = verify(conn)
    assert out["ok"]
    assert out["total"] == 5


def test_update_is_refused(conn):
    _append(conn)
    with pytest.raises(Exception):
        conn.execute("UPDATE predictions SET pH = 0.99 WHERE id = 1")


def test_delete_is_refused(conn):
    _append(conn)
    with pytest.raises(Exception):
        conn.execute("DELETE FROM predictions WHERE id = 1")


def test_verify_detects_a_flipped_bit(conn):
    """Tampering with a stored row must break the chain and be named."""
    for i in range(3):
        _append(conn, model="elo", date=_day(i))
    conn.execute("DROP TRIGGER predictions_no_update")
    conn.execute("UPDATE predictions SET pH = 0.99 WHERE id = 2")
    conn.commit()

    out = verify(conn)
    assert not out["ok"]
    assert out["row_id"] == 2


def test_a_late_prediction_is_rejected(conn):
    late = datetime.fromisoformat(FUTURE + "T18:00:00+00:00")
    digest = _append(conn, date=FUTURE, created_at=late)
    assert digest is None
    assert conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == 0
    rows = conn.execute("SELECT note FROM ingest_log WHERE kind='late_prediction'").fetchall()
    assert rows


def test_a_prediction_after_kickoff_is_rejected(conn):
    kickoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    digest = _append(conn, date=FUTURE, kickoff_utc=kickoff)
    assert digest is None
    assert conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == 0


def test_predicting_twice_in_one_day_is_a_no_op(conn):
    assert _append(conn) is not None
    assert _append(conn) is None
    assert conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == 1


def test_a_different_model_on_the_same_day_is_a_new_row(conn):
    _append(conn, model="elo")
    _append(conn, model="dc")
    assert conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == 2


def test_a_different_config_hash_is_a_new_row(conn):
    _append(conn, config_hash="cfg1")
    _append(conn, config_hash="cfg2")
    assert conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == 2


def test_is_duplicate_looks_at_the_day_not_the_second(conn):
    stamp = FUTURE + "T06:00:00+00:00"
    _append(conn, created_at=datetime.fromisoformat(stamp))
    assert is_duplicate(conn, "E0", FUTURE, 1, 2, "elo", "cfg1",
                        FUTURE + "T18:00:00+00:00")
    assert not is_duplicate(conn, "E0", FUTURE_2, 1, 2, "elo", "cfg1",
                            FUTURE_2 + "T06:00:00+00:00")


def test_settlements_do_not_touch_predictions(conn):
    _append(conn)
    before = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    settle(conn, [{"league": "E0", "date": FUTURE, "home_id": 1, "away_id": 2,
                   "res": "H", "hg": 2, "ag": 0}])
    assert conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == before
    row = conn.execute("SELECT res, hg FROM settlements").fetchone()
    assert row["res"] == "H"
    assert row["hg"] == 2


def test_settlement_is_idempotent(conn):
    payload = [{"league": "E0", "date": FUTURE, "home_id": 1, "away_id": 2,
                "res": "H", "hg": 2, "ag": 0}]
    settle(conn, payload)
    settle(conn, payload)
    assert conn.execute("SELECT COUNT(*) c FROM settlements").fetchone()["c"] == 1


def test_the_chain_still_verifies_after_settlements(conn):
    for i in range(3):
        _append(conn, date=_day(i))
    settle(conn, [{"league": "E0", "date": FUTURE, "home_id": 1, "away_id": 2,
                   "res": "D", "hg": 1, "ag": 1}])
    assert verify(conn)["ok"]


def test_last_hash_is_the_zero_hash_for_an_empty_ledger(conn):
    assert last_hash(conn) == ZERO_HASH
