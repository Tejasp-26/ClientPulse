# tests/test_db.py
# Unit tests for backend/db.py
# Uses a temporary database file to avoid touching clientpulse.db.
# Run: pytest tests/test_db.py -v

import os
import pytest
import pandas as pd
import sqlite3

# Patch the DB_PATH before importing db, so tests use a temp file
TEST_DB = "db/test_clientpulse.db"


@pytest.fixture(autouse=True)
def use_temp_db(monkeypatch, tmp_path):
    """Redirect every test to a fresh temp database."""
    import backend.db as db_module
    test_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.create_tables()
    yield
    # Cleanup: the tmp_path fixture handles deletion


def make_sample_df(n_clients=3, n_months=6):
    rows = []
    for i in range(1, n_clients + 1):
        for m in range(1, n_months + 1):
            rows.append({
                "client_id": f"T{i:03d}",
                "month": f"2026-{m:02d}",
                "logins": 50,
                "features_used": 10,
                "tickets": 1,
                "last_login_days_ago": 3,
                "contract_end_date": "2027-06-01",
                "plan_type": "Standard",
                "monthly_revenue": 100000.0,
            })
    return pd.DataFrame(rows)


def test_save_and_load_history():
    from backend.db import _save_usage_replace, load_history
    df = make_sample_df()
    _save_usage_replace(df)
    loaded = load_history()
    assert len(loaded) == len(df), "Row count must match after save"


def test_no_duplicates_on_re_save():
    """Saving the same data twice must not duplicate rows."""
    from backend.db import _save_usage_replace, load_history
    df = make_sample_df()
    _save_usage_replace(df)
    _save_usage_replace(df)   # second save
    loaded = load_history()
    assert len(loaded) == len(df), "Re-saving must not create duplicate rows"


def test_count_history():
    from backend.db import _save_usage_replace, count_history
    df = make_sample_df(n_clients=5)
    _save_usage_replace(df)
    assert count_history() == 5


def test_clear_usage():
    from backend.db import _save_usage_replace, clear_usage, load_history
    _save_usage_replace(make_sample_df())
    clear_usage()
    assert load_history().empty


def test_save_and_load_feedback():
    from backend.db import save_feedback, load_feedback
    save_feedback(run_id="RUN001", client_id="T001", outcome="good_call", note="Great call")
    df = load_feedback()
    assert len(df) == 1
    assert df.iloc[0]["outcome"] == "good_call"


def test_list_runs():
    from backend.db import save_run, list_runs
    save_run("20261101-090000", n=50, high=0.70, medium=0.40)
    save_run("20261101-100000", n=50, high=0.70, medium=0.40)
    runs = list_runs()
    assert len(runs) == 2
    assert runs[0]["run_id"] > runs[1]["run_id"]   # newest first
