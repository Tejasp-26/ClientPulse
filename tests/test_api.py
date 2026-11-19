# tests/test_api.py
# API integration tests using FastAPI TestClient.
# Uses a temporary database file (monkeypatched) to avoid touching production data.
# Run: pytest tests/test_api.py -v
#
# Note: these tests require models/model.pkl to exist.
# Run `python ml/train.py` first.

import io
import os
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Set up a test client with a fresh temp database."""
    tmp = tmp_path_factory.mktemp("db")
    test_db = str(tmp / "test.db")

    import backend.db as db_module
    # Patch before importing main
    original_db = db_module.DB_PATH
    db_module.DB_PATH = test_db

    from backend.main import app
    db_module.create_tables()

    with TestClient(app) as c:
        yield c

    db_module.DB_PATH = original_db


def make_csv_bytes(n_clients=5, n_months=9) -> bytes:
    """Build a minimal valid CSV as bytes."""
    rows = []
    for i in range(1, n_clients + 1):
        for m in range(1, n_months + 1):
            rows.append({
                "client_id": f"API{i:03d}",
                "month": f"2026-{m:02d}",
                "logins": 60,
                "features_used": 12,
                "tickets": 1,
                "last_login_days_ago": 3,
                "contract_end_date": "2027-06-01",
                "plan_type": "Standard",
                "monthly_revenue": 150000.0,
            })
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_upload_valid(client):
    csv_bytes = make_csv_bytes()
    resp = client.post(
        "/upload",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["clients"] > 0
    assert "months" in data


def test_upload_missing_column(client):
    """A CSV missing 'tickets' must return 400."""
    df = pd.DataFrame({
        "client_id": ["X001"],
        "month": ["2026-01"],
        "logins": [10],
        "features_used": [5],
        # tickets missing
        "last_login_days_ago": [3],
        "contract_end_date": ["2027-01-01"],
        "plan_type": ["Standard"],
        "monthly_revenue": [100000],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/upload",
        files={"file": ("bad.csv", buf.getvalue().encode(), "text/csv")},
    )
    assert resp.status_code == 400
    assert "tickets" in resp.json()["detail"].lower() or "tickets" in str(resp.json())


def test_upload_churned_column_ignored(client):
    """A CSV with a 'churned' column must be accepted; the column is silently dropped."""
    rows = []
    for i in range(1, 4):
        for m in range(1, 10):
            rows.append({
                "client_id": f"CHK{i:03d}",
                "month": f"2026-{m:02d}",
                "logins": 60,
                "features_used": 12,
                "tickets": 0,
                "last_login_days_ago": 2,
                "contract_end_date": "2027-06-01",
                "plan_type": "Standard",
                "monthly_revenue": 150000,
                "churned": 1,   # must be ignored
            })
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/upload",
        files={"file": ("with_churned.csv", buf.getvalue().encode(), "text/csv")},
    )
    assert resp.status_code == 200


def test_upload_too_short_history(client):
    """A client with only 2 months must cause a 400."""
    rows = []
    for m in range(1, 3):   # only 2 months
        rows.append({
            "client_id": "SHORT01",
            "month": f"2026-{m:02d}",
            "logins": 60,
            "features_used": 10,
            "tickets": 0,
            "last_login_days_ago": 2,
            "contract_end_date": "2027-06-01",
            "plan_type": "Standard",
            "monthly_revenue": 100000,
        })
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/upload",
        files={"file": ("short.csv", buf.getvalue().encode(), "text/csv")},
    )
    assert resp.status_code == 400


def test_history_summary(client):
    resp = client.get("/history/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "clients" in data


def test_analyse_and_overview(client):
    """Run analysis after valid upload; check overview re-bucketing."""
    # Make sure we have data first (upload valid data)
    csv_bytes = make_csv_bytes(n_clients=5, n_months=9)
    client.post("/upload", files={"file": ("test2.csv", csv_bytes, "text/csv")})

    resp = client.post("/analyse", json={"high": 0.70, "medium": 0.40})
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    total = data["high_risk"] + data["medium_risk"] + data["low_risk"]
    assert total == data["n_clients"]

    # Overview – re-bucket with high=0.9 → high_risk must be <= original high_risk
    original_high = data["high_risk"]
    overview_resp = client.get(f"/overview?high=0.9&medium=0.4")
    assert overview_resp.status_code == 200
    new_high = overview_resp.json()["high_risk"]
    assert new_high <= original_high


def test_analyse_empty_db(client):
    """POST /analyse on an empty DB must return 400."""
    # Delete history first
    client.delete("/history")
    resp = client.post("/analyse", json={"high": 0.70, "medium": 0.40})
    assert resp.status_code == 400


def test_feedback_valid(client):
    # Re-upload and re-analyse to get a valid run_id
    csv_bytes = make_csv_bytes(n_clients=3, n_months=9)
    client.post("/upload", files={"file": ("fb.csv", csv_bytes, "text/csv")})
    ana = client.post("/analyse", json={}).json()
    run_id = ana["run_id"]

    resp = client.post("/feedback", json={
        "run_id": run_id,
        "client_id": "API001",
        "outcome": "good_call",
        "note": "Called, will renew",
    })
    assert resp.status_code == 200
    assert resp.json()["saved"] is True


def test_feedback_invalid_outcome(client):
    resp = client.post("/feedback", json={
        "run_id": "some_run",
        "client_id": "C001",
        "outcome": "not_a_real_outcome",
    })
    assert resp.status_code == 422


def test_unknown_client_detail(client):
    # Need a run to exist
    csv_bytes = make_csv_bytes(n_clients=2, n_months=9)
    client.post("/upload", files={"file": ("unk.csv", csv_bytes, "text/csv")})
    client.post("/analyse", json={})
    resp = client.get("/clients/DOESNOTEXIST")
    assert resp.status_code == 404
