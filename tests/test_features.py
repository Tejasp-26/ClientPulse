# tests/test_features.py
# Unit tests for ml/features.py
# Run: pytest tests/test_features.py -v

import pandas as pd
import pytest
from ml.features import build_features, check_columns, REQUIRED_COLUMNS, FEATURE_COLUMNS


def make_client_df(client_id="TEST01", n_months=9, plan="Standard", logins_per_month=None):
    """Helper: create a simple monthly DataFrame for one client."""
    months = [f"2026-{m:02d}" for m in range(1, n_months + 1)]
    rows = []
    for idx, month in enumerate(months):
        logins = logins_per_month[idx] if logins_per_month else 60
        rows.append({
            "client_id": client_id,
            "month": month,
            "logins": logins,
            "features_used": 12,
            "tickets": 1,
            "last_login_days_ago": 3,
            "contract_end_date": "2027-06-01",
            "plan_type": plan,
            "monthly_revenue": 150000.0,
        })
    return pd.DataFrame(rows)


def test_check_columns_all_present():
    df = make_client_df()
    missing = check_columns(df)
    assert missing == [], f"Expected no missing columns, got {missing}"


def test_check_columns_missing():
    df = make_client_df().drop(columns=["tickets", "logins"])
    missing = check_columns(df)
    assert "tickets" in missing
    assert "logins" in missing


def test_feature_columns_order():
    df = make_client_df()
    features_df, _ = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in features_df.columns, f"Feature column '{col}' missing from output"


def test_login_drop_pct_three_months_is_zero():
    """Clients with exactly 3 months should have login_drop_pct = 0 (no earlier data)."""
    df = make_client_df(n_months=3)
    features_df, _ = build_features(df)
    row = features_df.iloc[0]
    assert row["login_drop_pct"] == 0.0, "login_drop_pct must be 0 when there are only 3 months"
    assert row["feature_drop_pct"] == 0.0, "feature_drop_pct must be 0 when there are only 3 months"


def test_login_drop_pct_detects_decline():
    """A client whose logins fall sharply should have a negative login_drop_pct."""
    # First 6 months: high logins; last 3: low logins
    logins = [100, 100, 100, 100, 100, 100, 10, 10, 10]
    df = make_client_df(n_months=9, logins_per_month=logins)
    features_df, _ = build_features(df)
    drop = features_df.iloc[0]["login_drop_pct"]
    assert drop < -50, f"Expected big drop, got {drop}"


def test_no_division_by_zero():
    """Earlier avg logins = 0 must not crash."""
    logins = [0, 0, 0, 0, 0, 0, 10, 10, 10]
    df = make_client_df(n_months=9, logins_per_month=logins)
    features_df, _ = build_features(df)
    # Should not raise; drop should be 0
    assert features_df.iloc[0]["login_drop_pct"] == 0.0


def test_plan_code_mapping():
    for plan, expected_code in [("Standard", 0), ("Premium", 1), ("Enterprise", 2)]:
        df = make_client_df(plan=plan)
        features_df, _ = build_features(df)
        assert features_df.iloc[0]["plan_code"] == expected_code
