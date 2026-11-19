# tests/test_actions.py
# Unit tests for ml/actions.py
# Run: pytest tests/test_actions.py -v

import pytest
from ml.actions import is_real_problem, reason_text, action_for, SKIP_FEATURES


def test_revenue_is_never_a_reason():
    """monthly_revenue must never be flagged as a real problem."""
    assert not is_real_problem("monthly_revenue", 999999)
    assert not is_real_problem("monthly_revenue", 0)


def test_plan_code_is_never_a_reason():
    """plan_code must never be flagged as a real problem."""
    assert not is_real_problem("plan_code", 0)
    assert not is_real_problem("plan_code", 2)


def test_zero_tickets_is_not_a_reason():
    """0 tickets must not trigger the rule (>= 2 required)."""
    assert not is_real_problem("tickets_last_month", 0)
    assert not is_real_problem("tickets_last_month", 1)
    assert not is_real_problem("tickets_last_3m", 0)
    assert not is_real_problem("tickets_last_3m", 1)


def test_two_tickets_is_a_reason():
    assert is_real_problem("tickets_last_month", 2)
    assert is_real_problem("tickets_last_3m", 3)


def test_days_since_last_login_threshold():
    assert not is_real_problem("days_since_last_login", 4)
    assert is_real_problem("days_since_last_login", 5)
    assert is_real_problem("days_since_last_login", 30)


def test_login_drop_threshold():
    # Must be <= -10 to be a problem (negative = decline)
    assert not is_real_problem("login_drop_pct", -9)
    assert not is_real_problem("login_drop_pct", 0)
    assert not is_real_problem("login_drop_pct", 20)
    assert is_real_problem("login_drop_pct", -10)
    assert is_real_problem("login_drop_pct", -50)


def test_feature_adoption_threshold():
    assert is_real_problem("feature_adoption", 50)
    assert not is_real_problem("feature_adoption", 60)
    assert not is_real_problem("feature_adoption", 80)


def test_days_to_renewal_threshold():
    assert is_real_problem("days_to_renewal", 120)
    assert is_real_problem("days_to_renewal", 30)
    assert not is_real_problem("days_to_renewal", 121)


def test_reason_text_not_empty():
    """reason_text should return a non-empty string for every feature."""
    test_cases = [
        ("login_drop_pct", -40.0),
        ("feature_adoption", 45.0),
        ("tickets_last_month", 3),
        ("days_since_last_login", 10),
        ("days_to_renewal", 60),
    ]
    for feat, val in test_cases:
        text = reason_text(feat, val)
        assert isinstance(text, str) and len(text) > 0


def test_action_for_returns_string():
    assert isinstance(action_for("login_drop_pct"), str)
    assert isinstance(action_for("nonexistent_feature"), str)
    assert action_for("nonexistent_feature") == ""
