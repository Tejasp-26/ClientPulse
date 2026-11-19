# ml/actions.py
# Rules that turn SHAP feature names into human-readable warning texts and action suggestions.
# These rules sit on top of SHAP to keep reasons meaningful (SHAP can show "0 tickets" etc.).

ACTIONS = {
    "login_drop_pct":       "Account manager should call the client this week.",
    "feature_adoption":     "Offer a training session on unused features.",
    "feature_drop_pct":     "Offer a training session on unused features.",
    "tickets_last_month":   "Escalate open tickets to the support manager.",
    "tickets_last_3m":      "Escalate open tickets to the support manager.",
    "days_since_last_login":"Send a check-in message and offer help getting started again.",
    "days_to_renewal":      "Send a renewal offer and plan a renewal meeting.",
    "avg_logins_recent":    "Account manager should check in on usage trends.",
}

# Features that must NEVER appear as reasons
SKIP_FEATURES = {"monthly_revenue", "plan_code"}


def is_real_problem(feature: str, value: float) -> bool:
    """
    Return True only when the feature value is genuinely problematic.
    This keeps reasons meaningful – e.g. "0 tickets" is never a warning.
    """
    if feature in SKIP_FEATURES:
        return False

    if feature == "tickets_last_month":
        return value >= 2
    if feature == "tickets_last_3m":
        return value >= 2
    if feature == "days_since_last_login":
        return value >= 5
    if feature in ("login_drop_pct", "feature_drop_pct"):
        return value <= -10   # negative = decline
    if feature == "feature_adoption":
        return value < 60     # below 60% adoption
    if feature == "days_to_renewal":
        return value <= 120

    # avg_logins_recent: flag if very low (below 10)
    if feature == "avg_logins_recent":
        return value < 10

    return False   # unknown feature → skip


def reason_text(feature: str, value: float) -> str:
    """Plain-English sentence explaining why this feature is a warning."""
    if feature == "login_drop_pct":
        return f"Logins fell {abs(value):.0f}% compared to earlier months."
    if feature == "feature_adoption":
        return f"Only {value:.0f}% of product features are being used."
    if feature == "feature_drop_pct":
        return f"Feature usage dropped {abs(value):.0f}% compared to earlier months."
    if feature == "tickets_last_month":
        return f"Client raised {int(value)} support ticket(s) last month."
    if feature == "tickets_last_3m":
        return f"Client raised {int(value)} support ticket(s) in the last 3 months."
    if feature == "days_since_last_login":
        return f"No login in the last {int(value)} day(s)."
    if feature == "days_to_renewal":
        return f"Contract renews in {int(value)} day(s)."
    if feature == "avg_logins_recent":
        return f"Only {value:.1f} logins per month on average recently."
    return f"{feature} is at {value:.2f}."


def action_for(feature: str) -> str:
    """Return the recommended action for a feature, or empty string if none."""
    return ACTIONS.get(feature, "")
