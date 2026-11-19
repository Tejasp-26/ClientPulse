# ml/features.py
# Turns raw monthly rows into one-row-per-client features.
# build_features(df) is the single most important function here;
# both train.py and the online pipeline call it so they always use the same features.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

TOTAL_PRODUCT_FEATURES = 20   # fixed total number of features in the product

REQUIRED_COLUMNS = [
    "client_id", "month", "logins", "features_used", "tickets",
    "last_login_days_ago", "contract_end_date", "plan_type", "monthly_revenue",
]

FEATURE_COLUMNS = [
    "avg_logins_recent",
    "login_drop_pct",
    "feature_adoption",
    "feature_drop_pct",
    "tickets_last_month",
    "tickets_last_3m",
    "days_since_last_login",
    "days_to_renewal",
    "plan_code",
    "monthly_revenue",
]

PLAN_MAP = {"Standard": 0, "Premium": 1, "Enterprise": 2}


def check_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of required columns that are missing from df."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Takes a DataFrame of monthly rows and returns:
      - features_df: one row per client with FEATURE_COLUMNS + 'client_id'
      - warnings: list of warning strings (e.g. for clients with < 6 months)

    'Today' is defined as the first day after the last month in the data.
    """
    df = df.copy()
    df["month"] = df["month"].astype(str)
    df = df.sort_values(["client_id", "month"])

    # Parse "today" from the latest month in the whole dataset
    last_month_str = df["month"].max()   # e.g. "2026-09"
    last_month_dt = datetime.strptime(last_month_str, "%Y-%m")
    # First day of the month after the last month
    if last_month_dt.month == 12:
        today = last_month_dt.replace(year=last_month_dt.year + 1, month=1, day=1)
    else:
        today = last_month_dt.replace(month=last_month_dt.month + 1, day=1)

    rows = []
    warnings = []
    short_history_clients = []

    for client_id, group in df.groupby("client_id"):
        group = group.sort_values("month")
        n_months = len(group)

        # --- recent = last 3 months, earlier = everything before ---
        recent = group.tail(3)
        earlier = group.iloc[:-3] if n_months > 3 else pd.DataFrame()

        # avg_logins_recent
        avg_logins_recent = recent["logins"].mean()

        # login_drop_pct
        if earlier.empty or earlier["logins"].mean() == 0:
            login_drop_pct = 0.0
            if n_months <= 3:
                pass  # expected, no warning per client needed
        else:
            earlier_avg = earlier["logins"].mean()
            login_drop_pct = ((avg_logins_recent - earlier_avg) / earlier_avg) * 100

        # feature_adoption (use recent, as % of TOTAL_PRODUCT_FEATURES)
        avg_features_recent = recent["features_used"].mean()
        feature_adoption = (avg_features_recent / TOTAL_PRODUCT_FEATURES) * 100

        # feature_drop_pct
        if earlier.empty or earlier["features_used"].mean() == 0:
            feature_drop_pct = 0.0
        else:
            earlier_feat_avg = earlier["features_used"].mean()
            recent_feat_avg = avg_features_recent
            feature_drop_pct = ((recent_feat_avg - earlier_feat_avg) / earlier_feat_avg) * 100

        # tickets
        last_row = group.iloc[-1]
        tickets_last_month = int(last_row["tickets"])
        tickets_last_3m = int(recent["tickets"].sum())

        # days_since_last_login (most recent month's value)
        days_since_last_login = int(last_row["last_login_days_ago"])

        # days_to_renewal
        try:
            contract_end = datetime.strptime(str(last_row["contract_end_date"]), "%Y-%m-%d")
            days_to_renewal = (contract_end - today).days
        except Exception:
            days_to_renewal = 365   # default if date is unparseable

        # plan_code
        plan_code = PLAN_MAP.get(str(last_row["plan_type"]), 0)

        # monthly_revenue (most recent month)
        monthly_revenue = float(last_row["monthly_revenue"])

        # Short-history warning (3-5 months)
        if n_months <= 5:
            short_history_clients.append(client_id)

        rows.append({
            "client_id": client_id,
            "avg_logins_recent":    round(avg_logins_recent, 2),
            "login_drop_pct":       round(login_drop_pct, 2),
            "feature_adoption":     round(feature_adoption, 2),
            "feature_drop_pct":     round(feature_drop_pct, 2),
            "tickets_last_month":   tickets_last_month,
            "tickets_last_3m":      tickets_last_3m,
            "days_since_last_login":days_since_last_login,
            "days_to_renewal":      days_to_renewal,
            "plan_code":            plan_code,
            "monthly_revenue":      monthly_revenue,
        })

    if short_history_clients:
        warnings.append(
            f"{len(short_history_clients)} clients have fewer than 6 months of history; "
            "trend features (login_drop_pct, feature_drop_pct) are set to 0 for clients with ≤3 months."
        )

    features_df = pd.DataFrame(rows)
    return features_df, warnings
