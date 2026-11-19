# data/generate_data.py
# Generates synthetic training data: 500 clients x 9 months = 4,500 rows.
# The churn label is planted by the generator (not observed from real clients).
# Run once: python data/generate_data.py
#
# Design choices:
# - CHURN_RATE = 0.15 → about 75 churners
# - 30% of churners are "quiet" (usage does not drop)
# - 12% of healthy clients have a temporary bad patch but stay
# - Noise on logins: uniform(0.7, 1.3) to keep AUC below 1.0

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

N_CLIENTS = 500
MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]   # 2026-01 to 2026-09
N_MONTHS = len(MONTHS)
CHURN_RATE = 0.15
TOTAL_FEATURES = 20   # fixed product feature count

OUTPUT_PATH = os.path.join("data", "clients_monthly.csv")


def base_revenue_logins(plan):
    """Return (monthly_revenue, base_logins) for a plan type."""
    if plan == "Standard":
        revenue = rng.normal(150_000, 30_000)
        logins = rng.normal(60, 10)
    elif plan == "Premium":
        revenue = rng.normal(500_000, 100_000)
        logins = rng.normal(120, 20)
    else:  # Enterprise
        revenue = rng.normal(800_000, 160_000)
        logins = rng.normal(200, 30)
    # Vary revenue +/-20%
    revenue = revenue * rng.uniform(0.80, 1.20)
    return max(revenue, 10_000), max(logins, 5)


def generate():
    plans = rng.choice(
        ["Standard", "Premium", "Enterprise"],
        size=N_CLIENTS,
        p=[0.50, 0.30, 0.20],
    )
    churn_flags = (rng.random(N_CLIENTS) < CHURN_RATE).astype(int)

    # Renewal dates
    base_date = datetime(2026, 10, 1)
    renewal_dates = []
    for churned in churn_flags:
        if churned:
            days_offset = int(rng.integers(10, 251))
        else:
            days_offset = int(rng.integers(30, 401))
        renewal_dates.append((base_date + timedelta(days=days_offset)).strftime("%Y-%m-%d"))

    rows = []
    for i in range(N_CLIENTS):
        client_id = f"C{i+1:04d}"
        plan = plans[i]
        churned = int(churn_flags[i])
        base_rev, base_logins = base_revenue_logins(plan)
        base_features = rng.integers(8, TOTAL_FEATURES + 1)
        contract_end = renewal_dates[i]

        # Is this churner a "quiet" churner? (30%)
        quiet_churner = churned and rng.random() < 0.30
        # Is this healthy client having a bad patch? (12%)
        bad_patch = (not churned) and rng.random() < 0.12

        for m_idx, month in enumerate(MONTHS):
            # Usage factor for churners (decline from month 3 onwards)
            if churned and not quiet_churner and m_idx >= 2:
                decay_steps = m_idx - 2
                usage_factor = max(0.20, 1.0 - 0.18 * decay_steps)
            elif bad_patch and m_idx in (3, 4, 5):
                usage_factor = rng.uniform(0.50, 0.75)
            else:
                usage_factor = 1.0

            # Logins with noise
            logins = int(max(0, base_logins * usage_factor * rng.uniform(0.70, 1.30)))

            # Features used
            if churned and not quiet_churner and m_idx >= 2:
                features_used = int(max(1, base_features * usage_factor * rng.uniform(0.80, 1.10)))
            else:
                features_used = int(max(1, min(TOTAL_FEATURES, base_features * rng.uniform(0.90, 1.10))))

            # Tickets: rise with declining usage for churners
            if churned and not quiet_churner and m_idx >= 2:
                tickets = int(rng.poisson(max(0.5, (1 - usage_factor) * 4)))
            else:
                tickets = int(rng.poisson(0.3))

            # Last login days ago: rises as usage drops
            if churned and not quiet_churner and m_idx >= 3:
                days_ago = int(rng.integers(10, 30 + int((1 - usage_factor) * 60)))
            else:
                days_ago = int(rng.integers(1, 8))

            revenue = max(10_000, base_rev * rng.uniform(0.95, 1.05))

            rows.append({
                "client_id":          client_id,
                "month":              month,
                "logins":             logins,
                "features_used":      features_used,
                "tickets":            tickets,
                "last_login_days_ago":days_ago,
                "contract_end_date":  contract_end,
                "plan_type":          plan,
                "monthly_revenue":    round(revenue, 2),
                "churned":            churned,
            })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows -> {OUTPUT_PATH}")
    print(f"Clients: {df['client_id'].nunique()}, Months: {df['month'].nunique()}")
    churn_by_client = df.groupby("client_id")["churned"].max()
    print(f"Churn rate (by client): {churn_by_client.mean():.3f}")


if __name__ == "__main__":
    generate()
