# data/make_sample.py
# Generates 80 unseen sample clients (N001-N080) for the demo.
# Uses a different seed (7) so the model has never seen these clients.
# Run once: python data/make_sample.py

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 7
rng = np.random.default_rng(SEED)

N_CLIENTS = 80
MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]   # 2026-01 to 2026-09
TOTAL_FEATURES = 20
OUTPUT_PATH = os.path.join("data", "sample_clients.csv")


def base_revenue_logins(plan):
    if plan == "Standard":
        return max(10_000, rng.normal(150_000, 30_000) * rng.uniform(0.80, 1.20)), max(5, rng.normal(60, 10))
    elif plan == "Premium":
        return max(10_000, rng.normal(500_000, 100_000) * rng.uniform(0.80, 1.20)), max(5, rng.normal(120, 20))
    else:
        return max(10_000, rng.normal(800_000, 160_000) * rng.uniform(0.80, 1.20)), max(5, rng.normal(200, 30))


def generate():
    plans = rng.choice(
        ["Standard", "Premium", "Enterprise"],
        size=N_CLIENTS,
        p=[0.50, 0.30, 0.20],
    )

    # Mix of risky and healthy clients (no churn label – this is production-like data)
    risk_profiles = rng.choice(["high_risk", "medium_risk", "healthy"], size=N_CLIENTS, p=[0.15, 0.25, 0.60])

    base_date = datetime(2026, 10, 1)
    renewal_dates = []
    for profile in risk_profiles:
        if profile == "high_risk":
            days = int(rng.integers(10, 120))
        elif profile == "medium_risk":
            days = int(rng.integers(100, 250))
        else:
            days = int(rng.integers(200, 400))
        renewal_dates.append((base_date + timedelta(days=days)).strftime("%Y-%m-%d"))

    rows = []
    for i in range(N_CLIENTS):
        client_id = f"N{i+1:03d}"
        plan = plans[i]
        profile = risk_profiles[i]
        base_rev, base_logins = base_revenue_logins(plan)
        base_features = rng.integers(8, TOTAL_FEATURES + 1)
        contract_end = renewal_dates[i]

        for m_idx, month in enumerate(MONTHS):
            if profile == "high_risk" and m_idx >= 2:
                decay = max(0.20, 1.0 - 0.18 * (m_idx - 2))
                logins = int(max(0, base_logins * decay * rng.uniform(0.70, 1.20)))
                features_used = int(max(1, base_features * decay * rng.uniform(0.80, 1.10)))
                tickets = int(rng.poisson(max(0.5, (1 - decay) * 4)))
                days_ago = int(rng.integers(8, 30 + int((1 - decay) * 50)))
            elif profile == "medium_risk" and m_idx >= 4:
                logins = int(max(0, base_logins * 0.75 * rng.uniform(0.80, 1.10)))
                features_used = int(max(1, base_features * 0.80 * rng.uniform(0.90, 1.10)))
                tickets = int(rng.poisson(0.8))
                days_ago = int(rng.integers(3, 15))
            else:
                logins = int(max(0, base_logins * rng.uniform(0.90, 1.10)))
                features_used = int(max(1, min(TOTAL_FEATURES, base_features * rng.uniform(0.90, 1.10))))
                tickets = int(rng.poisson(0.2))
                days_ago = int(rng.integers(1, 5))

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
            })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows -> {OUTPUT_PATH}")
    print(f"Clients: {df['client_id'].nunique()}, Months: {df['month'].nunique()}")


if __name__ == "__main__":
    generate()
