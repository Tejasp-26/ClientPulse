# ml/utils.py
# Small helper functions used across the project.
# money()          – format rupees as ₹1.5 L or ₹1.37 Cr
# impact_estimate() – estimate revenue saved based on user assumptions

import pandas as pd


def money(x: float) -> str:
    """Format a rupee amount as ₹X.XX L or ₹X.XX Cr."""
    if x >= 1_00_00_000:   # 1 crore = 10,000,000
        return f"₹{x / 1_00_00_000:.2f} Cr"
    elif x >= 1_00_000:    # 1 lakh = 100,000
        return f"₹{x / 1_00_000:.2f} L"
    else:
        return f"₹{x:,.0f}"


def impact_estimate(
    scored_df: pd.DataFrame,
    contact_pct: float = 20,
    save_pct: float = 30,
) -> dict:
    """
    Estimate revenue impact from proactive outreach.

    Steps:
    1. Rank clients by risk; contact the top contact_pct% (at least 1).
    2. Of those, 'likely leavers' = risk >= 0.50.
    3. Revenue at stake = sum of their monthly_revenue.
    4. Saved per month = revenue at stake x save_pct / 100.
    5. Saved per year = saved per month x 12.

    Returns a dict. All numbers are estimates from user-set assumptions.
    """
    df = scored_df.sort_values("risk", ascending=False).reset_index(drop=True)

    n_total = len(df)
    n_contact = max(1, int(round(n_total * contact_pct / 100)))
    contacted = df.head(n_contact)

    likely_leavers = contacted[contacted["risk"] >= 0.50]
    revenue_at_stake = float(likely_leavers["monthly_revenue"].sum())
    saved_per_month = revenue_at_stake * save_pct / 100
    saved_per_year = saved_per_month * 12

    return {
        "clients_contacted": n_contact,
        "likely_leavers": len(likely_leavers),
        "revenue_at_stake": revenue_at_stake,
        "saved_per_month": saved_per_month,
        "saved_per_year": saved_per_year,
        "contact_pct": contact_pct,
        "save_pct": save_pct,
    }


def simple_note(client_facts: dict) -> str:
    """
    Rule-based fallback note when Gemini is not available.
    Uses the risk level and top reasons to build a short sentence.
    """
    level = client_facts.get("risk_level", "unknown")
    reasons = client_facts.get("reasons", [])
    client_id = client_facts.get("client_id", "This client")

    if not reasons:
        return (
            f"{client_id} is at {level.lower()} churn risk. "
            "No specific warning signs were identified by the model."
        )

    top_reason = reasons[0]
    rest = f" Other signals: {'; '.join(reasons[1:])}." if len(reasons) > 1 else ""
    return (
        f"{client_id} is at {level.lower()} churn risk. "
        f"Key warning: {top_reason}.{rest}"
    )
