# backend/pipeline.py
# run_analysis() is the single function that ties ML to the database.
# It loads history, builds features, scores clients, gets reasons, and saves
# the run and predictions. The analysis router calls this when /analyse is hit.

import pandas as pd
from datetime import datetime, timezone

from backend import db
from ml.features import build_features
from ml.explain import load_model, score_clients, get_reasons, get_shap_values
from ml.actions import action_for


def run_analysis(high: float = 0.70, medium: float = 0.40) -> dict:
    """
    Full online pipeline:
      1. Load stored history from SQLite.
      2. Build one-row-per-client features.
      3. Score with XGBoost.
      4. Get top reasons and actions (SHAP + rules).
      5. Save the run and predictions.
      6. Return a summary dict.
    """
    history_df = db.load_history()
    if history_df.empty:
        raise ValueError("Database is empty. Upload data first.")

    # Only keep clients with at least 3 months of history
    months_per_client = history_df.groupby("client_id")["month"].nunique()
    eligible = months_per_client[months_per_client >= 3].index
    if len(eligible) == 0:
        raise ValueError("No client has 3 or more months of history.")

    history_df = history_df[history_df["client_id"].isin(eligible)].copy()

    # Build features (one row per client)
    features_df, warnings = build_features(history_df)

    # Load model and score
    model = load_model()
    scored_df = score_clients(model, features_df, high=high, medium=medium)

    # Get reasons and actions for each client
    reasons_df = get_reasons(model, features_df, top_n=3)
    scored_df = scored_df.merge(reasons_df, on="client_id", how="left")

    # Build top_actions from top_reasons
    def make_actions(reasons):
        if not isinstance(reasons, list):
            return []
        return [action_for(r["feature"]) for r in reasons if action_for(r["feature"])]

    scored_df["top_actions"] = scored_df["top_reasons_raw"].apply(make_actions)

    # Convert top_reasons_raw (list of dicts) to list of plain strings for storage
    def reasons_to_text(reasons):
        if not isinstance(reasons, list):
            return []
        return [r["text"] for r in reasons]

    scored_df["top_reasons"] = scored_df["top_reasons_raw"].apply(reasons_to_text)

    # Merge revenue and plan_type back from the most recent month
    latest_month = history_df.sort_values("month").groupby("client_id").last().reset_index()
    scored_df = scored_df.merge(
        latest_month[["client_id", "monthly_revenue", "plan_type"]],
        on="client_id",
        how="left",
    )

    # Create a timestamp-based run_id
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Save run and predictions
    db.save_run(run_id, n=len(scored_df), high=high, medium=medium)
    db.save_predictions(run_id, scored_df)

    counts = scored_df["risk_level"].value_counts().to_dict()
    return {
        "run_id": run_id,
        "n_clients": len(scored_df),
        "high_risk": counts.get("High", 0),
        "medium_risk": counts.get("Medium", 0),
        "low_risk": counts.get("Low", 0),
        "warnings": warnings,
    }
