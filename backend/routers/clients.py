# backend/routers/clients.py
# Handles per-client endpoints.
# /clients           GET  – ranked list with filters
# /clients/export    GET  – CSV download
# /clients/{id}      GET  – full client detail + SHAP + login trend
# /clients/{id}/note POST – AI or fallback note

import io
import json
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional

from backend import db
from ml.explain import load_model, get_shap_values
from ml.features import build_features, FEATURE_COLUMNS
from ml.llm_note import write_note
from ml.actions import reason_text

router = APIRouter(tags=["clients"])

FRIENDLY_NAMES = {
    "avg_logins_recent":    "Recent avg logins",
    "login_drop_pct":       "Change in logins",
    "feature_adoption":     "Feature adoption %",
    "feature_drop_pct":     "Change in features used",
    "tickets_last_month":   "Tickets last month",
    "tickets_last_3m":      "Tickets (3 months)",
    "days_since_last_login":"Days since last login",
    "days_to_renewal":      "Days to renewal",
    "plan_code":            "Plan type",
    "monthly_revenue":      "Monthly revenue",
}


def _get_predictions(run_id: Optional[str]) -> tuple[str, pd.DataFrame]:
    rid = run_id or db.latest_run_id()
    if not rid:
        raise HTTPException(status_code=404, detail="No runs found. Run /analyse first.")
    preds = db.load_predictions(rid)
    if preds.empty:
        raise HTTPException(status_code=404, detail=f"Run '{rid}' not found.")
    return rid, preds


@router.get("/clients")
def list_clients(
    run_id: Optional[str] = Query(default=None),
    level: Optional[str] = Query(default=None),
    plan: Optional[str] = Query(default=None),
    sort: str = Query(default="risk_desc"),
):
    """Return a ranked list of clients with optional filters."""
    rid, preds = _get_predictions(run_id)

    if level:
        preds = preds[preds["risk_level"] == level]
    if plan:
        preds = preds[preds["plan_type"] == plan]

    ascending = sort.endswith("_asc")
    preds = preds.sort_values("risk", ascending=ascending)

    result = []
    for _, row in preds.iterrows():
        result.append({
            "client_id": row["client_id"],
            "risk": round(float(row["risk"]), 4),
            "level": row["risk_level"],
            "plan": row["plan_type"],
            "monthly_revenue": float(row["monthly_revenue"]),
            "top_reasons": row["top_reasons"] if isinstance(row["top_reasons"], list) else [],
            "top_actions": row["top_actions"] if isinstance(row["top_actions"], list) else [],
        })
    return result


@router.get("/clients/export")
def export_clients(run_id: Optional[str] = Query(default=None)):
    """Download the ranked client list as a CSV."""
    rid, preds = _get_predictions(run_id)
    preds = preds.sort_values("risk", ascending=False)
    # Flatten list columns
    preds["top_reasons"] = preds["top_reasons"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else ""
    )
    preds["top_actions"] = preds["top_actions"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else ""
    )
    buf = io.StringIO()
    preds.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=clients_{rid}.csv"},
    )


@router.get("/clients/{client_id}")
def client_detail(
    client_id: str,
    run_id: Optional[str] = Query(default=None),
):
    """Full client detail: risk, SHAP values, login trend, reasons, feedback."""
    rid, preds = _get_predictions(run_id)

    row = preds[preds["client_id"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found in run '{rid}'.")
    row = row.iloc[0]

    # Login trend from stored history
    history = db.load_history()
    client_history = history[history["client_id"] == client_id].sort_values("month")
    login_trend = [
        {"month": r["month"], "logins": int(r["logins"])}
        for _, r in client_history.iterrows()
    ]

    # SHAP values for this client
    model = load_model()
    features_df, _ = build_features(history)
    client_features = features_df[features_df["client_id"] == client_id]

    shap_vals = []
    if not client_features.empty:
        raw_shap = get_shap_values(model, client_features)
        for feat, val in zip(FEATURE_COLUMNS, raw_shap):
            shap_vals.append({
                "feature": feat,
                "friendly_name": FRIENDLY_NAMES.get(feat, feat),
                "value": round(float(val), 4),
            })

    # Previous risk
    prev_rid = db.previous_run_id(rid)
    previous_risk = None
    if prev_rid:
        prev_preds = db.load_predictions(prev_rid)
        prev_row = prev_preds[prev_preds["client_id"] == client_id]
        if not prev_row.empty:
            previous_risk = round(float(prev_row.iloc[0]["risk"]), 4)

    # Feedback for this client (all runs)
    feedback_df = db.load_feedback()
    client_feedback = feedback_df[feedback_df["client_id"] == client_id]
    feedback_list = client_feedback.to_dict(orient="records")

    return {
        "client_id": client_id,
        "run_id": rid,
        "risk": round(float(row["risk"]), 4),
        "risk_level": row["risk_level"],
        "plan_type": row["plan_type"],
        "monthly_revenue": float(row["monthly_revenue"]),
        "login_trend": login_trend,
        "shap_values": shap_vals,
        "reasons": row["top_reasons"] if isinstance(row["top_reasons"], list) else [],
        "actions": row["top_actions"] if isinstance(row["top_actions"], list) else [],
        "previous_risk": previous_risk,
        "feedback": feedback_list,
    }


@router.post("/clients/{client_id}/note")
def generate_note(
    client_id: str,
    run_id: Optional[str] = Query(default=None),
):
    """Generate an AI or fallback note for a client."""
    rid, preds = _get_predictions(run_id)

    row = preds[preds["client_id"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found in run '{rid}'.")
    row = row.iloc[0]

    client_facts = {
        "client_id": client_id,
        "risk": round(float(row["risk"]), 4),
        "risk_level": row["risk_level"],
        "reasons": row["top_reasons"] if isinstance(row["top_reasons"], list) else [],
        "actions": row["top_actions"] if isinstance(row["top_actions"], list) else [],
        "plan_type": row["plan_type"],
        "monthly_revenue": float(row["monthly_revenue"]),
    }

    note, source = write_note(client_facts)
    return {"note": note, "source": source}
