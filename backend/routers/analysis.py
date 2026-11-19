# backend/routers/analysis.py
# Handles running the analysis and returning results.
# /analyse  POST  – run the ML pipeline on stored history
# /runs     GET   – list all past runs
# /overview GET   – summary metrics for a run
# /impact   GET   – revenue impact estimate
# /changes  GET   – month-over-month risk changes
# /model/info GET – model metrics from metrics.json

import json
import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend import db
from backend.schemas import AnalyseRequest
from backend.pipeline import run_analysis
from ml.utils import impact_estimate, money

router = APIRouter(tags=["analysis"])


@router.post("/analyse")
def analyse(req: AnalyseRequest = None):
    """Run churn analysis on all stored history."""
    if req is None:
        req = AnalyseRequest()
    try:
        result = run_analysis(high=req.high, medium=req.medium)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/runs")
def list_runs():
    """List all past analysis runs, newest first."""
    return db.list_runs()


@router.get("/overview")
def overview(
    run_id: Optional[str] = Query(default=None),
    high: float = Query(default=0.70),
    medium: float = Query(default=0.40),
):
    """Return overview metrics for a run. Re-buckets without re-running the model."""
    rid = run_id or db.latest_run_id()
    if not rid:
        raise HTTPException(status_code=404, detail="No runs found. Run /analyse first.")

    preds = db.load_predictions(rid)
    if preds.empty:
        raise HTTPException(status_code=404, detail=f"Run '{rid}' not found.")

    # Re-bucket using the provided thresholds (no re-run needed)
    def bucket(risk):
        if risk >= high:
            return "High"
        elif risk >= medium:
            return "Medium"
        return "Low"

    preds["risk_level"] = preds["risk"].apply(bucket)

    n_high = (preds["risk_level"] == "High").sum()
    n_medium = (preds["risk_level"] == "Medium").sum()
    n_low = (preds["risk_level"] == "Low").sum()

    # Predicted churn rate: share of clients with risk >= 0.50
    churn_rate = (preds["risk"] >= 0.50).mean()

    # Revenue at risk: sum of monthly_revenue for High-risk clients
    rev_at_risk = preds.loc[preds["risk_level"] == "High", "monthly_revenue"].sum()

    # Top warning sign: most common first reason across all clients
    all_reasons = []
    for reasons in preds["top_reasons"]:
        if isinstance(reasons, list):
            all_reasons.extend(reasons)
    top_warning = None
    if all_reasons:
        from collections import Counter
        top_warning = Counter(all_reasons).most_common(1)[0][0]

    # Top reasons counts (for chart)
    from collections import Counter
    reason_counts = {}
    for reasons in preds["top_reasons"]:
        if isinstance(reasons, list):
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1

    # Revenue at risk by plan
    rev_by_plan = (
        preds[preds["risk_level"] == "High"]
        .groupby("plan_type")["monthly_revenue"]
        .sum()
        .to_dict()
    )

    return {
        "run_id": rid,
        "clients_analysed": len(preds),
        "predicted_churn_rate": round(churn_rate, 4),
        "high_risk": int(n_high),
        "medium_risk": int(n_medium),
        "low_risk": int(n_low),
        "monthly_revenue_at_risk": float(rev_at_risk),
        "top_warning_sign": top_warning,
        "risk_group_counts": {"High": int(n_high), "Medium": int(n_medium), "Low": int(n_low)},
        "top_reasons_counts": reason_counts,
        "revenue_at_risk_by_plan": rev_by_plan,
    }


@router.get("/impact")
def impact(
    run_id: Optional[str] = Query(default=None),
    contact_pct: float = Query(default=20),
    save_pct: float = Query(default=30),
):
    """Estimate revenue impact based on user-set assumptions."""
    rid = run_id or db.latest_run_id()
    if not rid:
        raise HTTPException(status_code=404, detail="No runs found. Run /analyse first.")

    preds = db.load_predictions(rid)
    if preds.empty:
        raise HTTPException(status_code=404, detail=f"Run '{rid}' not found.")

    result = impact_estimate(preds, contact_pct=contact_pct, save_pct=save_pct)
    result["caption"] = (
        f"If you contact the top {contact_pct:.0f}% of clients by risk "
        f"and save {save_pct:.0f}% of those who are likely to leave, "
        f"you could retain {money(result['saved_per_month'])} / month "
        f"({money(result['saved_per_year'])} / year)."
    )
    return result


@router.get("/changes")
def changes(
    run_id: Optional[str] = Query(default=None),
    top: int = Query(default=10),
):
    """Compare current run with the previous run to find risers and improvers."""
    rid = run_id or db.latest_run_id()
    if not rid:
        raise HTTPException(status_code=404, detail="No runs found.")

    prev_rid = db.previous_run_id(rid)
    if not prev_rid:
        return {
            "has_previous": False,
            "message": "Run another analysis next month to see changes.",
        }

    current = db.load_predictions(rid)[["client_id", "risk"]].rename(columns={"risk": "risk"})
    previous = db.load_predictions(prev_rid)[["client_id", "risk"]].rename(columns={"risk": "prev_risk"})

    merged = current.merge(previous, on="client_id", how="inner")
    merged["change"] = merged["risk"] - merged["prev_risk"]

    risers = (
        merged[merged["change"] > 0]
        .sort_values("change", ascending=False)
        .head(top)
    )
    improvers = (
        merged[merged["change"] < 0]
        .sort_values("change", ascending=True)
        .head(top)
    )

    def to_list(df):
        return [
            {
                "client_id": row["client_id"],
                "previous_risk": round(float(row["prev_risk"]), 4),
                "risk": round(float(row["risk"]), 4),
                "change": round(float(row["change"]), 4),
            }
            for _, row in df.iterrows()
        ]

    return {
        "has_previous": True,
        "run_id": rid,
        "previous_run_id": prev_rid,
        "risers": to_list(risers),
        "improvers": to_list(improvers),
    }


@router.get("/model/info")
def model_info():
    """Return the saved model metrics and feature list."""
    metrics_path = os.path.join("models", "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found. Run: python ml/train.py",
        )
    with open(metrics_path) as f:
        metrics = json.load(f)

    from ml.features import FEATURE_COLUMNS
    return {"metrics": metrics, "feature_columns": FEATURE_COLUMNS}
