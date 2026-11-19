# backend/routers/feedback.py
# Handles saving and summarising account manager feedback.
# /feedback         POST – save one feedback entry
# /feedback/summary GET  – breakdown of outcomes

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend import db
from backend.schemas import FeedbackRequest

router = APIRouter(tags=["feedback"])

VALID_OUTCOMES = {"good_call", "false_alarm", "left", "stayed"}


@router.post("/feedback")
def save_feedback(req: FeedbackRequest):
    """Save one feedback entry for a client prediction."""
    # Validate outcome (Pydantic also does this, but belt-and-suspenders)
    if req.outcome not in VALID_OUTCOMES:
        raise HTTPException(
            status_code=422,
            detail=f"outcome must be one of {sorted(VALID_OUTCOMES)}",
        )
    db.save_feedback(
        run_id=req.run_id,
        client_id=req.client_id,
        outcome=req.outcome,
        note=req.note or "",
    )
    return {"saved": True}


@router.get("/feedback/summary")
def feedback_summary(run_id: Optional[str] = Query(default=None)):
    """Return a breakdown of feedback outcomes."""
    df = db.load_feedback(run_id=run_id)
    if df.empty:
        return {
            "total_feedback": 0,
            "good_call": 0,
            "false_alarm": 0,
            "left": 0,
            "stayed": 0,
            "good_call_rate": None,
        }

    total = len(df)
    counts = df["outcome"].value_counts().to_dict()
    good_call = counts.get("good_call", 0)

    return {
        "total_feedback": total,
        "good_call": good_call,
        "false_alarm": counts.get("false_alarm", 0),
        "left": counts.get("left", 0),
        "stayed": counts.get("stayed", 0),
        "good_call_rate": round(good_call / total, 4) if total > 0 else None,
    }
