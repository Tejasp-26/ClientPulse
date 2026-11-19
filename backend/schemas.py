# backend/schemas.py
# Pydantic models used for request bodies and response shapes.
# FastAPI reads these to validate incoming data and auto-generate API docs.

from pydantic import BaseModel, field_validator
from typing import Optional


VALID_OUTCOMES = {"good_call", "false_alarm", "left", "stayed"}


class AnalyseRequest(BaseModel):
    high: float = 0.70
    medium: float = 0.40


class FeedbackRequest(BaseModel):
    run_id: str
    client_id: str
    outcome: str
    note: Optional[str] = ""

    @field_validator("outcome")
    @classmethod
    def outcome_must_be_valid(cls, v):
        if v not in VALID_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}")
        return v
