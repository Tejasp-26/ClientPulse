# backend/routers/upload.py
# Handles file upload, sample loading, history summary, and history reset.
# /upload        POST  – validate and store a CSV or xlsx file
# /load-sample   POST  – load the built-in sample_clients.csv
# /history/summary GET – counts of what is stored
# /history       DELETE – wipe usage data (for testing / reset)

import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend import db
from ml.features import REQUIRED_COLUMNS, check_columns

router = APIRouter(tags=["upload"])

NUMERIC_COLUMNS = ["logins", "features_used", "tickets", "last_login_days_ago", "monthly_revenue"]


def _validate_and_save(df: pd.DataFrame) -> dict:
    """
    Shared logic for both /upload and /load-sample.
    Returns a response dict or raises HTTPException.
    """
    # Drop the 'churned' column silently if present
    if "churned" in df.columns:
        df = df.drop(columns=["churned"])

    # Check for required columns
    missing = check_columns(df)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    # Check numeric columns contain numbers
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad_count = coerced.isna().sum() - df[col].isna().sum()
            if bad_count > 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{col}' contains non-numeric values. All values must be numbers.",
                )
            df[col] = coerced

    # Remove exact duplicate (client_id, month) rows inside this file
    before = len(df)
    df = df.drop_duplicates(subset=["client_id", "month"])
    dupes_dropped = before - len(df)

    # Load existing history to check total months per client after merge
    existing = db.load_history()
    if not existing.empty:
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    months_per_client = combined.groupby("client_id")["month"].nunique()
    too_short = months_per_client[months_per_client < 3].index.tolist()
    if too_short:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{len(too_short)} client(s) would still have fewer than 3 months after upload "
                f"(e.g. {too_short[:3]}). Each client needs at least 3 months total. "
                "Upload was rejected. Please include more history."
            ),
        )

    # Warn for clients with 3-5 months
    short_history = months_per_client[(months_per_client >= 3) & (months_per_client <= 5)].index.tolist()
    warnings = []
    if short_history:
        warnings.append(
            f"{len(short_history)} clients have fewer than 6 months of history; "
            "trend features are less reliable for these clients."
        )
    if dupes_dropped > 0:
        warnings.append(f"{dupes_dropped} duplicate (client_id, month) rows were removed before saving.")

    # Save to database (INSERT OR REPLACE handles conflicts)
    db._save_usage_replace(df)

    months_saved = sorted(df["month"].unique().tolist())
    clients_count = db.count_history()

    return {
        "rows_received": len(df) + dupes_dropped,
        "rows_saved": len(df),
        "clients": clients_count,
        "months": months_saved,
        "warnings": warnings,
    }


@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """Upload a CSV or xlsx file with client usage data."""
    filename = file.filename or ""
    content = file.file.read()

    try:
        if filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="File must be .csv or .xlsx")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    return _validate_and_save(df)


@router.post("/load-sample")
def load_sample():
    """Load the built-in sample_clients.csv into the database."""
    import os
    sample_path = os.path.join("data", "sample_clients.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(
            status_code=400,
            detail="sample_clients.csv not found. Run: python data/make_sample.py",
        )
    df = pd.read_csv(sample_path)
    return _validate_and_save(df)


@router.get("/history/summary")
def history_summary():
    """Return counts of what is stored in the database."""
    df = db.load_history()
    if df.empty:
        return {"clients": 0, "first_month": None, "last_month": None, "rows": 0}
    return {
        "clients": df["client_id"].nunique(),
        "first_month": df["month"].min(),
        "last_month": df["month"].max(),
        "rows": len(df),
    }


@router.delete("/history")
def delete_history():
    """Wipe all usage data. Use for testing and reset."""
    db.clear_usage()
    return {"deleted": True, "message": "All usage data cleared."}
