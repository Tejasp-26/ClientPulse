# backend/db.py
# All SQLite code lives here.
# No ORM – just plain sqlite3 queries and pandas for bulk reads/writes.
# Every other file that needs the database calls these functions.

import sqlite3
import os
import json
import pandas as pd
from datetime import datetime, timezone

DB_PATH = os.path.join("db", "clientpulse.db")


def get_connection():
    """Return a sqlite3 connection to the database file."""
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


def create_tables():
    """Create all tables if they don't exist yet. Safe to call every startup."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS usage_monthly (
            client_id           TEXT,
            month               TEXT,
            logins              INTEGER,
            features_used       INTEGER,
            tickets             INTEGER,
            last_login_days_ago INTEGER,
            contract_end_date   TEXT,
            plan_type           TEXT,
            monthly_revenue     REAL,
            PRIMARY KEY (client_id, month)
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id           TEXT PRIMARY KEY,
            run_date         TEXT,
            n_clients        INTEGER,
            high_threshold   REAL,
            medium_threshold REAL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            run_id          TEXT,
            client_id       TEXT,
            risk            REAL,
            risk_level      TEXT,
            top_reasons     TEXT,
            top_actions     TEXT,
            monthly_revenue REAL,
            plan_type       TEXT,
            PRIMARY KEY (run_id, client_id)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id     TEXT,
            client_id  TEXT,
            outcome    TEXT,
            note       TEXT,
            created_at TEXT
        );
    """)

    conn.commit()
    conn.close()


def save_usage(df: pd.DataFrame):
    """
    Insert or replace rows into usage_monthly.
    Uses INSERT OR REPLACE so uploading the same month again just updates the row.
    """
    conn = get_connection()
    df.to_sql("usage_monthly", conn, if_exists="append", index=False, method="multi")
    conn.close()


def _save_usage_replace(df: pd.DataFrame):
    """
    Low-level INSERT OR REPLACE using executemany.
    Called by save_usage to avoid duplicates.
    """
    conn = get_connection()
    cur = conn.cursor()
    cols = [
        "client_id", "month", "logins", "features_used",
        "tickets", "last_login_days_ago", "contract_end_date",
        "plan_type", "monthly_revenue",
    ]
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT OR REPLACE INTO usage_monthly ({', '.join(cols)}) VALUES ({placeholders})"
    rows = df[cols].values.tolist()
    cur.executemany(sql, rows)
    conn.commit()
    conn.close()


def load_history() -> pd.DataFrame:
    """Load the full usage_monthly table as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM usage_monthly", conn)
    conn.close()
    return df


def count_history() -> int:
    """Return the number of distinct clients in storage."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT client_id) FROM usage_monthly")
    result = cur.fetchone()[0]
    conn.close()
    return result


def save_run(run_id: str, n: int, high: float, medium: float):
    """Insert a new row into the runs table."""
    conn = get_connection()
    cur = conn.cursor()
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT OR REPLACE INTO runs (run_id, run_date, n_clients, high_threshold, medium_threshold) VALUES (?, ?, ?, ?, ?)",
        (run_id, run_date, n, high, medium),
    )
    conn.commit()
    conn.close()


def save_predictions(run_id: str, scored_df: pd.DataFrame):
    """
    Save per-client predictions for a given run.
    top_reasons and top_actions are stored as JSON strings.
    """
    conn = get_connection()
    cur = conn.cursor()
    for _, row in scored_df.iterrows():
        cur.execute(
            """INSERT OR REPLACE INTO predictions
               (run_id, client_id, risk, risk_level, top_reasons, top_actions, monthly_revenue, plan_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                row["client_id"],
                float(row["risk"]),
                row["risk_level"],
                json.dumps(row.get("top_reasons", [])),
                json.dumps(row.get("top_actions", [])),
                float(row["monthly_revenue"]),
                row["plan_type"],
            ),
        )
    conn.commit()
    conn.close()


def load_predictions(run_id: str) -> pd.DataFrame:
    """Load all predictions for a run as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM predictions WHERE run_id = ?", conn, params=(run_id,)
    )
    conn.close()
    # Parse JSON columns back to lists
    if not df.empty:
        df["top_reasons"] = df["top_reasons"].apply(json.loads)
        df["top_actions"] = df["top_actions"].apply(json.loads)
    return df


def latest_run_id() -> str | None:
    """Return the most recent run_id, or None if no runs exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return row["run_id"] if row else None


def previous_run_id(run_id: str) -> str | None:
    """Return the run that came just before run_id, or None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT run_id FROM runs WHERE run_id < ? ORDER BY run_id DESC LIMIT 1",
        (run_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row["run_id"] if row else None


def list_runs() -> list[dict]:
    """Return all runs, newest first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT run_id, run_date, n_clients FROM runs ORDER BY run_id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def save_feedback(run_id: str, client_id: str, outcome: str, note: str = ""):
    """Insert one feedback row."""
    conn = get_connection()
    cur = conn.cursor()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO feedback (run_id, client_id, outcome, note, created_at) VALUES (?, ?, ?, ?, ?)",
        (run_id, client_id, outcome, note, created_at),
    )
    conn.commit()
    conn.close()


def load_feedback(run_id: str | None = None) -> pd.DataFrame:
    """Load feedback rows, optionally filtered by run_id."""
    conn = get_connection()
    if run_id:
        df = pd.read_sql(
            "SELECT * FROM feedback WHERE run_id = ?", conn, params=(run_id,)
        )
    else:
        df = pd.read_sql("SELECT * FROM feedback", conn)
    conn.close()
    return df


def clear_usage():
    """Delete all rows in usage_monthly. Used for testing and reset."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM usage_monthly")
    conn.commit()
    conn.close()
