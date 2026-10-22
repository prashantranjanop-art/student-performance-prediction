import sqlite3
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# Local SQLite configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "data"
DATABASE_PATH = DATABASE_DIR / "predictions.db"

# --------------------------------------------------
# Supabase client configuration
# --------------------------------------------------

_supabase_client = None
_supabase_checked = False


def _get_supabase_client():
    """Return the configured Supabase client, if available."""

    global _supabase_client, _supabase_checked

    if _supabase_checked:
        return _supabase_client

    # No configured Supabase secrets? Use local SQLite.
    try:
        import streamlit as st
        settings = st.secrets.get("supabase", {})
    except Exception:
        settings = {}

    if not settings:
        _supabase_checked = True
        return None

    url = str(settings.get("url", "")).strip().rstrip("/")
    key = str(settings.get("key", "")).strip()

    if not url or not key:
        raise RuntimeError(
            "Supabase configuration is incomplete. "
            "Check the url and key in .streamlit/secrets.toml."
        )

    if url.endswith("/rest/v1"):
        raise RuntimeError(
            "Use the base Supabase Project URL without /rest/v1/."
        )

    from supabase import create_client

    _supabase_client = create_client(url, key)
    _supabase_checked = True

    return _supabase_client


def using_supabase():
    """Return True when Supabase is configured."""
    return _get_supabase_client() is not None


# --------------------------------------------------
# SQLite connection
# --------------------------------------------------

def get_connection():
    """Create a connection to the local SQLite database."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


# --------------------------------------------------
# Initialize database
# --------------------------------------------------

def initialize_database():
    """Initialize SQLite when Supabase is not configured."""

    if _get_supabase_client() is not None:
        # The predictions table already exists in Supabase.
        return

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_hours REAL NOT NULL,
                attendance REAL NOT NULL,
                previous_score REAL NOT NULL,
                assignment_completion REAL NOT NULL,
                sleep_hours REAL NOT NULL,
                predicted_score REAL NOT NULL,
                performance TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


# --------------------------------------------------
# Save a prediction
# --------------------------------------------------

def save_prediction(
    study_hours,
    attendance,
    previous_score,
    assignment_completion,
    sleep_hours,
    predicted_score,
    performance,
):
    """Save a prediction to Supabase or local SQLite."""

    client = _get_supabase_client()

    if client is not None:
        record = {
            "study_hours": float(study_hours),
            "attendance": float(attendance),
            "previous_score": float(previous_score),
            "assignment_completion": float(assignment_completion),
            "sleep_hours": float(sleep_hours),
            "predicted_score": float(predicted_score),
            "performance": str(performance),
        }

        response = (
            client.table("predictions")
            .insert(record)
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "Supabase did not return the saved prediction."
            )

        return int(response.data[0]["id"])

    created_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                study_hours,
                attendance,
                previous_score,
                assignment_completion,
                sleep_hours,
                predicted_score,
                performance,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study_hours,
                attendance,
                previous_score,
                assignment_completion,
                sleep_hours,
                predicted_score,
                performance,
                created_at,
            ),
        )

        return cursor.lastrowid


# --------------------------------------------------
# Retrieve prediction history
# --------------------------------------------------

def get_all_predictions():
    """Retrieve all saved predictions, newest first."""

    client = _get_supabase_client()

    if client is not None:
        response = (
            client.table("predictions")
            .select("*")
            .order("id", desc=True)
            .execute()
        )

        return response.data or []

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


# --------------------------------------------------
# Clear local prediction history
# --------------------------------------------------

def delete_all_predictions():
    """Clear local SQLite history only."""

    if _get_supabase_client() is not None:
        raise RuntimeError(
            "Clearing shared cloud history is disabled "
            "to protect saved records."
        )

    with get_connection() as connection:
        connection.execute("DELETE FROM predictions")