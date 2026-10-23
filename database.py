import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st


# --------------------------------------------------
# Local SQLite configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "data"
DATABASE_PATH = DATABASE_DIR / "predictions.db"


# --------------------------------------------------
# Supabase configuration
# --------------------------------------------------

def _supabase_configured():
    """Check whether Supabase credentials are configured."""

    try:
        settings = st.secrets.get("supabase", {})

        return bool(
            settings.get("url")
            and settings.get("key")
        )

    except Exception:
        return False


def using_supabase():
    """Return True when the application uses Supabase."""

    return _supabase_configured()


def _get_current_user_id():
    """Return the logged-in user's ID."""

    if not _supabase_configured():
        return None

    user_id = st.session_state.get("supabase_user_id")

    if not user_id:
        raise PermissionError(
            "Please log in before accessing prediction history."
        )

    return str(user_id)


def _get_supabase_client():
    """Use the authenticated client belonging to this session."""

    if not _supabase_configured():
        return None

    _get_current_user_id()

    # Reuse the session-specific client from auth.py.
    from auth import get_auth_client

    return get_auth_client()


# --------------------------------------------------
# SQLite connection
# --------------------------------------------------

def get_connection():
    """Connect to the local SQLite database."""

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


# --------------------------------------------------
# Initialize database
# --------------------------------------------------

def initialize_database():
    """
    Initialize SQLite for local development.

    Supabase tables are managed through the Supabase dashboard.
    """

    if _supabase_configured():
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
    """Save a prediction to the appropriate database."""

    client = _get_supabase_client()

    if client is not None:

        user_id = _get_current_user_id()

        record = {
            "user_id": user_id,
            "study_hours": float(study_hours),
            "attendance": float(attendance),
            "previous_score": float(previous_score),
            "assignment_completion": float(
                assignment_completion
            ),
            "sleep_hours": float(sleep_hours),
            "predicted_score": float(predicted_score),
            "performance": str(performance),
        }

        response = (
            client
            .table("predictions")
            .insert(record)
            .select("id")
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "The prediction could not be confirmed "
                "as saved in Supabase."
            )

        return int(response.data[0]["id"])

    # Local SQLite mode
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
    """Retrieve only the current user's predictions."""

    client = _get_supabase_client()

    if client is not None:

        user_id = _get_current_user_id()

        response = (
            client
            .table("predictions")
            .select("*")
            .eq("user_id", user_id)
            .order("id", desc=True)
            .execute()
        )

        return response.data or []

    # Local SQLite mode
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
    """
    Allow clearing local SQLite history only.

    Public cloud users must not clear shared records.
    """

    if _supabase_configured():
        raise RuntimeError(
            "Clearing cloud prediction history is disabled."
        )

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM predictions"
        )