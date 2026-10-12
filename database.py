import sqlite3
from pathlib import Path
from datetime import datetime

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "data"
DATABASE_PATH = DATABASE_DIR / "predictions.db"


def get_connection():
    """Create a connection to the SQLite database."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """Create the prediction history table if it does not exist."""
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


def save_prediction(
    study_hours,
    attendance,
    previous_score,
    assignment_completion,
    sleep_hours,
    predicted_score,
    performance,
):
    """Save a prediction permanently in the database."""
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

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


def get_all_predictions():
    """Retrieve all saved predictions, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]

def delete_all_predictions():
    """Delete all saved predictions from the database."""
    with get_connection() as connection:
        connection.execute("DELETE FROM predictions")