from types import SimpleNamespace

import pytest
import database


# --------------------------------------------------
# Test helpers
# --------------------------------------------------

def save_sample_prediction(
    db,
    study_hours=4.0,
    predicted_score=72.0,
    performance="Good",
):
    """Save a test prediction using the database module."""

    return db.save_prediction(
        study_hours=study_hours,
        attendance=80,
        previous_score=70,
        assignment_completion=75,
        sleep_hours=7,
        predicted_score=predicted_score,
        performance=performance,
    )


@pytest.fixture
def sqlite_database(monkeypatch, tmp_path):
    """
    Use a temporary SQLite file instead of the real project database.

    These tests do not modify the application's existing prediction
    history or contact the real Supabase database.
    """

    test_database_path = tmp_path / "test_predictions.db"

    monkeypatch.setattr(
        database,
        "_supabase_configured",
        lambda: False,
    )

    monkeypatch.setattr(
        database,
        "DATABASE_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        test_database_path,
    )

    database.initialize_database()

    return database


# --------------------------------------------------
# Fake Supabase objects for isolated unit tests
# --------------------------------------------------

class FakeQuery:
    """Minimal fake query object for testing database behavior."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.filters = []
        self.inserted_record = None
        self.ordering = None
        self.selected_columns = None

    def insert(self, record):
        self.inserted_record = record
        return self

    def select(self, columns):
        self.selected_columns = columns
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def order(self, column, desc=False):
        self.ordering = (column, desc)
        return self

    def execute(self):
        if self.inserted_record is not None:
            return SimpleNamespace(
                data=[{"id": 42}]
            )

        return SimpleNamespace(
            data=self.rows
        )


class FakeSupabaseClient:
    """Minimal fake client that records requested table operations."""

    def __init__(self, query):
        self.query = query
        self.table_name = None

    def table(self, name):
        self.table_name = name
        return self.query


# --------------------------------------------------
# Test 1: Saving and retrieving a local prediction
# --------------------------------------------------

def test_save_and_retrieve_prediction(sqlite_database):
    prediction_id = save_sample_prediction(
        sqlite_database,
        study_hours=4.5,
        predicted_score=76.25,
        performance="Good",
    )

    records = sqlite_database.get_all_predictions()

    assert len(records) == 1
    assert records[0]["id"] == prediction_id
    assert records[0]["study_hours"] == 4.5
    assert records[0]["predicted_score"] == 76.25
    assert records[0]["performance"] == "Good"
    assert records[0]["created_at"]


# --------------------------------------------------
# Test 2: Newest predictions appear first
# --------------------------------------------------

def test_predictions_are_returned_newest_first(
    sqlite_database,
):
    first_id = save_sample_prediction(
        sqlite_database,
        study_hours=3.0,
    )

    second_id = save_sample_prediction(
        sqlite_database,
        study_hours=6.0,
    )

    records = sqlite_database.get_all_predictions()

    assert [record["id"] for record in records] == [
        second_id,
        first_id,
    ]


# --------------------------------------------------
# Test 3: Local prediction history can be cleared
# --------------------------------------------------

def test_clear_local_prediction_history(
    sqlite_database,
):
    save_sample_prediction(sqlite_database)

    assert len(
        sqlite_database.get_all_predictions()
    ) == 1

    sqlite_database.delete_all_predictions()

    assert sqlite_database.get_all_predictions() == []


# --------------------------------------------------
# Test 4: Cloud predictions include the current user ID
# --------------------------------------------------

def test_supabase_save_includes_user_id(
    monkeypatch,
):
    query = FakeQuery()
    fake_client = FakeSupabaseClient(query)

    monkeypatch.setattr(
        database,
        "_get_supabase_client",
        lambda: fake_client,
    )

    monkeypatch.setattr(
        database,
        "_get_current_user_id",
        lambda: "test-user-123",
    )

    prediction_id = save_sample_prediction(database)

    assert prediction_id == 42
    assert fake_client.table_name == "predictions"

    assert query.inserted_record is not None

    assert (
        query.inserted_record["user_id"]
        == "test-user-123"
    )


# --------------------------------------------------
# Test 5: Cloud history queries filter by user ID
# --------------------------------------------------

def test_supabase_history_filters_current_user(
    monkeypatch,
):
    own_record = {
        "id": 42,
        "user_id": "test-user-123",
        "study_hours": 4.0,
        "predicted_score": 72.0,
        "performance": "Good",
    }

    query = FakeQuery(rows=[own_record])
    fake_client = FakeSupabaseClient(query)

    monkeypatch.setattr(
        database,
        "_get_supabase_client",
        lambda: fake_client,
    )

    monkeypatch.setattr(
        database,
        "_get_current_user_id",
        lambda: "test-user-123",
    )

    records = database.get_all_predictions()

    assert records == [own_record]

    assert (
        "user_id",
        "test-user-123",
    ) in query.filters

    assert fake_client.table_name == "predictions"


# --------------------------------------------------
# Test 6: Clearing cloud history is prohibited
# --------------------------------------------------

def test_cloud_prediction_history_cannot_be_cleared(
    monkeypatch,
):
    monkeypatch.setattr(
        database,
        "_supabase_configured",
        lambda: True,
    )

    with pytest.raises(
        RuntimeError,
        match="Clearing cloud prediction history is disabled",
    ):
        database.delete_all_predictions()