from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "student_performance_model.pkl"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "student_performance.csv"
)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture(scope="module")
def model_artifact():
    """Load the trained model artifact."""

    if not MODEL_PATH.exists():
        pytest.fail(
            f"Trained model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    assert isinstance(artifact, dict), (
        "The model artifact should be a dictionary."
    )

    return artifact


@pytest.fixture(scope="module")
def dataset():
    """Load the student performance dataset."""

    if not DATA_PATH.exists():
        pytest.fail(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Test 1: Model artifact contains required information
# --------------------------------------------------

def test_model_artifact_structure(model_artifact):
    assert "model" in model_artifact
    assert "features" in model_artifact
    assert "metrics" in model_artifact

    assert model_artifact["model"] is not None
    assert len(model_artifact["features"]) > 0


# --------------------------------------------------
# Test 2: Dataset has the required columns
# --------------------------------------------------

def test_dataset_has_required_columns(
    model_artifact,
    dataset,
):
    required_columns = set(
        model_artifact["features"]
    )

    required_columns.add("final_score")

    assert required_columns.issubset(
        set(dataset.columns)
    ), (
        "Dataset is missing one or more columns "
        "required by the trained model."
    )


# --------------------------------------------------
# Test 3: Dataset contains valid records
# --------------------------------------------------

def test_dataset_contains_valid_records(dataset):
    assert len(dataset) > 0

    assert dataset["final_score"].notna().all()

    assert dataset["final_score"].between(
        0,
        100,
    ).all()

    assert dataset["study_hours"].notna().all()

    assert dataset["attendance"].between(
        0,
        100,
    ).all()


# --------------------------------------------------
# Test 4: Model produces valid predictions
# --------------------------------------------------

def test_model_produces_valid_predictions(
    model_artifact,
    dataset,
):
    model = model_artifact["model"]
    features = model_artifact["features"]

    sample = dataset[features].head(10)

    predictions = np.asarray(
        model.predict(sample),
        dtype=float,
    )

    assert len(predictions) == len(sample)

    assert np.isfinite(predictions).all(), (
        "Model returned a non-finite prediction."
    )

    assert (predictions >= 0).all(), (
        "Model returned a negative predicted score."
    )

    assert (predictions <= 100).all(), (
        "Model returned a score greater than 100."
    )


# --------------------------------------------------
# Test 5: Evaluation metrics are valid
# --------------------------------------------------

def test_model_evaluation_metrics(model_artifact):
    metrics = model_artifact["metrics"]

    for metric_name in ("mae", "rmse", "r2"):
        assert metric_name in metrics, (
            f"Missing model evaluation metric: {metric_name}"
        )

        metric_value = float(
            metrics[metric_name]
        )

        assert math.isfinite(metric_value), (
            f"{metric_name} must be a finite number."
        )

    assert float(metrics["mae"]) >= 0

    assert float(metrics["rmse"]) >= 0


# --------------------------------------------------
# Test 6: Training and testing records are recorded
# --------------------------------------------------

def test_model_training_metadata(model_artifact):
    training_records = model_artifact.get(
        "training_records"
    )

    testing_records = model_artifact.get(
        "testing_records"
    )

    assert isinstance(training_records, int)
    assert isinstance(testing_records, int)

    assert training_records > 0
    assert testing_records > 0

    assert (
        training_records + testing_records
    ) <= 1000

    