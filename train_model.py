from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "student_performance_model.pkl"


def train_model():
    """Train and evaluate the student performance prediction model."""

    # Check whether the dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run 'python src/generate_data.py' first."
        )

    # Load dataset
    data = pd.read_csv(DATA_PATH)

    # Define features and target
    features = [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_completion",
        "sleep_hours",
    ]

    X = data[features]
    y = data["final_score"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Create the model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    # Train
    print("Training the Random Forest model...")
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation")
    print("-" * 35)
    print(f"Training records: {len(X_train)}")
    print(f"Testing records:  {len(X_test)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

    # Save model, metadata, and evaluation metrics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "features": features,
        "target": "final_score",
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
    }

    joblib.dump(artifact, MODEL_PATH)



    print(f"\nModel saved successfully:\n{MODEL_PATH}")

    # Show sample predictions
    sample = X_test.head(5).copy()
    sample["actual_score"] = y_test.loc[sample.index]
    sample["predicted_score"] = model.predict(X_test.head(5)).round(2)

    print("\nSample Predictions")
    print("-" * 55)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    train_model()

