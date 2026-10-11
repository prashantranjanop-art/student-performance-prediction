from pathlib import Path

import numpy as np
import pandas as pd


def generate_student_data(
    number_of_students=1000,
    random_state=42,
):
    """Generate synthetic student performance data."""

    rng = np.random.default_rng(random_state)

    study_hours = rng.uniform(0.5, 8.0, number_of_students)
    attendance = rng.uniform(50, 100, number_of_students)
    previous_score = rng.uniform(30, 100, number_of_students)
    assignment_completion = rng.uniform(40, 100, number_of_students)
    sleep_hours = rng.uniform(4, 10, number_of_students)

    noise = rng.normal(0, 5, number_of_students)

    final_score = (
        0.30 * study_hours / 8 * 100
        + 0.20 * attendance
        + 0.25 * previous_score
        + 0.20 * assignment_completion
        + 0.05 * sleep_hours / 10 * 100
        + noise
    )

    final_score = np.clip(final_score, 0, 100)

    performance = pd.cut(
        final_score,
        bins=[-0.01, 39.99, 59.99, 79.99, 100],
        labels=["Needs Improvement", "Average", "Good", "Excellent"],
    )

    return pd.DataFrame(
        {
            "study_hours": study_hours.round(2),
            "attendance": attendance.round(2),
            "previous_score": previous_score.round(2),
            "assignment_completion": assignment_completion.round(2),
            "sleep_hours": sleep_hours.round(2),
            "final_score": final_score.round(2),
            "performance": performance,
        }
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_student_data()
    output_path = data_dir / "student_performance.csv"

    dataset.to_csv(output_path, index=False)

    print(f"Dataset created: {output_path}")
    print(f"Number of students: {len(dataset)}")
    print("\nFirst five records:")
    print(dataset.head())
    print("\nPerformance distribution:")
    print(dataset["performance"].value_counts().sort_index())
    print("\nMissing values:")
    print(dataset.isnull().sum())
