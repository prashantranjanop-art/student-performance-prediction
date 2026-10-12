import database
database.initialize_database()

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)


# --------------------------------------------------
# Load model and dataset
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


artifact = load_model()
data = load_dataset()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict expected academic performance using study habits, "
    "attendance, previous scores, and assignment completion."
)

st.info(
    "This educational demonstration uses a model trained on "
    "synthetic data. Predictions are estimates, not guarantees "
    "of a student's actual performance."
)


if artifact is None:
    st.error(
        "Trained model not found. Run "
        "'python train_model.py' before starting the application."
    )
    st.stop()


model = artifact["model"]
features = artifact["features"]


# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------

st.sidebar.header("📝 Student Information")

study_hours = st.sidebar.slider(
    "Daily study hours",
    min_value=0.5,
    max_value=8.0,
    value=4.0,
    step=0.5,
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=50,
    max_value=100,
    value=80,
    step=1,
)

previous_score = st.sidebar.slider(
    "Previous score",
    min_value=30,
    max_value=100,
    value=70,
    step=1,
)

assignment_completion = st.sidebar.slider(
    "Assignment completion (%)",
    min_value=40,
    max_value=100,
    value=75,
    step=1,
)

sleep_hours = st.sidebar.slider(
    "Average sleep hours",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.5,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

input_data = pd.DataFrame(
    [
        {
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "assignment_completion": assignment_completion,
            "sleep_hours": sleep_hours,
        }
    ],
    columns=features,
)

predicted_score = float(model.predict(input_data)[0])
predicted_score = max(0.0, min(100.0, predicted_score))


def get_performance_category(score):
    if score < 40:
        return "Needs Improvement"
    elif score < 60:
        return "Average"
    elif score < 80:
        return "Good"
    return "Excellent"


category = get_performance_category(predicted_score)


# --------------------------------------------------
# Prediction results
# --------------------------------------------------

st.subheader("📊 Prediction Results")
st.caption(
    "The score is a model estimate and may differ from actual results."
)

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

st.subheader("🤖 Model Performance")

metrics = artifact.get("metrics", {})

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    mae = metrics.get("mae")
    st.metric(
        "Mean Absolute Error (MAE)",
        f"{mae:.2f}" if mae is not None else "N/A",
    )

with metric_col2:
    rmse = metrics.get("rmse")
    st.metric(
        "Root Mean Squared Error (RMSE)",
        f"{rmse:.2f}" if rmse is not None else "N/A",
    )

with metric_col3:
    r2 = metrics.get("r2")
    st.metric(
        "R² Score",
        f"{r2:.3f}" if r2 is not None else "N/A",
    )

training_records = artifact.get("training_records")
testing_records = artifact.get("testing_records")

record_col1, record_col2 = st.columns(2)

with record_col1:
    st.metric(
        "Training Records",
        f"{training_records:,}" if training_records is not None else "N/A",
    )

with record_col2:
    st.metric(
        "Testing Records",
        f"{testing_records:,}" if testing_records is not None else "N/A",
    )

st.caption(
    "Metrics were calculated on a held-out test set from synthetic data. "
    "They describe performance on this dataset and do not guarantee "
    "accuracy on real student outcomes."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Final Score",
        f"{predicted_score:.2f}/100",
    )

with col2:
    st.metric(
        "Performance Category",
        category,
    )

with col3:
    st.metric(
        "Daily Study Hours",
        f"{study_hours:.1f} hours",
    )

st.progress(predicted_score / 100)

st.caption(
    "The score is a model estimate and may differ from actual results."
)

# --------------------------------------------------
# Permanent Prediction History - SQLite
# --------------------------------------------------

st.subheader("📋 Prediction History")

if st.button("➕ Save Current Prediction"):
    try:
        prediction_id = database.save_prediction(
            study_hours=study_hours,
            attendance=attendance,
            previous_score=previous_score,
            assignment_completion=assignment_completion,
            sleep_hours=sleep_hours,
            predicted_score=round(float(predicted_score), 2),
            performance=category,
        )

        st.success(
            f"Prediction #{prediction_id} saved permanently!"
        )
        st.rerun()

    except Exception as error:
        st.error(f"Could not save prediction: {error}")


# Load saved predictions from SQLite
try:
    saved_predictions = database.get_all_predictions()

    if saved_predictions:
        history_df = pd.DataFrame(saved_predictions)

        # Use a clearer column name in the dashboard
        history_df = history_df.rename(
            columns={"performance": "performance_category"}
        )

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
        )

        # Download prediction history
        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction History (CSV)",
            data=csv_data,
            file_name="student_prediction_history.csv",
            mime="text/csv",
        )

        # Clear permanently saved history
        if st.button("🗑️ Clear Prediction History"):
            database.delete_all_predictions()
            st.success("Prediction history cleared successfully!")
            st.rerun()

    else:
        st.info(
            "No predictions saved yet. Adjust the student information "
            "in the sidebar and click 'Save Current Prediction'."
        )

except Exception as error:
    st.error(f"Could not load prediction history: {error}")

# --------------------------------------------------
# Personalized suggestions
# --------------------------------------------------

st.subheader("💡 Suggestions")

suggestions = []

if study_hours < 3:
    suggestions.append(
        "Consider establishing a consistent study routine "
        "with manageable daily goals."
    )

if attendance < 75:
    suggestions.append(
        "Improving class attendance may help you keep up "
        "with lessons and assignments."
    )

if previous_score < 50:
    suggestions.append(
        "Review foundational concepts and practise topics "
        "that have been difficult."
    )

if assignment_completion < 70:
    suggestions.append(
        "Try planning assignments ahead of their deadlines "
        "and tracking your progress."
    )

if sleep_hours < 6:
    suggestions.append(
        "Consider maintaining a regular sleep schedule "
        "that supports your learning and wellbeing."
    )

if not suggestions:
    suggestions.append(
        "Continue building consistent study habits and "
        "review your progress regularly."
    )

for suggestion in suggestions:
    st.write(f"• {suggestion}")


# --------------------------------------------------
# Dataset analytics
# --------------------------------------------------

st.divider()
st.header("📈 Student Dataset Analytics")

if data is not None:
    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Dataset Records", f"{len(data):,}")

    with metric2:
        st.metric(
            "Average Final Score",
            f"{data['final_score'].mean():.2f}",
        )

    with metric3:
        st.metric(
            "Average Attendance",
            f"{data['attendance'].mean():.1f}%",
        )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Final Score Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            data["final_score"],
            bins=20,
            edgecolor="black",
        )

        ax.set_xlabel("Final Score")
        ax.set_ylabel("Number of Students")
        ax.set_title("Distribution of Student Scores")

        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.subheader("Study Hours vs Final Score")

        fig, ax = plt.subplots()

        ax.scatter(
            data["study_hours"],
            data["final_score"],
            alpha=0.5,
        )

        ax.set_xlabel("Daily Study Hours")
        ax.set_ylabel("Final Score")
        ax.set_title("Study Hours and Final Scores")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Sample Dataset")
    st.dataframe(data.head(20), width="stretch")

else:
    st.warning(
        "Dataset not found. Generate it using "
        "'python src/generate_data.py'."
    )

# --------------------------------------------------
# Actual vs Predicted Score Analysis
# --------------------------------------------------

st.divider()
st.header("🎯 Actual vs Predicted Score Analysis")

if data is not None:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    features = artifact["features"]

    X = data[features]
    y = data["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    test_predictions = model.predict(X_test)

    comparison_df = pd.DataFrame({
        "Actual Score": y_test.to_numpy(),
        "Predicted Score": test_predictions,
    })

    comparison_df["Prediction Error"] = (
        comparison_df["Actual Score"]
        - comparison_df["Predicted Score"]
    )

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
        st.metric(
            "Test Dataset MAE",
            f"{mean_absolute_error(y_test, test_predictions):.2f}",
        )

    with comparison_col2:
        st.metric(
            "Records Compared",
            f"{len(comparison_df):,}",
        )

    fig, ax = plt.subplots()

    ax.scatter(
        comparison_df["Actual Score"],
        comparison_df["Predicted Score"],
        alpha=0.6,
    )

    ax.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
    )

    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Actual vs Predicted Scores")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Points closer to the diagonal line represent predictions "
        "closer to the actual scores. This evaluation uses synthetic data."
    )

    with st.expander("View sample prediction comparisons"):
        st.dataframe(
            comparison_df.head(20).round(2),
            width="stretch",
        )

else:
    st.warning(
        "Generate the dataset before viewing prediction comparisons."
    )



# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Student Performance Predictor | Machine Learning Portfolio Project"
)

