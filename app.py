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

if st.button(
    "➕ Save Current Prediction",
    key="save_prediction_btn",
):
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

        st.session_state["prediction_save_message"] = (
            f"Prediction #{prediction_id} saved successfully!"
        )
        st.rerun()

    except Exception as error:
        st.error(f"Could not save prediction: {error}")


# Display the save confirmation after rerunning.
if st.session_state.get("prediction_save_message"):
    st.success(st.session_state["prediction_save_message"])
    st.session_state["prediction_save_message"] = None


# --------------------------------------------------
# Load and Display Prediction History
# --------------------------------------------------

try:
    saved_predictions = database.get_all_predictions()

    if saved_predictions:
        history_df = pd.DataFrame(saved_predictions)

        history_df = history_df.rename(
            columns={"performance": "performance_category"}
        )

        st.subheader("🔎 Search and Filter Predictions")

        search_id = st.text_input(
            "Search by Prediction ID",
            placeholder="Enter an ID, e.g. 5",
            key="search_prediction_id",
        )

        categories = ["All"] + sorted(
            history_df["performance_category"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_category = st.selectbox(
            "Filter by Performance Category",
            categories,
            key="filter_prediction_category",
        )

        sort_option = st.selectbox(
            "Sort Predictions By",
            [
                "Newest First",
                "Oldest First",
                "Highest Predicted Score",
                "Lowest Predicted Score",
            ],
            key="sort_prediction_history",
        )

        filtered_df = history_df.copy()

        if search_id.strip():
            filtered_df = filtered_df[
                filtered_df["id"]
                .astype(str)
                .str.contains(search_id.strip(), regex=False)
            ]

        if selected_category != "All":
            filtered_df = filtered_df[
                filtered_df["performance_category"] == selected_category
            ]

        if sort_option == "Newest First":
            filtered_df = filtered_df.sort_values(
                "created_at", ascending=False
            )
        elif sort_option == "Oldest First":
            filtered_df = filtered_df.sort_values(
                "created_at", ascending=True
            )
        elif sort_option == "Highest Predicted Score":
            filtered_df = filtered_df.sort_values(
                "predicted_score", ascending=False
            )
        else:
            filtered_df = filtered_df.sort_values(
                "predicted_score", ascending=True
            )

        st.caption(
            f"Showing {len(filtered_df)} of "
            f"{len(history_df)} saved predictions."
        )

        st.dataframe(
            filtered_df,
            width="stretch",
            hide_index=True,
        )

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Filtered Predictions (CSV)",
            data=csv_data,
            file_name="student_prediction_history.csv",
            mime="text/csv",
            key="download_filtered_predictions",
        )

        if st.button(
            "🗑️ Clear Prediction History",
            key="clear_prediction_history_btn",
        ):
            database.delete_all_predictions()
            st.session_state["prediction_save_message"] = None
            st.rerun()

    else:
        st.info(
            "No predictions saved yet. Adjust the sidebar inputs "
            "and click 'Save Current Prediction'."
        )

except Exception as error:
    st.error(f"Could not load prediction history: {error}")


# --------------------------------------------------
# Student Performance Comparison Dashboard
# --------------------------------------------------

st.divider()

st.header("📊 Student Performance Comparison Dashboard")

st.write(
    "Compare saved predictions, explore performance patterns, "
    "and download useful reports from your prediction history."
)

st.caption(
    "This dashboard compares saved prediction records by ID. "
    "The scores are model estimates, not actual examination results."
)

# --------------------------------------------------
# Load all saved predictions
# --------------------------------------------------

dashboard_load_error = None

try:
    comparison_all_df = pd.DataFrame(
        database.get_all_predictions()
    )
except Exception as error:
    dashboard_load_error = error
    comparison_all_df = pd.DataFrame()

if dashboard_load_error is not None:

    st.error(
        f"Could not load comparison dashboard data: "
        f"{dashboard_load_error}"
    )

elif comparison_all_df.empty:

    st.info(
        "No saved predictions are available yet. "
        "Generate and save at least one prediction to get started."
    )

else:

    # Prepare the database records for analysis.
    comparison_all_df = comparison_all_df.rename(
        columns={"performance": "performance_category"}
    )

    numeric_columns = [
        "id",
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_completion",
        "sleep_hours",
        "predicted_score",
    ]

    for column in numeric_columns:
        comparison_all_df[column] = pd.to_numeric(
            comparison_all_df[column],
            errors="coerce",
        )

    comparison_all_df = comparison_all_df.dropna(
        subset=["id", "predicted_score"]
    ).copy()

    comparison_all_df["id"] = (
        comparison_all_df["id"].astype(int)
    )

    comparison_all_df["performance_category"] = (
        comparison_all_df["performance_category"]
        .fillna("Unknown")
        .astype(str)
    )

    # --------------------------------------------------
    # Part 1: Overall insights
    # --------------------------------------------------

    st.subheader("📈 Saved Prediction Overview")

    total_predictions = len(comparison_all_df)

    average_score = comparison_all_df[
        "predicted_score"
    ].mean()

    average_attendance = comparison_all_df[
        "attendance"
    ].mean()

    average_study_hours = comparison_all_df[
        "study_hours"
    ].mean()

    highest_prediction = comparison_all_df.loc[
        comparison_all_df["predicted_score"].idxmax()
    ]

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Total Saved Predictions",
            f"{total_predictions:,}",
        )

    with metric2:
        st.metric(
            "Average Predicted Score",
            f"{average_score:.2f}/100",
        )

    with metric3:
        st.metric(
            "Average Attendance",
            f"{average_attendance:.1f}%",
        )

    with metric4:
        st.metric(
            "Average Study Hours",
            f"{average_study_hours:.1f} hrs/day",
        )

    st.info(
        f"The highest saved predicted score is "
        f"{highest_prediction['predicted_score']:.2f}/100 "
        f"(Prediction #{highest_prediction['id']})."
    )

    # --------------------------------------------------
    # Part 2: Performance distribution
    # --------------------------------------------------

    st.subheader("📊 Performance Category Distribution")

    category_order = [
        "Needs Improvement",
        "Average",
        "Good",
        "Excellent",
    ]

    existing_categories = (
        comparison_all_df["performance_category"]
        .unique()
        .tolist()
    )

    all_categories = category_order + [
        category
        for category in sorted(existing_categories)
        if category not in category_order
    ]

    distribution_df = pd.DataFrame(
        {
            "performance_category": all_categories,
            "prediction_count": [
                int(
                    (
                        comparison_all_df[
                            "performance_category"
                        ] == category
                    ).sum()
                )
                for category in all_categories
            ],
        }
    )

    st.bar_chart(
        distribution_df,
        x="performance_category",
        y="prediction_count",
        width="stretch",
    )

    st.caption(
        "This chart counts saved predictions in each category. "
        "It does not represent the actual performance of a "
        "student population."
    )

    # --------------------------------------------------
    # Part 3: Select predictions to compare
    # --------------------------------------------------

    st.subheader("🔎 Compare Multiple Predictions")

    st.write(
        "Choose the saved prediction records you want to compare. "
        "The three most recent records are selected by default."
    )

    prediction_labels = {
        int(row["id"]): (
            f"Prediction #{int(row['id'])} — "
            f"{row['predicted_score']:.1f}/100"
        )
        for _, row in comparison_all_df.iterrows()
    }

    prediction_ids = list(prediction_labels.keys())

    selected_ids = st.multiselect(
        "Select prediction records",
        options=prediction_ids,
        default=prediction_ids[:min(3, len(prediction_ids))],
        format_func=lambda prediction_id: (
            prediction_labels[prediction_id]
        ),
        key="comparison_selected_prediction_ids",
    )

    selected_comparison_df = comparison_all_df[
        comparison_all_df["id"].isin(selected_ids)
    ].sort_values("id").copy()

    if selected_comparison_df.empty:

        st.warning(
            "Select at least one prediction to view the comparison."
        )

    else:

        # --------------------------------------------------
        # Part 4: Comparison metrics
        # --------------------------------------------------

        st.subheader("📌 Selected Prediction Summary")

        selected_average = selected_comparison_df[
            "predicted_score"
        ].mean()

        selected_highest = selected_comparison_df.loc[
            selected_comparison_df["predicted_score"].idxmax()
        ]

        selected_lowest = selected_comparison_df.loc[
            selected_comparison_df["predicted_score"].idxmin()
        ]

        compare_col1, compare_col2, compare_col3 = st.columns(3)

        with compare_col1:
            st.metric(
                "Records Selected",
                len(selected_comparison_df),
            )

        with compare_col2:
            st.metric(
                "Average Predicted Score",
                f"{selected_average:.2f}/100",
            )

        with compare_col3:
            st.metric(
                "Highest Selected Score",
                f"{selected_highest['predicted_score']:.2f}/100",
                help=(
                    f"Prediction #{selected_highest['id']}"
                ),
            )

        # --------------------------------------------------
        # Part 5: Interactive comparison charts
        # --------------------------------------------------

        st.subheader("📉 Study Habits and Predicted Scores")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.markdown("**Study Hours vs Predicted Score**")

            st.scatter_chart(
                selected_comparison_df,
                x="study_hours",
                y="predicted_score",
                color="performance_category",
                size="attendance",
                width="stretch",
            )

        with chart_col2:

            st.markdown("**Attendance vs Predicted Score**")

            st.scatter_chart(
                selected_comparison_df,
                x="attendance",
                y="predicted_score",
                color="performance_category",
                size="study_hours",
                width="stretch",
            )

        # --------------------------------------------------
        # Part 6: Detailed comparison table
        # --------------------------------------------------

        st.subheader("📋 Detailed Prediction Comparison")

        display_columns = [
            "id",
            "study_hours",
            "attendance",
            "previous_score",
            "assignment_completion",
            "sleep_hours",
            "predicted_score",
            "performance_category",
            "created_at",
        ]

        comparison_table = selected_comparison_df[
            display_columns
        ].copy()

        comparison_table = comparison_table.round(
            {
                "study_hours": 1,
                "attendance": 1,
                "previous_score": 1,
                "assignment_completion": 1,
                "sleep_hours": 1,
                "predicted_score": 2,
            }
        )

        comparison_table = comparison_table.rename(
            columns={
                "id": "prediction_id",
                "performance_category": "performance",
            }
        )

        st.dataframe(
            comparison_table,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------
    # Part 7: Downloadable reports
    # --------------------------------------------------

    st.subheader("📥 Download Comparison Reports")

    download_col1, download_col2, download_col3 = st.columns(3)

    with download_col1:

        st.download_button(
            label="Selected Comparison CSV",
            data=selected_comparison_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="student_prediction_comparison.csv",
            mime="text/csv",
            disabled=selected_comparison_df.empty,
            key="download_comparison_dashboard_csv",
        )

    with download_col2:

        st.download_button(
            label="All Saved Predictions CSV",
            data=comparison_all_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="all_student_predictions.csv",
            mime="text/csv",
            key="download_all_dashboard_predictions_csv",
        )

    with download_col3:

        st.download_button(
            label="Category Distribution CSV",
            data=distribution_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="student_performance_distribution.csv",
            mime="text/csv",
            key="download_category_distribution_csv",
        )

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

