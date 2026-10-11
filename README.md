# 🎓 AI-Based Student Performance Prediction

An interactive machine learning application that estimates student final scores using study habits, attendance, previous academic scores, assignment completion, and sleep duration.

Built with Python, Scikit-learn, Pandas, Matplotlib, and Streamlit.

> **Disclaimer:** This educational project uses synthetic data generated for demonstration purposes. Its results do not establish accuracy on real student records or guarantee actual academic outcomes.

## 🚀 Features

* **Score prediction:** Estimate a final score from student-related input features.
* **Performance categories:** Classify predictions as Needs Improvement, Average, Good, or Excellent.
* **Personalized suggestions:** Display study-habit suggestions based on entered information.
* **Interactive dashboard:** Adjust inputs through Streamlit controls.
* **Dataset analytics:** Explore score distributions, attendance, and study-hour relationships.
* **Model evaluation:** Display MAE, RMSE, and R² calculated on a held-out test dataset.
* **Actual vs. predicted analysis:** Visualize predictions against actual synthetic test scores.
* **Prediction history:** Save multiple predictions during a session.
* **CSV export:** Download saved predictions for further analysis.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib
* Git and GitHub

## 📁 Project Structure

```text
student-performance-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── student_performance.csv
├── models/
│   └── student_performance_model.pkl
└── src/
    ├── __init__.py
    └── generate_data.py
```

## 📊 Dataset

The project uses a generated dataset containing 1,000 synthetic records.

| Feature                 | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `study_hours`           | Daily study hours                                   |
| `attendance`            | Attendance percentage                               |
| `previous_score`        | Previous academic score                             |
| `assignment_completion` | Percentage of assignments completed                 |
| `sleep_hours`           | Average daily sleep duration                        |
| `final_score`           | Generated final score used as the prediction target |
| `performance`           | Performance category derived from the final score   |

The dataset is synthetic and should not be treated as evidence of real-world academic performance.

## 🤖 Machine Learning Model

The application uses a Random Forest Regressor from Scikit-learn.

### Model Configuration

* Number of estimators: 200
* Maximum depth: 12
* Test set size: 20%
* Random state: 42

### Evaluation Results

Results from the current saved model:

| Metric                         | Result |
| ------------------------------ | -----: |
| Mean Absolute Error (MAE)      |   4.29 |
| Root Mean Squared Error (RMSE) |   5.48 |
| R² Score                       |  0.810 |

These metrics were calculated on a held-out synthetic test set. They can change when the dataset or model configuration changes.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/prashantranjanop-art/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can invoke the environment's Python directly using `.venv\Scripts\python.exe`.

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Generate the dataset if it is missing

```powershell
python src/generate_data.py
```

### 6. Train the model if the saved model is missing

```powershell
python train_model.py
```

### 7. Start the application

```powershell
python -m streamlit run app.py
```

Open the local address displayed in the terminal to use the dashboard.

## 🧪 Model Training

To retrain the model after generating the dataset:

```powershell
python train_model.py
```

The script evaluates the model and saves the trained model and its metadata to:

```text
models/student_performance_model.pkl
```

## 🔮 Future Improvements

* Deploy the application to a public hosting platform.
* Add automated tests and input validation.
* Improve experiment tracking and model comparison.
* Evaluate performance using suitable real-world data, where authorized and ethically collected.
* Add persistent prediction storage if needed.

## 👨‍💻 Author

Developed as an AI and Machine Learning portfolio project.

## 📜 License

No license has been specified yet. Add a suitable open-source license if you want others to reuse or modify this project.

````

## 3. Save and check the README

1. Press **Ctrl + S**.
2. Review the file in VS Code.
3. Check that the project structure and commands match your repository.

The README describes the features we've added so far and explains that the dataset is synthetic.

## 4. Upload the README to GitHub

Once you've saved and checked it, run these commands one at a time:

**Check the changes:**
```powershell
git status
````

**Stage the README:**

```powershell
git add README.md
```

**Commit it:**

```powershell
git commit -m "Improve project documentation"
```

**Push it:**

```powershell
git push origin main
```

Your repository is here:

[Student Performance Prediction on GitHub](https://github.com/prashantranjanop-art/student-performance-prediction?utm_source=chatgpt.com)

### Your next milestone 🚀

After the README is uploaded, we'll move on to **Step 8: Add screenshots of your working dashboard to GitHub**. This helps visitors understand the application before they install or run it.

First, finish the README update and tell me when it has been pushed successfully.
