# Student Performance Factors — Regression Analysis

A machine learning project that analyzes and predicts students' exam scores based on academic, behavioral, and socio-economic factors, using the [Student Performance Factors dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).

## 📌 Project Overview

This project explores which factors most influence student exam performance and builds regression models to predict `Exam_Score` from 19 input features covering study habits, parental involvement, school environment, and personal well-being.

The workflow covers the full data science pipeline:
1. Data loading & exploration
2. Data cleaning (missing values, outliers, duplicates)
3. Feature engineering (standardization, one-hot encoding, ordinal encoding)
4. Exploratory data analysis & correlation analysis
5. Model training & evaluation
6. Model export for deployment

## 📂 Dataset

- **File:** `StudentPerformanceFactors.csv`
- **Rows:** 6,607 students
- **Columns:** 20 (19 features + 1 target)

**Features include:**

| Category | Columns |
|---|---|
| Study habits | `Hours_Studied`, `Attendance`, `Tutoring_Sessions`, `Previous_Scores` |
| Environment & support | `Parental_Involvement`, `Access_to_Resources`, `Teacher_Quality`, `School_Type`, `Peer_Influence` |
| Lifestyle | `Sleep_Hours`, `Physical_Activity`, `Extracurricular_Activities` |
| Socio-economic | `Family_Income`, `Internet_Access`, `Distance_from_Home`, `Parental_Education_Level` |
| Personal | `Motivation_Level`, `Learning_Disabilities`, `Gender` |
| **Target** | `Exam_Score` |

## 🛠️ Methodology

### Data Preprocessing
- Filled missing categorical values using the **mode**
- Checked for and handled outliers and duplicate rows
- **Standardized** numerical features using `StandardScaler`
- **One-Hot Encoding** for nominal categorical features (`Extracurricular_Activities`, `Internet_Access`, `School_Type`, `Learning_Disabilities`, `Gender`)
- **Ordinal Encoding** for ordered categorical features (`Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality`, `Peer_Influence`, `Parental_Education_Level`, `Distance_from_Home`)
- Final feature set: **24 columns** after encoding

### Exploratory Data Analysis
- Distribution plots for key numerical features (`Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`)
- Correlation heatmap across all features
- Feature-target correlation ranking against `Exam_Score`

### Modeling
Data was split 80/20 into training and testing sets (`random_state=1`). Three regression algorithms were trained and compared:

| Model | MAE | MSE | R² |
|---|---|---|---|
| Least Angle Regression (LARS) | 0.661 | 0.829 | 0.140 |
| **Linear Regression** | **0.118** | **0.254** | **0.737** |
| Gradient Boosting Regressor | 0.206 | 0.291 | 0.699 |

**Linear Regression** achieved the best performance across all three metrics and was selected as the final model.

## 📁 Repository Structure

```
.
├── Student_Performence_Factors.ipynb   # Full analysis & modeling notebook
├── StudentPerformanceFactors.csv       # Dataset
├── lr_model.joblib                     # Trained Linear Regression model
└── README.md
```

## 🚀 Getting Started

### Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
```

### Usage

**Run the notebook:**
```bash
jupyter notebook Student_Performence_Factors.ipynb
```

**Load the trained model for inference:**
```python
import joblib

model = joblib.load('lr_model.joblib')
predictions = model.predict(X_new)  # X_new must match the 24-column encoded feature format
```

## 📊 Key Insights
- Linear Regression significantly outperformed both LARS and Gradient Boosting on this dataset, suggesting the relationship between the engineered features and `Exam_Score` is largely linear.
- `Hours_Studied`, `Attendance`, and `Previous_Scores` showed strong correlation with exam performance.

## 📝 Notes
- This notebook and its comments are written in **Bahasa Indonesia**; this README is provided in English for accessibility.
- The model expects preprocessed input (standardized numerics + one-hot/ordinal encoded categoricals) matching the 24-feature schema used in training.


