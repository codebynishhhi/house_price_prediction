# 🏡 House Price Prediction – End-to-End ML System

An end-to-end **production-ready Machine Learning project** that predicts house prices using structured housing data.  
The project covers the **complete ML lifecycle** — from data ingestion and feature engineering to model training, inference pipelines, and a FastAPI prediction service.

This project is designed with **real-world ML engineering practices**, not just notebooks.

---

##  Project Highlights

- End-to-end **training + inference pipelines**
- Modular, reusable **ML architecture**
- Handles **partial user input** intelligently
- Full **FastAPI inference service**
- Production-ready **sklearn Pipeline**
- Robust **logging, exception handling, and config management**

---

## Problem Statement

Predict the **sale price of a house** based on numerical and categorical features such as:
- Property size
- Quality indicators
- Location (Neighborhood)
- Year built / renovated
- Garage and basement features

---

## How to Run & Test the Project (Quick Start)
1. Train the Model
python pipeline/train_pipeline.py


This will:
Train multiple models
Select the best model
Save the full production pipeline
Store evaluation metrics

2. Start the FastAPI Server
uvicorn app.main:app --reload

3. Swagger UI (API testing):
http://127.0.0.1:8000/docs

4. Test Prediction via API
POST /predict

Sample Payload

{
  "Overall Qual": 7,
  "Gr Liv Area": 1710,
  "Neighborhood": "NridgHt",
  "Garage Cars": 2,
  "Kitchen Qual": "Gd",
  "Exter Qual": "Gd"
}


Sample Response

{
  "predicted_price": 285432.67
}

5. Test Prediction Locally (Without API)
python pipeline/sample_test_prediction.py


## 🏗️ Project Architecture

house_price_prediction/
│
├── app/
│ └── main.py # FastAPI app
│
├── pipeline/
│ ├── train_pipeline.py # End-to-end training pipeline
│ ├── prediction_pipeline.py # Inference pipeline
│ └── sample_test_prediction.py
│
├── src/
│ ├── data_ingestion.py
│ ├── feature_engineering.py
│ ├── outlier_handling.py
│ ├── data_preprocessing.py
│ ├── encoding.py
│ ├── scaling.py
│ ├── model_training.py
│ ├── model_selection.py
│ ├── model_evaluation.py
│ └── utils/
│ ├── logger.py
│ ├── exception.py
│ └── config.py
│
├── artifacts/
│ ├── model/
│ │ ├── best_model.pkl
│ │ └── full_pipeline.pkl 
│ └── reports/
│ └── model_metrics.json
│
├── data/
│ ├── raw/
│ └── processed/
│
├── requirements.txt
└── README.md


---

## ⚙️ ML Pipeline Overview

### 🔹 1. Data Ingestion
- Loads raw housing data
- Splits into train & test sets
- Saves processed datasets for reproducibility

### 🔹 2. Feature Engineering
Custom domain features:
- `House_Age`
- `Remod_Age`
- `Total_Bathrooms`
- `Total_SF`
- Binary indicators (`Has_Garage`, `Has_Basement`)
- Drops redundant columns

---

### 🔹 3. Outlier Handling
- IQR-based bounds learned **only from training data**
- Applied consistently during inference

---

### 🔹 4. Encoding
- Ordinal encoding for quality-based categorical features
- ColumnTransformer-based architecture

---

### 🔹 5. Imputation
- Numerical → median
- Categorical → most frequent
- Ensures inference stability for missing values

---

### 🔹 6. Scaling
- StandardScaler applied to numerical features
- Fitted on training data only

---

### 🔹 7. Model Training
Trained and evaluated multiple models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

Metrics tracked:
- Train R²
- Test R²
- RMSE

---

### 🔹 8. Model Selection
- Selected based on **test performance + generalization gap**
- Prevents overfitting
- Best model persisted

---

### 🔹 9. Full Production Pipeline
All preprocessing + model steps are wrapped into a single **sklearn Pipeline**:

```python
full_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineering()),
    ("outlier_handler", OutlierHandler()),
    ("encoding", encoder),
    ("imputation", preprocessor),
    ("scaling", scaler),
    ("model", best_model)
])
Saved as:
artifacts/model/full_pipeline.pkl
This ensures:
No training/inference skew
One-line .predict() in production

---

## Model Training Results & Evaluation

Multiple regression models were trained and evaluated on the same train–test split to ensure fair comparison.

### Evaluation Metrics
- **R² Score** – goodness of fit
- **RMSE** – error magnitude
- **Generalization Gap** – |Train R² − Test R²|

---

###  Model Performance Comparison

| Model            | Train R² | Test R² | Train RMSE | Test RMSE | Generalization Gap |
|------------------|----------|---------|------------|-----------|--------------------|
| Linear Regression | 0.9282   | 0.8679  | 20,663     | 32,540    | 0.0603             |
| Ridge Regression  | 0.9281   | 0.8719  | 20,673     | 32,047    | 0.0562             |
| Lasso Regression  | 0.9282   | 0.8680  | 20,663     | 32,538    | 0.0602             |
| **Random Forest** ⭐ | **0.9843** | **0.9265** | **9,656** | **24,282** | **0.0579** |

---

###  Final Model Selection

The **Random Forest Regressor** was selected as the final model because:

- Achieved the **highest Test R² (0.9265)**
- Maintained a **reasonable generalization gap**
- Significantly reduced **RMSE** compared to linear models
- Demonstrated strong non-linear learning capability

To prevent overfitting, model selection logic enforced:
```text
- Maximize Test R²
- Keep generalization gap under control


##  Inference System

🔹 Input Adapter
Accepts partial user input
Automatically fills missing features using:
Training medians (numerical)
Training modes (categorical)
This allows realistic user interaction without forcing 80+ inputs.

## Production Features - 

1. Custom exception handling
2. Structured logging
3. Config-driven paths
4. Input validation with Pydantic
5. End-to-end reproducibility


📌 Tech Stack
Python
Pandas, NumPy
Scikit-learn
FastAPI
Pydantic
Joblib

✨ Key Learnings

Designing ML systems beyond notebooks
Handling real-world inference constraints
Avoiding training-serving skew
Writing clean, modular ML code
Building user-facing ML APIs

👤 Author

Nishi Gupta
Aspiring Machine Learning Engineer