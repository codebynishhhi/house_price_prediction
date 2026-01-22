# House Price Prediction Project 

## Project Overview
This project builds a machine learning model to predict house prices using the Ames Housing dataset. The goal is to understand the factors that influence house prices and build a production-ready predictive model.

## Dataset
- **Source:** Ames Housing Dataset
- **Rows:** 1,460 houses
- **Features:** 79 features (numerical & categorical)
- **Target:** SalePrice (in dollars)
 
## Project Structure
```
house_price_prediction/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned & preprocessed data
├── src/
│   ├── data_loader.py    # Load and explore data
│   ├── eda.py            # Exploratory Data Analysis
│   ├── preprocessing.py  # Data preprocessing
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── pipeline.py       # End-to-end pipeline
│   └── utils.py
├── notebooks/
│   └── exploration.ipynb # Optional: Quick exploration
├── models/               # Trained models
├── logs/                 # Logs & outputs
├── config/
│   └── config.yaml       # Configuration settings
├── requirements.txt      # Package dependencies
└── README.md            # This file
```

## How to Run

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python main.py
```

### 3. View Results
Results will be saved in:
- `data/processed/` - Processed datasets
- `models/` - Trained model
- `logs/` - Training logs

## Results
# Summary of the Project
In this project, we performed an end-to-end machine learning workflow to predict house prices using the Ames Housing Dataset. 
The main steps included:
1. Data Understanding & Preprocessing :

Performed extensive EDA to understand missing values, skewness, outliers, and feature distributions.
Dropped features with excessive missing values (>60%) and non-informative identifiers.

Applied:
Median / zero imputation for numerical features
Mode imputation for categorical features
Handled skewness using log transformation on the target variable.
Applied outlier capping (IQR method) for heavily skewed numerical features to reduce the influence of extreme values.
Encoded categorical variables using appropriate encoding strategies.

2. Modeling Approach

The following regression models were trained and evaluated:
Linear Regression (baseline)
Ridge Regression
Lasso Regression
Random Forest Regression

Each model was evaluated using:
Train/Test split
R² score
RMSE
Residual analysis
Cross-validation

3. Model Comparison

| Model             | R² (Train) | R² (Test) | RMSE (Train) | RMSE (Test) |
| ----------------- | ---------- | --------- | ------------ | ----------- |
| Linear Regression | 0.930      | 0.931     | 0.106        | 0.113       |
| Ridge Regression  | 0.926      | **0.932** | 0.109        | **0.112**   |
| Lasso Regression  | 0.813      | 0.870     | 0.173        | 0.155       |
| Random Forest     | **0.982**  | 0.922     | **0.054**    | 0.120       |

4. Model Selection

Although Random Forest achieved the highest training performance, Ridge Regression was selected as the final model due to:
- Strong generalization (highest test R²)
- Low variance between train and test scores
- Better interpretability
- Lower risk of overfitting
- Simpler and more stable behavior on tabular data

5. Model Diagnostics:

Residuals are centered around zero with no clear pattern → model assumptions hold
Actual vs Predicted plot shows strong linear alignment
Errors are proportionally small in log-space

Model Persistence : The final Ridge Regression model was saved using joblib for future inference and deployment.

## Author
Nishi Gupta

## Contact
https://www.linkedin.com/in/nishi-gupta-b46b66179/