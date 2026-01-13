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
(Will be updated as project progresses)

## Author
Nishi Gupta

## Contact
https://www.linkedin.com/in/nishi-gupta-b46b66179/