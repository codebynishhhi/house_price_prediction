import pandas as pd
import joblib

def compute_defaults(train_csv_path, save_path):
    df = pd.read_csv(train_csv_path)

    defaults = {}

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            defaults[col] = df[col].median()
        else:
            defaults[col] = df[col].mode()[0]

    joblib.dump(defaults, save_path)
