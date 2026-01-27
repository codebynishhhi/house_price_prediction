import os
from dataclasses import dataclass

# Project root: house_price_prediction
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(BASE_DIR, "data", "raw", "raw_data.csv")
    train_data_path: str = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
    test_data_path: str = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")

@dataclass
class DataPreprocessingConfig:
    processed_data_path: str = os.path.join(BASE_DIR, "data", "processed", "processed.csv")

@dataclass
class ModelTrainingConfig:
    model_path: str = os.path.join(BASE_DIR, "artifacts", "model.pkl")

@dataclass
class PreprocessorConfig:
    preprocessor_path: str = os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")

@dataclass
class PredictionConfig:
    model_path: str = os.path.join(
        BASE_DIR, "artifacts", "model", "full_pipeline.pkl"
    )
