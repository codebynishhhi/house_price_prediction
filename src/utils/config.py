import os
from dataclasses import dataclass


# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(BASE_DIR, "data", "raw", "raw.csv")
    train_data_path: str = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    test_data_path: str = os.path.join(BASE_DIR, "data", "processed", "test.csv")


@dataclass
class DataPreprocessingConfig:
    processed_data_path: str = os.path.join(BASE_DIR, "data", "processed", "processed.csv")


@dataclass
class ModelTrainingConfig:
    model_path: str = os.path.join(BASE_DIR, "artifacts", "model.pkl")


@dataclass
class PreprocessorConfig:
    preprocessor_path: str = os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")
