import os
import sys
import json
import joblib
from typing import Dict

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.common import create_directories

logger = get_logger(__name__)


class ModelEvaluation:
    def __init__(self):
        self.model_dir = "artifacts/model"
        self.report_dir = "artifacts/reports"

        create_directories([self.model_dir, self.report_dir])

    def save_model(self, model, model_name: str):
        try:
            model_path = os.path.join(self.model_dir, "best_model.pkl")
            joblib.dump(model, model_path)

            logger.info(f"Saved best model [{model_name}] at {model_path}")
            return model_path

        except Exception as e:
            raise CustomException(e, sys)

    def save_metrics(self, metrics: Dict):
        try:
            metrics_path = os.path.join(self.report_dir, "model_metrics.json")

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Saved model metrics at {metrics_path}")
            return metrics_path

        except Exception as e:
            raise CustomException(e, sys)
