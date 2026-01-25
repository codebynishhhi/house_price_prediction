import sys
from typing import Dict

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class ModelSelector:
    def select_best_model(self, model_results: Dict[str, dict]):

        try:
            logger.info("Selecting best model based on test performance")

            best_model_name = None
            best_model = None
            best_score = float("-inf")

            for name, metrics in model_results.items():
                test_r2 = metrics["test_r2"]
                train_r2 = metrics["train_r2"]

                generalization_gap = abs(train_r2 - test_r2)

                logger.info(
                    f"{name} | "
                    f"Train R2: {train_r2:.4f}, "
                    f"Test R2: {test_r2:.4f}, "
                    f"Gap: {generalization_gap:.4f}"
                )

                # Core selection logic
                if test_r2 > best_score and generalization_gap < 0.05:
                    best_score = test_r2
                    best_model_name = name
                    best_model = metrics["model"]

            logger.info(
                f"Best model selected: {best_model_name} "
                f"with Test R2 = {best_score:.4f}"
            )

            return best_model_name, best_model, best_score

        except Exception as e:
            raise CustomException(e, sys)
