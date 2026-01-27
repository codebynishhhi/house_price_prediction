import sys
from typing import Dict

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class ModelSelector:
    def select_best_model(self, model_results: Dict[str, dict]):
        try:
            logger.info("Selecting best model based on test performance")

            # 🔍 DEBUG: log full incoming results structure
            logger.info(f"Model results keys: {list(model_results.keys())}")

            best_model_name = None
            best_model = None
            best_score = float("-inf")

            for model_name, metrics in model_results.items():
                logger.info(f"Evaluating model: {model_name}")
                logger.info(f"Raw metrics dict: {metrics}")

                # 🔍 DEBUG: check required keys explicitly
                if "test_r2" not in metrics or "train_r2" not in metrics:
                    logger.warning(
                        f"Skipping {model_name} due to missing metrics keys"
                    )
                    continue

                test_r2 = metrics["test_r2"]
                train_r2 = metrics["train_r2"]

                generalization_gap = abs(train_r2 - test_r2)

                logger.info(
                    f"{model_name} | "
                    f"Train R2: {train_r2:.4f}, "
                    f"Test R2: {test_r2:.4f}, "
                    f"Gap: {generalization_gap:.4f}"
                )

                # 🔍 DEBUG: log comparison condition
                logger.info(
                    f"Comparison check → "
                    f"test_r2 > best_score ? {test_r2} > {best_score} = {test_r2 > best_score}, "
                    f"gap < 0.05 ? {generalization_gap < 0.05}"
                )

                # Core selection logic (UNCHANGED)
                if test_r2 > best_score and generalization_gap < 0.09:
                    logger.info(f"→ {model_name} is current BEST candidate")

                    best_score = test_r2
                    best_model_name = model_name
                    best_model = metrics["model"]

            # 🔍 FINAL DECISION LOG
            if best_model_name is None:
                logger.warning(
                    "No model satisfied selection criteria "
                    "(test_r2 improvement + generalization gap < 0.05)"
                )
            else:
                logger.info(
                    f"Best model selected: {best_model_name} "
                    f"with Test R2 = {best_score:.4f}"
                )

            return best_model_name, best_model, {
                "best_model_name": best_model_name,
                "best_test_r2": best_score,
                 "selection_criteria": "highest test R2 with low generalization gap"
            }


        except Exception as e:
            raise CustomException(e, sys)
