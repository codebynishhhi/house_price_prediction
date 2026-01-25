import sys
import numpy as np
from typing import Dict

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001),
            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        }

    def train_and_evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ) -> Dict[str, dict]:

        try:
            logger.info("Starting model training")

            results = {}

            for name, model in self.models.items():
                logger.info(f"Training model: {name}")

                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

                results[name] = {
                    "model": model,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse
                }

                logger.info(
                    f"{name} | "
                    f"Train R2: {train_r2:.4f}, "
                    f"Test R2: {test_r2:.4f}"
                )

            logger.info("Model training completed")
            return results

        except Exception as e:
            raise CustomException(e, sys)

