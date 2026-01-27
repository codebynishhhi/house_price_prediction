import sys
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            logger.info("Applying feature engineering")
            X = X.copy()

            X["House_Age"] = X["Yr Sold"] - X["Year Built"]
            X["Remod_Age"] = X["Yr Sold"] - X["Year Remod/Add"]

            X["Total_Bathrooms"] = (
                X["Full Bath"]
                + 0.5 * X["Half Bath"]
                + X["Bsmt Full Bath"]
                + 0.5 * X["Bsmt Half Bath"]
            )

            X["Total_SF"] = (
                X["Total Bsmt SF"]
                + X["1st Flr SF"]
                + X["2nd Flr SF"]
            )

            X["Has_Garage"] = (X["Garage Area"] > 0).astype(int)
            X["Has_Basement"] = (X["Total Bsmt SF"] > 0).astype(int)

            X.drop(
                columns=["Year Built", "Year Remod/Add", "PID", "Order"],
                errors="ignore",
                inplace=True
            )

            return X

        except Exception as e:
            raise CustomException(e, sys)
