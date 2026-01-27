import sys
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class OutlierHandler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.bounds = {}

    def fit(self, X, y=None):
        try:
            logger.info("Learning IQR bounds")
            for col in X.select_dtypes(include=["int64", "float64"]).columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        try:
            logger.info("Applying outlier capping")
            X = X.copy()
            for col, (low, high) in self.bounds.items():
                if col in X.columns:
                    X[col] = X[col].clip(low, high)
            return X
        except Exception as e:
            raise CustomException(e, sys)
