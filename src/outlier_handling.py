import sys
import pandas as pd
from typing import Dict

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class OutlierHandler:
    def __init__(self):
        self.bounds: Dict[str, tuple] = {}

    def calculate_iqr_bounds(self, df: pd.DataFrame, columns: list):
        """
        Learn IQR bounds from TRAIN data only
        """
        try:
            logger.info("Fitting outlier bounds using IQR")

            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                self.bounds[col] = (lower, upper)

            logger.info("Outlier bounds fitted successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def transform_dataframe_using_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned caps to any dataset
        """
        try:
            logger.info("Applying outlier capping")

            df = df.copy()

            for col, (lower, upper) in self.bounds.items():
                df[col] = df[col].clip(lower, upper)

            logger.info("Outlier capping applied successfully")
            return df

        except Exception as e:
            raise CustomException(e, sys)
