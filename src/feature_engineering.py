import sys
import os

# Add the parent directory to the Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineering:
    def __init__(self):
        pass

    def start_feature_engineering(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the given DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame after feature engineering.
        """
        try:
            logger.info("Starting feature engineering")

            # age feature
            df['House_Age'] = df["Yr Sold"] - df["Year Built"]
            df['Remod_Age'] = df["Yr Sold"] - df["Year Remod/Add"]

            # total bathrooms
            df["Total_Bathrooms"] = (
                df["Full Bath"]
                + 0.5 * df["Half Bath"]
                + df["Bsmt Full Bath"]
                + 0.5 * df["Bsmt Half Bath"]
            )

            # Total square footage
            df["Total_SF"] = (
                df["Total Bsmt SF"]
                + df["1st Flr SF"]
                + df["2nd Flr SF"]
            )

            # Binary indicators
            df["Has_Garage"] = df["Garage Area"].apply(lambda x: 1 if x > 0 else 0)
            df["Has_Basement"] = df["Total Bsmt SF"].apply(lambda x: 1 if x > 0 else 0)

            # Drop redundant columns
            drop_cols = [
                "Year Built",
                "Year Remod/Add", 
                "PID",
                "Order"
            ]

            df.drop(columns=drop_cols, inplace=True, errors='ignore')

            logger.info("Feature engineering completed")
            return df

        except Exception as e:
            logger.error("Error occurred during feature engineering")
            raise CustomException(e, sys) from e