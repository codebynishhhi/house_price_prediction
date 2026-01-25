import os
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.utils.exception import CustomException
from src.utils.logger import get_logger 

logger = get_logger(__name__)

class DataScaling:
    def __init__(self):
        pass

    def get_scaled_features(self, df:pd.DataFrame) -> ColumnTransformer:
        """
        Build column transformer for scaling numerical features
        """ 
        try:
            logger.info("Creating scaling transformer for numerical features")

            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

            # target column should not be scaled
            numerical_columns = [col for col in numerical_columns if col not in ['SalePrice', 'SalePrice_log']]
            scaler = StandardScaler()

            scaling_pipeline = ColumnTransformer(
                transformers=[
                    ('scaler', scaler, numerical_columns)
                ],
                remainder='passthrough'
            )

            logger.info("Scaling transformer created successfully")
            return scaling_pipeline
        except Exception as e:
            logger.error("Error in creating scaling transformer")
            raise CustomException(e, sys) from e