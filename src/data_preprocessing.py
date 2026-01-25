# This file is used for Data loading and basic eda to -
# 1. handle missing values
# 2. column separation - categorical and numerical
# 3. create a preprocessing pipeline
# 4. save the preprocessor object

import os 
import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.common import save_model_object
from src.utils.config import PreprocessorConfig

logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self):
        self.config = PreprocessorConfig()
        self.preprocessor_path = self.config.preprocessor_path

    def get_preprocessor_object(self, df:pd.DataFrame):
        """
            Creates preprocessing pipeline for numeric and categorical features
        """
        try:
            logger.info("Identifying numerical and categorical columns")
            
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            logger.info(f"Numerical columns:{list(numerical_cols)}")
            logger.info(f"Categorical columns : {list(categorical_cols)}")

            # Numerical pipeline
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))
            ])

            # Categorical pipeline
            categorical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy='most_frequent'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numeric_pipeline, numerical_cols),
                    ('cat_pipeline', categorical_pipeline, categorical_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            logger.error("Error in creating preprocessor object")
            raise CustomException(e, sys) from e
                    
                    
    def initiate_data_preprocessing(self, train_path:str, test_path:str):
        """
            This function is used to load the train and test data,
            create preprocessing object, fit and transform the train data,
            transform the test data and save the preprocessor object.
        """
        logger.info("Data Preprocessing method starts")
        try:
            logger.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = 'SalePrice'

            # droop target column from input features
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logger.info("Creating preprocessor object")
            preprocessor = self.get_preprocessor_object(X_train)

            logger.info("Fitting and transforming train data")
            X_train_processed = preprocessor.fit_transform(X_train)
            logger.info("Transforming test data")
            X_test_processed = preprocessor.transform(X_test)

            save_model_object(self.preprocessor_path, preprocessor)

            train_arr = np.c_[X_train_processed, y_train.values]
            test_arr = np.c_[X_test_processed, y_test.values]

            logger.info("Data Preprocessing method completed")
            return train_arr, test_arr
        
        except Exception as e:
            logger.error("Error occurred in data preprocessing")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    logger.info("Data Preprocessing script started")
    logger.info("Script execution completed")