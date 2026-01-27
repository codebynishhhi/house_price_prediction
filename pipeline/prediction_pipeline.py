import sys
import pandas as pd
import numpy as np
import joblib

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.config import PredictionConfig
from pipeline.input_adapter import InputAdapter


logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self):
        self.config = PredictionConfig()
        self.pipeline = joblib.load("artifacts/model/full_pipeline.pkl")
        self.adapter = InputAdapter("artifacts/input_defaults.pkl")

    def load_pipeline(self):
        try:
            logger.info("Loading full training saved pipeline")
            self.pipeline = joblib.load(self.config.model_path)
            logger.info("Pipeline loaded successfully!")

        except Exception as e:
            raise CustomException(e, sys)
        
    # def predict_results(self, input_data:dict):
    #     try:
    #         logger.info("Starting prediction")

    #         # input data can be both dict and dataframe
    #         if isinstance(input_data, pd.DataFrame):
    #             df = input_data.copy()
    #         else:
    #             df = pd.DataFrame([input_data])

    #         prediction = self.pipeline.predict(df)

    #         logger.info("Prediction completed successfully!")

    #         return float(prediction[0])
    #     except Exception as e:
    #         raise CustomException(e, sys)
    def predict_results(self, input_data: dict):
        try:
            logger.info("Starting prediction")

            # ALWAYS go through the adapter
            df = self.adapter.adapt(input_data)

            prediction = self.pipeline.predict(df)

            logger.info("Prediction completed successfully!")
            return float(prediction[0])

        except Exception as e:
            raise CustomException(e, sys)
