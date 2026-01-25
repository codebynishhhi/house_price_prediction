# This entry file for ML pipeline and is used to - 
# read raw data from various sources,
# train/test splitting,
# and save the raw data, train data, test data for further use.

import os 
import sys
from venv import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from utils.exception import CustomException
from utils.logger import get_logger

logger = get_logger(__name__)

# configuration for data ingestion
@dataclass
class DataIngestionConfig :
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path:str = os.path.join('artifacts', 'train_data.csv')
    test_data_path:str = os.path.join('artifacts', 'test_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self, data_path=None):
        '''
        This function is used to read raw data from various sources,
        perform train/test splitting
        and save the raw data, train data, test data for further use.
        '''
        logger.info("Data Ingestion method starts")
        try:
            # read the raw data
            if data_path is None:
                data_path = '../data/raw/AmesHousing.csv'
            df = pd.read_csv(data_path)

            # create the artifacts directory if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info("Raw data saved")

            # split the data into train and test sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            # save the train data and test data
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            logger.info("Train and test data saved")    
            logger.info("Data Ingestion method completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logger.error("Error occurred in data ingestion")
            raise CustomException(e, sys)