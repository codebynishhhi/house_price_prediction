import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

# function to save object
def save_model_object(file_path:str, obj:object) -> None:
    """
    Save a model object to a file using pickle.

    Args:
        file_path (str): The path where the object should be saved.
        obj (object): The model object to be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

        logger.info(f"Model object saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving model object: {e}")
        raise CustomException(e, sys)   
    
# function to load object
def load_model_object(file_path:str) -> object:
    """
    Load a model object from a file using pickle.

    Args:
        file_path (str): The path from where the object should be loaded.

    Returns:
        object: The loaded model object.
    """
    try:
        with open(file_path, "rb") as file:
            obj = pickle.load(file)

        logger.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        logger.error("Error occurred while saving object")
        raise CustomException(e, sys)
    
# function to get model metrics
def get_model_metrics(y_true, y_pred) -> dict:
    """
    Calculate regression metrics  -  R2 score and RMSE for the given true and predicted values.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: A dictionary containing R2 score and RMSE.
    """
    try:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return{
            "r2_score": r2,
            "rmse": rmse
        }
    except Exception as e:
        raise CustomException(e, sys)

# function to evaluate model
def evaluate_model(X_train, y_train, X_test, y_test, models:dict) -> dict:
    """
    Evaluate multiple regression models and return their R2 scores.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        models (dict): A dictionary of model name and model instance pairs.

    Returns:
        dict: A dictionary with model names as keys and their R2 scores as values.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_metrics = get_model_metrics(y_train, y_train_pred)
            test_metrics = get_model_metrics(y_test, y_test_pred)

            report[model_name] = {
                "train_r2_score": train_metrics['r2_score'],
                "test_r2_score": test_metrics['r2_score'],
                "train_rmse": train_metrics['rmse'],
                "test_rmse": test_metrics['rmse']
            }

            logger.info((f"{model_name} evaluation completed. "))

        return report
    except Exception as e:
        logger.error("Error occurred during model evaluation")
        raise CustomException(e, sys)