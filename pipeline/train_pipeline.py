import sys

from src.utils.logger import get_logger
from src.utils.exception import CustomException

from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.encoding import DataEncoding
from src.scaling import DataScaling
from src.outlier_handling import OutlierHandler
from src.model_training import ModelTrainer
from src.model_selection import ModelSelector
from src.model_evaluation import ModelEvaluation

logger = get_logger(__name__)


class TrainingPipeline:
    def run_pipeline(self):
        try:
            logger.info("Training pipeline started")

            # Phase 3: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # Phase 4: Data Preprocessing
            preprocessing = DataPreprocessing()
            X_train, X_test, y_train, y_test = preprocessing.process(
                train_path, test_path
            )

            # Phase 5: Feature Engineering
            fe = FeatureEngineering()
            X_train, X_test = fe.transform(X_train, X_test)

            # Phase 6: Encoding
            encoder = DataEncoding()
            X_train, X_test = encoder.transform(X_train, X_test)

            # Phase 7: Scaling
            scaler = DataScaling()
            X_train, X_test = scaler.transform(X_train, X_test)

            # Phase 8: Outlier Handling
            outlier = OutlierHandler()
            X_train, X_test = outlier.transform(X_train, X_test)

            # Phase 9: Model Training
            trainer = ModelTrainer()
            trained_models = trainer.train(X_train, y_train)

            # Phase 10: Model Selection
            selector = ModelSelector()
            best_model, best_model_name, metrics = selector.select(
                trained_models, X_train, X_test, y_train, y_test
            )

            # Phase 11: Model Evaluation & Saving
            evaluator = ModelEvaluation()
            evaluator.save_model(best_model, best_model_name)
            evaluator.save_metrics(metrics)

            logger.info("Training pipeline completed successfully")

        except Exception as e:
            logger.error(" Training pipeline failed")
            raise CustomException(e, sys)
