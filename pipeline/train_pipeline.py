import sys
import pandas as pd

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
import joblib
from sklearn.pipeline import Pipeline

logger = get_logger(__name__)


class TrainingPipeline:
    def run_pipeline(self):
        try:
            logger.info("Training pipeline started")

            # Phase 2: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initate_data_ingestion()

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "SalePrice"

            y_train = train_df[target_col]
            y_test = test_df[target_col]

            X_train = train_df.drop(columns=[target_col])
            X_test = test_df.drop(columns=[target_col])

            # Phase 3: Feature Engineering (DATAFRAME ONLY)
            fe = FeatureEngineering()
            X_train = fe.transform(X_train)
            X_test = fe.transform(X_test)

            # Phase 4: Outlier Handling
            outlier = OutlierHandler()

            # identigy numeric columns and exclude the target
            numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
            
            # learn outlier bounds(lower & upper) from training data
            outlier.fit(X_train, numeric_cols)

            X_train = outlier.transform(X_train)
            X_test = outlier.transform(X_test)

            # Phase 5: Encoding (NEEDS COLUMN NAMES)
            encoder = DataEncoding()
            encoding_transformer = encoder.get_transformer(X_train)

            X_train = encoding_transformer.fit_transform(X_train)
            X_test = encoding_transformer.transform(X_test)

            # Phase 6: Imputation
            preprocessing = DataPreprocessing()
            preprocessor = preprocessing.get_preprocessor_object(
                pd.DataFrame(X_train)
            )

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Phase 7: Scaling
            scaler = DataScaling()
            scaler_transformer = scaler.get_scaled_features(pd.DataFrame(X_train))

            X_train = scaler_transformer.fit_transform(X_train)
            X_test = scaler_transformer.transform(X_test)

            # Phase 9: Model Training
            trainer = ModelTrainer()
            trained_models = trainer.train_and_evaluate(
                X_train, y_train, X_test, y_test
            )

            # Phase 10: Model Selection
            selector = ModelSelector()
            best_model_name, best_model, metrics = selector.select_best_model(
                trained_models
            )

            # Phase 11:
            # Creating and saving a full production pipeline
            logger.info("Creating a full production pipeline")
            full_pipeline = Pipeline(steps=[
                ("feature_engineering", fe),
                ("outlier_handler", outlier),
                ("encoding", encoding_transformer),
                ("imputation", preprocessor),
                ("scaling", scaler_transformer),
                ("model", best_model)  
            ])

            pipeline_path = "artifacts/model/full_pipeline.pkl"
            joblib.dump(full_pipeline, pipeline_path)

            logger.info(f"Best model is - {best_model}")
            logger.info(f"Full pipeline saved at {pipeline_path}")
            
            # Phase 12: Save model & metrics
            evaluator = ModelEvaluation()
            evaluator.save_model(best_model, best_model_name)
            evaluator.save_metrics(metrics)

            logger.info("Training pipeline completed successfully")

        except Exception as e:
            logger.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    logger.info("Starting Training Pipeline execution")
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
    logger.info("Training Pipeline execution finished")
