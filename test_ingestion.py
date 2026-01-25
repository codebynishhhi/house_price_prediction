from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.encoding import DataEncoding
from src.scaling import DataScaling
from src.outlier_handling import OutlierHandler
from src.model_training import ModelTrainer
from src.model_selection import ModelSelector
from src.model_evaluation import ModelEvaluation
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Test Data Ingestion
    print("="*50)
    print("Testing Data Ingestion...")
    print("="*50)
    
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initate_data_ingestion(
        data_path = "data/raw/AmesHousing.csv"
    )

    print(f"Train data path: {train_path}")
    print(f"Test data path: {test_path}")
    
    # Test Data Preprocessing
    print("\n" + "="*50)
    print("Testing Data Preprocessing...")
    print("="*50)
    
    preprocessing = DataPreprocessing()
    train_arr, test_arr = preprocessing.initiate_data_preprocessing(
        train_path=train_path,
        test_path=test_path
    )
    
    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    print("Data Preprocessing completed successfully!")
    
    # Test Feature Engineering
    print("\n" + "="*50)
    print("Testing Feature Engineering...")
    print("="*50)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    feature_eng = FeatureEngineering()
    train_df_fe = feature_eng.start_feature_engineering(train_df)
    test_df_fe = feature_eng.start_feature_engineering(test_df)
    
    print(f"Train data after feature engineering shape: {train_df_fe.shape}")
    print(f"Test data after feature engineering shape: {test_df_fe.shape}")
    print("Feature Engineering completed successfully!")
    
    # Test Data Encoding
    print("\n" + "="*50)
    print("Testing Data Encoding...")
    print("="*50)
    
    encoding = DataEncoding()
    transformer = encoding.get_transformer(train_df_fe)
    
    print("Data Encoding transformer created successfully!")
    
    # Test Data Scaling
    print("\n" + "="*50)
    print("Testing Data Scaling...")
    print("="*50)
    
    scaling = DataScaling()
    scaling_transformer = scaling.get_scaled_features(train_df_fe)
    
    print("Data Scaling transformer created successfully!")
    
    # Test Outlier Handling
    print("\n" + "="*50)
    print("Testing Outlier Handling...")
    print("="*50)
    
    # Get numerical columns for outlier handling
    numerical_cols = train_df_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['SalePrice', 'SalePrice_log']]
    
    outlier_handler = OutlierHandler()
    outlier_handler.calculate_iqr_bounds(train_df_fe, numerical_cols)
    
    train_df_cleaned = outlier_handler.transform_dataframe_using_bounds(train_df_fe)
    test_df_cleaned = outlier_handler.transform_dataframe_using_bounds(test_df_fe)
    
    print(f"Train data after outlier handling shape: {train_df_cleaned.shape}")
    print(f"Test data after outlier handling shape: {test_df_cleaned.shape}")
    print("Outlier Handling completed successfully!")
    
    # Test Model Training
    print("\n" + "="*50)
    print("Testing Model Training...")
    print("="*50)
    
    # Prepare numerical features for model training
    numerical_cols = train_df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target columns if present
    if 'SalePrice' in numerical_cols:
        numerical_cols.remove('SalePrice')
    if 'SalePrice_log' in numerical_cols:
        numerical_cols.remove('SalePrice_log')
    
    # Create a copy and drop NaN values
    train_df_model = train_df_cleaned[[*numerical_cols, 'SalePrice']].dropna()
    test_df_model = test_df_cleaned[[*numerical_cols, 'SalePrice']].dropna()
    
    # Extract features and target
    X_train_model = train_df_model[numerical_cols].values
    y_train_model = train_df_model['SalePrice'].values
    
    X_test_model = test_df_model[numerical_cols].values
    y_test_model = test_df_model['SalePrice'].values
    
    print(f"Training data shape: {X_train_model.shape}")
    print(f"Test data shape: {X_test_model.shape}")
    
    # Train models
    model_trainer = ModelTrainer()
    results = model_trainer.train_and_evaluate(X_train_model, y_train_model, X_test_model, y_test_model)
    
    # Display results
    print("\nModel Performance Comparison:")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"\n{model_name:20s} | Train R²: {metrics['train_r2']:.4f} | Test R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f}")
    
    print("\n✓ Model Training completed successfully!")
    
    # Test Model Selection
    print("\n" + "="*50)
    print("Testing Model Selection...")
    print("="*50)
    
    model_selector = ModelSelector()
    best_model_name, best_model, best_score = model_selector.select_best_model(results)
    
    print(f"\n✓ Best Model Selected: {best_model_name}")
    print(f"  Test R² Score: {best_score:.4f}")
    print(f"  Model Object: {best_model}")
    
    print("\n✓ Model Selection completed successfully!")
    
    # Test Model Evaluation
    print("\n" + "="*50)
    print("Testing Model Evaluation...")
    print("="*50)
    
    model_evaluator = ModelEvaluation()
    
    # Save the best model
    model_path = model_evaluator.save_model(best_model, best_model_name)
    print(f"\n✓ Best model saved at: {model_path}")
    
    # Create metrics dictionary
    metrics_dict = {
        "best_model": best_model_name,
        "test_r2_score": float(best_score),
        "training_data_size": X_train_model.shape[0],
        "test_data_size": X_test_model.shape[0],
        "number_of_features": X_train_model.shape[1],
    }
    
    # Save metrics
    metrics_path = model_evaluator.save_metrics(metrics_dict)
    print(f"✓ Model metrics saved at: {metrics_path}")
    
    print("\n✓ Model Evaluation completed successfully!")
    
    print("\n" + "="*50)
    print("\n✓ All tests completed successfully!")
    print("="*50)