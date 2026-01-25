from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.encoding import DataEncoding
from src.scaling import DataScaling
import pandas as pd

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
    print("="*50)
    print("\n✓ All tests completed successfully!")
    print("="*50)