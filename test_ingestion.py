from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing

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
    print("="*50)