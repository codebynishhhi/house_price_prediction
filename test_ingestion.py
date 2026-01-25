from src.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion()

    train_path, test_path = ingestion.initate_data_ingestion(
        data_path = "data/raw/AmesHousing.csv"
    )

    print(f"Train data path: {train_path}")
    print(f"Test data path: {test_path}")