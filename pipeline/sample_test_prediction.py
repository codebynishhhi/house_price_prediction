import pandas as pd
from pipeline.prediction_pipeline import PredictionPipeline
from src.utils.config import DataIngestionConfig

# Load test data using config
config = DataIngestionConfig()
test_df = pd.read_csv(config.test_data_path)

# Take ONE ROW, DROP TARGET, KEEP FULL SCHEMA
sample_test_input = test_df.drop(columns=["SalePrice"]).iloc[[0]]

pipeline = PredictionPipeline()
pipeline.load_pipeline()

price = pipeline.predict_results(sample_test_input)
print(f"Predicted House Price: ₹{price:,.2f}")
