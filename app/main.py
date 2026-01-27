from fastapi import FastAPI
from app.schema import HouseInput
from pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="House Price Prediction API",
    version="1.0"
)

# Load pipeline once (IMPORTANT)
prediction_pipeline = PredictionPipeline()
prediction_pipeline.load_pipeline()


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict_price(data: HouseInput):
    prediction = prediction_pipeline.predict_results(data)
    return {
        "predicted_price": prediction
    }
