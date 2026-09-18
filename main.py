import os
import numpy as np
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

# Build paths relative to this file's location so the app works
# no matter which directory it's launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#here note very importent you should put file 'The_Best_Model_prediction_fraud_model.pkl' and file 'Scaler_Transform.pkl' in the same folder have file main
MODEL_PATH = os.path.join(BASE_DIR, "The_Best_Model_prediction_fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Scaler_Transform.pkl")

app = FastAPI(title="Credit Card Fraud Detection API")

try:
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
except Exception as error:
    # Stop the app from starting in a broken state instead of failing
    # later with a confusing NameError on the first prediction request.
    raise RuntimeError(
        f"Failed to load model or scaler. Check the file paths.\n{error}"
    )


class InputFeatures(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class OutputFeatures(BaseModel):
    predict: int
    value: str


@app.get("/")
async def get_page():
    return {"message": "Welcome to the Credit Card Fraud Detection API"}


@app.post("/predict", response_model=OutputFeatures)
async def predict_case(data: InputFeatures):
    input_array = np.array([[
        data.Time,
        data.V1,
        data.V2,
        data.V3,
        data.V4,
        data.V5,
        data.V6,
        data.V7,
        data.V8,
        data.V9,
        data.V10,
        data.V11,
        data.V12,
        data.V13,
        data.V14,
        data.V15,
        data.V16,
        data.V17,
        data.V18,
        data.V19,
        data.V20,
        data.V21,
        data.V22,
        data.V23,
        data.V24,
        data.V25,
        data.V26,
        data.V27,
        data.V28,
        data.Amount,
    ]])

    scaled_features = scaler.transform(input_array)
    prediction = int(model.predict(scaled_features)[0])
    value = "fraud_case" if prediction == 1 else "normal_case"

    return {"predict": prediction, "value": value}