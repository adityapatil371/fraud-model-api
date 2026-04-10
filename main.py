from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import mlflow.pyfunc
import pandas as pd
import joblib
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    #model = mlflow.pyfunc.load_model("models:/fraud-detection-model@champion")
    model = joblib.load("models/best_model.pkl")
    print("Model loaded successfully")
    yield

app = FastAPI(lifespan=lifespan)

class FraudFeatures(BaseModel):
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
    Amount: float = Field(gt=0, description="Transaction amount must be > 0")

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": "champion"}

@app.post("/predict")
def predict(features: FraudFeatures):
    data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(data)
    return {
        "prediction": int(prediction[0]),
        "is_fraud": bool(prediction[0] == 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)