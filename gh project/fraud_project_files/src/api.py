from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class Transaction(BaseModel):
    features: list

# Load model and scaler (ensure files exist: models/xgb.joblib, models/scaler.joblib)
try:
    model = joblib.load('models/xgb.joblib')
    scaler = joblib.load('models/scaler.joblib')
except Exception as e:
    model = None
    scaler = None

@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.post('/predict')
async def predict(tx: Transaction):
    if model is None or scaler is None:
        return {'error': 'Model or scaler not found. Run training first.'}
    x = np.array(tx.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prob = float(model.predict_proba(x_scaled)[0,1])
    label = int(prob > 0.5)
    return {'probability': prob, 'label': label}
