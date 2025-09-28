# backend/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="SIH Crop Rec & Yield API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(__file__)
MODEL_STORE = os.path.join(BASE, "model_store")
CROP_MODEL_PATH = os.path.join(MODEL_STORE, "crop_rec.joblib")
YIELD_MODEL_PATH = os.path.join(MODEL_STORE, "yield_reg.joblib")
CROP_LE_PATH = os.path.join(MODEL_STORE, "crop_le.joblib")

crop_model = None
yield_model = None
crop_le = None

if os.path.exists(CROP_MODEL_PATH):
    crop_model = joblib.load(CROP_MODEL_PATH)
if os.path.exists(YIELD_MODEL_PATH):
    yield_model = joblib.load(YIELD_MODEL_PATH)
if os.path.exists(CROP_LE_PATH):
    crop_le = joblib.load(CROP_LE_PATH)

class RecRequest(BaseModel):
    N: float
    P: float
    K: float
    ph: float
    rainfall: float
    temperature: float
    humidity: Optional[float] = None
    location: Optional[str] = None

class YieldRequest(RecRequest):
    crop: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecRequest):
    if crop_model is None:
        raise HTTPException(status_code=500, detail="Model not trained. Run train_model.py")
    X = [[req.N, req.P, req.K, req.ph, req.rainfall, req.temperature]]
    pred_enc = crop_model.predict(X)
    recommended = None
    try:
        if crop_le is not None:
            recommended = str(crop_le.inverse_transform(pred_enc)[0])
        else:
            # fallback: return encoded label
            recommended = str(pred_enc[0])
    except Exception:
        recommended = str(pred_enc[0])
    probs = None
    try:
        probs = crop_model.predict_proba(X).tolist()
    except Exception:
        probs = None
    return {"recommended_crop": recommended, "proba": probs}

@app.post("/predict")
def predict_yield(req: YieldRequest):
    if yield_model is None:
        raise HTTPException(status_code=500, detail="Yield model not trained. Run train_model.py")
    # For demo: simple crop encoding; must match train_model.py mapping
    crop_map = {"rice": 0, "wheat": 1, "maize": 2, "cotton": 3}
    crop_enc = crop_map.get(req.crop.lower(), 0)
    X = [[req.N, req.P, req.K, req.ph, req.rainfall, req.temperature, crop_enc]]
    yhat = yield_model.predict(X)
    return {"predicted_yield_tonnes_per_hectare": float(yhat[0])}

@app.get("/weather")
def weather(location: Optional[str] = None):
    # Mock response — replace with real weather API
    return {"location": location or "unknown", "rainfall_next_week_mm": 12, "temp_avg_C": 28}

@app.get("/price")
def price(crop: str, mandi: Optional[str] = None):
    prices = {"rice": 2500, "wheat": 2200, "maize": 1800, "cotton": 4500}
    return {"crop": crop, "price_per_quintal": prices.get(crop.lower(), 1500)}
