# backend/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "sample_agri_dataset.csv")
MODEL_STORE = os.path.join(BASE, "model_store")
os.makedirs(MODEL_STORE, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Dataset not found. Run dataset_generator.py first.")
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH)

# Crop classifier
X = df[["N", "P", "K", "ph", "rainfall", "temperature"]]
y = df["crop"]
le = LabelEncoder()
y_enc = le.fit_transform(y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y_enc)
joblib.dump(clf, os.path.join(MODEL_STORE, "crop_rec.joblib"))
joblib.dump(le, os.path.join(MODEL_STORE, "crop_le.joblib"))
print("Saved crop_rec.joblib and crop_le.joblib")

# Yield regressor (include crop encoding)
crop_map = {"rice": 0, "wheat": 1, "maize": 2, "cotton": 3}
df["crop_enc"] = df["crop"].map(crop_map)
Xr = df[["N", "P", "K", "ph", "rainfall", "temperature", "crop_enc"]]
yr = df["yield"]
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(Xr, yr)
joblib.dump(reg, os.path.join(MODEL_STORE, "yield_reg.joblib"))
print("Saved yield_reg.joblib")

