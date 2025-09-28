# backend/dataset_generator.py
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 5000

N_levels = np.random.uniform(50, 250, N)
P_levels = np.random.uniform(10, 80, N)
K_levels = np.random.uniform(20, 200, N)
ph = np.random.uniform(4.5, 8.5, N)
rainfall = np.random.uniform(50, 800, N)
temperature = np.random.uniform(10, 38, N)

crops = []
yields = []
for i in range(N):
    if rainfall[i] > 400 and ph[i] > 5.5:
        crop = "rice"
        y = 3.0 + 0.005 * N_levels[i] + 0.002 * P_levels[i] + np.random.normal(0, 0.3)
    elif temperature[i] < 20 and rainfall[i] < 300:
        crop = "wheat"
        y = 2.5 + 0.004 * K_levels[i] + np.random.normal(0, 0.25)
    elif ph[i] > 6.5 and rainfall[i] < 400:
        crop = "maize"
        y = 2.8 + 0.003 * (N_levels[i] + P_levels[i]) + np.random.normal(0, 0.3)
    else:
        crop = "cotton"
        y = 1.5 + 0.002 * K_levels[i] + np.random.normal(0, 0.2)
    crops.append(crop)
    yields.append(max(0.2, y))

df = pd.DataFrame({
    "N": N_levels,
    "P": P_levels,
    "K": K_levels,
    "ph": ph,
    "rainfall": rainfall,
    "temperature": temperature,
    "crop": crops,
    "yield": yields
})

out_path = os.path.join(os.path.dirname(__file__), "sample_agri_dataset.csv")
df.to_csv(out_path, index=False)
print("Saved", out_path)
