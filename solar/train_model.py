import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------
# Load dataset
# -------------------------------
DATA_PATH = "data/solarpowergeneration.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------
# Normalize column names
# -------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace("-", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

print("Columns after normalization:")
print(df.columns.tolist())

# -------------------------------
# Feature selection (FINAL)
# -------------------------------
FEATURES = [
    "distance_to_solar_noon",
    "temperature",
    "wind_speed",
    "sky_cover",
    "visibility",
    "humidity"
]

TARGET = "power_generated"

X = df[FEATURES]
y = df[TARGET]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Model training
# -------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation:")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.2f}")

# -------------------------------
# Save model and scaler
# -------------------------------
os.makedirs("models", exist_ok=True)

with open("models/regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model and scaler saved successfully in 'models/' folder")
