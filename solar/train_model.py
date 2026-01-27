import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Create models directory if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Load data
print("Loading data...")
try:
    df = pd.read_csv("solarpowergeneration (2) (1).csv")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file name.")
    exit(1)

# Feature selection based on notebook analysis
# Note: Notebook used 'distance_to_solar_noon' (underscores) but CSV likely has 'distance-to-solar-noon' (dashes) or similar.
# Inspecting CSV header earlier showed: distance-to-solar-noon,temperature,wind-direction,wind-speed,sky-cover,visibility,humidity,average-wind-speed-(period),average-pressure-(period),power-generated

selected_features = ['distance-to-solar-noon', 'temperature', 'sky-cover', 'visibility', 'humidity', 'wind-speed']
target = 'power-generated'

# Verify columns exist
missing_cols = [col for col in selected_features if col not in df.columns]
if missing_cols:
    print(f"Missing columns in CSV: {missing_cols}")
    # Fallback checking for underscore versions if dash versions fail, just in case
    exit(1)

X = df[selected_features]
y = df[target]

print(f"Features: {selected_features}")
print(f"Target: {target}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

# Save model and scaler
print("Saving model and scaler...")
joblib.dump(model, "models/regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Done.")
