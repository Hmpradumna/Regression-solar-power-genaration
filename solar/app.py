import streamlit as st
import numpy as np
import pickle
import os

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Solar Power Generation Prediction",
    page_icon="☀️",
    layout="centered"
)

# -------------------------------
# Load model and scaler
# -------------------------------
MODEL_PATH = os.path.join("models", "regression_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# App UI
# -------------------------------
st.title("☀️ Solar Power Generation Prediction")
st.write(
    "Enter weather and environmental conditions to predict solar power generation "
    "using a regression-based machine learning model."
)

st.divider()

# -------------------------------
# Input fields
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    distance_to_noon = st.number_input(
        "Distance to Solar Noon (radians)",
        min_value=0.0,
        max_value=3.14,
        value=1.0,
        step=0.01
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0,
        step=0.5
    )

    sky_cover = st.number_input(
        "Sky Cover (0–4)",
        min_value=0,
        max_value=4,
        value=1,
        step=1
    )

with col2:
    visibility = st.number_input(
        "Visibility (km)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.5
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=150.0,
        value=5.0,
        step=0.5
    )

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Power Generation"):
    try:
        # Arrange inputs in SAME ORDER as training
        input_data = np.array([[
            distance_to_noon,
            visibility,
            temperature,
            humidity,
            sky_cover,
            wind_speed
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        st.success(
            f"🔋 Predicted Solar Power Generation: **{prediction[0]:.2f} kW**"
        )

        # Optional transparency
        with st.expander("View Input Summary"):
            st.write({
                "Distance to Solar Noon": distance_to_noon,
                "Visibility (km)": visibility,
                "Temperature (°C)": temperature,
                "Humidity (%)": humidity,
                "Sky Cover": sky_cover,
                "Wind Speed (km/h)": wind_speed
            })

    except Exception as e:
        st.error("Something went wrong during prediction.")
        st.error(str(e))

# -------------------------------
# Footer
# -------------------------------
st.caption(
    "Model trained using regression techniques on historical solar power data. "
    "Deployment via Streamlit."
)
