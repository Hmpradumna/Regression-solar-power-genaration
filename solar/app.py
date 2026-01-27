import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Solar Power Prediction", layout="centered")

# Load model and scaler
@st.cache_resource
def load_model():
    if not os.path.exists("models/regression_model.pkl") or not os.path.exists("models/scaler.pkl"):
        st.error("Model or Scaler not found. Please run 'start_app.py' to train the model first.")
        return None, None
    
    model = joblib.load("models/regression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("☀️ Solar Power Generation Prediction")
st.write("Enter the weather conditions to predict solar power generation.")

if model is not None and scaler is not None:
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            distance_to_solar_noon = st.number_input("Distance to Solar Noon (radians)", min_value=0.0, format="%.4f")
            temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, format="%.1f")
            sky_cover = st.number_input("Sky Cover (0-4)", min_value=0, max_value=4, step=1)
            
        with col2:
            visibility = st.number_input("Visibility (km)", min_value=0.0, format="%.1f")
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, format="%.1f")
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, format="%.1f")
            
        submitted = st.form_submit_button("Predict Power Generation")
        
        if submitted:
            # Create input dataframe ensuring correct order of features
            # Features: ['distance-to-solar-noon', 'temperature', 'sky-cover', 'visibility', 'humidity', 'wind-speed']
            input_data = pd.DataFrame({
                'distance-to-solar-noon': [distance_to_solar_noon],
                'temperature': [temperature],
                'sky-cover': [sky_cover],
                'visibility': [visibility],
                'humidity': [humidity],
                'wind-speed': [wind_speed]
            })
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.success(f"Predicted Power Generation: **{prediction:.2f} Joules**")