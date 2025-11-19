import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("final_model.h5")
scaler = joblib.load("scaler.save")

# Title and instructions
st.title("Menstrual Cycle Predictor")
st.write("Enter your last cycle information to predict the next cycle and menstruation length.")

# User input
cycle_length = st.number_input("Previous Cycle Length (days):", min_value=10, max_value=60, value=28)
menstruation_length = st.number_input("Previous Menstruation Length (days):", min_value=1, max_value=15, value=5)

if st.button("Predict Next Cycle"):
    input_data = np.array([[cycle_length, menstruation_length]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input.reshape(1, 1, 2))
    predicted = scaler.inverse_transform(prediction)

    st.success(f"Predicted Cycle Length: {predicted[0][0]:.2f} days")
    st.success(f"Predicted Menstruation Length: {predicted[0][1]:.2f} days")
