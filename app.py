import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model

# Load the trained model
model = load_model("final_model.h5", compile=False)

# Title
st.title("Menstrual Cycle Length & Period Length Predictor")

# User input form
st.header("Enter Recent Cycle Data")
with st.form("cycle_form"):
    last_cycle_length = st.number_input("Last cycle length (days):", min_value=20, max_value=40, value=28)
    last_menstruation_length = st.number_input("Last menstruation length (days):", min_value=2, max_value=10, value=5)
    submit_button = st.form_submit_button(label='Predict')

# Perform prediction when the form is submitted
if submit_button:
    # Normalize the inputs for the model
    X_input = np.array([[last_cycle_length, last_menstruation_length]])
    X_input = (X_input - np.array([28, 5])) / np.array([7, 2])  # Assumes mean=28,5 and std=7,2 for normalization

    prediction = model.predict(X_input)
    predicted_cycle_length = prediction[0][0] * 7 + 28
    predicted_menstruation_length = prediction[0][1] * 2 + 5

    st.subheader("Predicted Results")
    st.write(f"Predicted Cycle Length: **{predicted_cycle_length:.1f} days**")
    st.write(f"Predicted Menstruation Length: **{predicted_menstruation_length:.1f} days**")
