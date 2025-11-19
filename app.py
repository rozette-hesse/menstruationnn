import streamlit as st
import numpy as np
from keras.models import load_model
from utils import make_prediction_from_user_input


# Load model
model = load_model("final_model.h5")

# UI
st.title("Menstrual Cycle Predictor")
st.write("Enter the last few period cycle stats:")

# Input fields
cycle_lengths = st.text_input("Cycle lengths (comma-separated)", "28,29,27")
menstruation_lengths = st.text_input("Menstruation lengths (comma-separated)", "5,5,6")

if st.button("Predict Next Period"):
    try:
        # Parse inputs
        cycle_data = list(map(int, cycle_lengths.strip().split(",")))
        menstruation_data = list(map(int, menstruation_lengths.strip().split(",")))

        if len(cycle_data) != len(menstruation_data):
            st.error("Cycle and menstruation lengths must have the same number of entries.")
        elif len(cycle_data) < 3:
            st.error("Please enter at least 3 records for better prediction.")
        else:
            input_data = np.array(list(zip(cycle_data, menstruation_data)))
            input_data = input_data[-3:].reshape(1, 3, 2)

            prediction = model.predict(input_data)[0]
            predicted_cycle = int(round(prediction[0]))
            predicted_menstruation = int(round(prediction[1]))

            st.success(f"Predicted cycle length: {predicted_cycle} days")
            st.success(f"Predicted menstruation length: {predicted_menstruation} days")
    except Exception as e:
        st.error(f"Error: {e}")
