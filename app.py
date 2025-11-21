import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods.")

start_dates = []
end_dates = []

# Input fields for 4 periods
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}", value=None)
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}", value=None)
    start_dates.append(start)
    end_dates.append(end)

if st.button("Predict Next Period"):
    try:
        if any(d is None for d in start_dates + end_dates):
            st.warning("⚠️ Please fill in all 4 start and end dates.")
        else:
            # Create input: 3 periods worth of (cycle_length, period_duration)
            input_data = [
                [(start_dates[1] - start_dates[0]).days, (end_dates[0] - start_dates[0]).days],
                [(start_dates[2] - start_dates[1]).days, (end_dates[1] - start_dates[1]).days],
                [(start_dates[3] - start_dates[2]).days, (end_dates[2] - start_dates[2]).days],
            ]
            input_data = np.array([input_data], dtype=np.float32)  # Shape: (1, 3, 2)

            # Predict next period start offset
            prediction = model.predict(input_data)
            predicted_days_until_next = int(prediction[0][0])

            next_start_date = start_dates[3] + timedelta(days=predicted_days_until_next)
            st.success(f"🗓️ Your next period is predicted to start on **{next_start_date.strftime('%B %d, %Y')}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        
