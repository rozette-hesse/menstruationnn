import streamlit as st
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# Load trained model (disable compilation for safety)
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Enter the **start dates** of your last 4 periods to predict your next cycle.")

# Get 4 dates from the user
dates = []
for i in range(4):
    date = st.date_input(f"Period {i + 1} Start Date", key=f"date{i}")
    dates.append(date)

# Sort dates just in case
dates = sorted(dates)

# Predict button
if st.button("Predict Next Period"):
    # Convert to days between periods
    try:
        cycle_lengths = [
            (dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)
        ]
        
        # Simple average cycle length
        avg_cycle_length = int(np.mean(cycle_lengths))

        # Predict next date
        last_period = dates[-1]
        predicted_next = last_period + np.timedelta64(avg_cycle_length, 'D')

        st.success(f"🗓️ Predicted next period start date: **{predicted_next.strftime('%B %d, %Y')}**")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
