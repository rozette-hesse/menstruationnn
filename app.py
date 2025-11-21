import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta

# Load model
model = tf.keras.models.load_model("/mnt/data/final_model.h5")

st.title("🩸 Menstrual Cycle Tracker")
st.markdown("Please enter the **start and end dates** of your last 4 periods.")

# Function to collect dates
def get_period_input(index):
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(f"Start Date {index}", key=f"start_{index}")
    with col2:
        end_date = st.date_input(f"End Date {index}", key=f"end_{index}")
    return start_date, end_date

# Collect 4 periods
periods = []
for i in range(1, 5):
    st.subheader(f"Period {i}")
    periods.append(get_period_input(i))

if st.button("Predict Next Period"):
    try:
        # Prepare features: duration and gap between each period
        durations = [(end - start).days for start, end in periods]
        gaps = [(periods[i][0] - periods[i+1][0]).days for i in range(3)]  # gaps between start dates

        features = np.array([durations[0], gaps[0], durations[1], gaps[1], durations[2], gaps[2]]).reshape(1, 3, 2)
        
        # Predict days until next period
        predicted_days_until_next = int(model.predict(features)[0][0])
        
        # Calculate next period start date from most recent one
        latest_start_date = periods[0][0]
        next_start_date = latest_start_date + timedelta(days=predicted_days_until_next)

        # Output result
        st.success(f"📅 Your next period is predicted to start on **{next_start_date.strftime('%B %d, %Y')}**")

        # Determine current cycle phase
        today = datetime.today().date()
        days_since_last_period = (today - latest_start_date).days

        if days_since_last_period < 0:
            phase = "Before Menstrual Phase"
        elif 0 <= days_since_last_period <= 5:
            phase = "Menstrual Phase"
        elif 6 <= days_since_last_period <= 13:
            phase = "Follicular Phase"
        elif 14 <= days_since_last_period <= 16:
            phase = "Ovulation Phase"
        elif 17 <= days_since_last_period <= 28:
            phase = "Luteal Phase"
        else:
            phase = "Unknown Phase"

        st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
