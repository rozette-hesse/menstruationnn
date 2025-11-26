import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Load model safely
model_path = "final_model.h5"
try:
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

st.title("Period Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods.")

# Input for 4 periods
def get_period_input(period_num):
    start = st.date_input(f"Start Date {period_num}", key=f"start{period_num}")
    end = st.date_input(f"End Date {period_num}", key=f"end{period_num}")
    return start, end

periods = []
for i in range(1, 5):
    start, end = get_period_input(i)
    periods.append((start, end))

# Calculate cycle lengths
cycle_lengths = []
for i in range(len(periods) - 1):
    diff = (periods[i][0] - periods[i + 1][0]).days
    cycle_lengths.append(abs(diff))

# Predict next period start
def predict_next_period(cycle_lengths):
    if len(cycle_lengths) < 3:
        st.warning("Need 4 periods (3 cycles) to make a prediction.")
        return None
    avg_cycle = np.mean(cycle_lengths)
    last_start = periods[0][0]
    next_start = last_start + pd.Timedelta(days=avg_cycle)
    return next_start

if st.button("Predict Next Period"):
    try:
        # Prepare model input
        start_dates = [p[0].toordinal() for p in periods]
        end_dates = [p[1].toordinal() for p in periods]
        inputs = np.array([start_dates[:3], end_dates[:3]]).T.reshape(1, 3, 2)

        # Predict
        prediction = model.predict(inputs)
        predicted_days = int(prediction[0][0])
        next_start = periods[0][0] + pd.Timedelta(days=predicted_days)

        st.success(f"📅 Next period predicted to start: **{next_start.strftime('%B %d, %Y')}**")

        # Estimate phase
        today = datetime.today().date()
        days_since_start = (today - periods[0][0]).days
        if days_since_start < 0:
            phase = "Pre-period"
        elif days_since_start <= (periods[0][1] - periods[0][0]).days:
            phase = "Menstrual Phase"
        elif days_since_start <= 14:
            phase = "Follicular Phase"
        elif days_since_start <= 21:
            phase = "Ovulation Phase"
        else:
            phase = "Luteal Phase"

        st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. Likely in: **{phase}**")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
