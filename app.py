import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
import os

# Title and instructions
st.title("🩸 Menstrual Cycle Predictor")
st.markdown("Please enter the **start and end dates** of your last 4 periods.")

# Function to parse dates safely
def parse_date(date_str):
    return datetime.datetime.strptime(date_str, "%Y/%m/%d")

# Collect user input for 4 periods
periods = []
for i in range(1, 5):
    st.header(f"Period {i}")
    start_date = st.date_input(f"Start Date {i}", key=f"start_{i}")
    end_date = st.date_input(f"End Date {i}", key=f"end_{i}")
    periods.append((start_date, end_date))

# Load the model safely
model_path = os.path.abspath("final_model.h5")

# Debug info in sidebar
st.sidebar.write("📂 Current Directory Files:", os.listdir())
st.sidebar.write("📄 Model Path:", model_path)

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Calculate average cycle length and period length
cycle_lengths = [(periods[i][0] - periods[i+1][0]).days for i in range(3)]
period_lengths = [(end - start).days for start, end in periods]
avg_cycle_length = int(np.mean(cycle_lengths))
avg_period_length = int(np.mean(period_lengths))

# Predict next period start date
last_period_start = max([p[0] for p in periods])
predicted_start = last_period_start + datetime.timedelta(days=avg_cycle_length)

# Show prediction
if st.button("Predict Next Period"):
    st.success(f"📅 Next period predicted to start: **{predicted_start.strftime('%B %d, %Y')}**")

    # Determine phase based on today
    today = datetime.date.today()
    days_since_period = (today - last_period_start).days

    if days_since_period < avg_period_length:
        phase = "Menstrual Phase"
    elif days_since_period < avg_period_length + 7:
        phase = "Follicular Phase"
    elif days_since_period < avg_cycle_length - 14:
        phase = "Ovulation Phase"
    else:
        phase = "Luteal Phase"

    st.info(f"📅 Today is {today.strftime('%B %d, %Y')}. Likely in: **{phase}**")
