import streamlit as st
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import os

# Debug: show working directory and files
st.sidebar.write("📂 Working Directory:", os.getcwd())
st.sidebar.write("📄 Files:", os.listdir())

# Load model from current directory
model_path = os.path.join(os.path.dirname(__file__), "final_model.h5")
model = tf.keras.models.load_model(model_path)

st.title("Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods.")

dates = []

# Get 4 periods
for i in range(4):
    st.subheader(f"Period {i+1}")
    start_date = st.date_input(f"Start Date {i+1}", key=f"start_{i}")
    end_date = st.date_input(f"End Date {i+1}", key=f"end_{i}")
    dates.append((start_date, end_date))

if st.button("Predict Next Period"):
    try:
        cycle_lengths = []
        period_lengths = []

        for i in range(1, 4):
            delta = (dates[i][0] - dates[i-1][0]).days
            cycle_lengths.append(delta)

        for start, end in dates:
            period_lengths.append((end - start).days)

        avg_cycle = sum(cycle_lengths) / len(cycle_lengths)
        avg_period = sum(period_lengths) / len(period_lengths)

        # Prepare input for model (shape must match model input: (1, 3, 2))
        last_cycle = (dates[-1][0] - dates[-2][0]).days
        last_period = (dates[-1][1] - dates[-1][0]).days

        input_data = [[[avg_cycle, avg_period], [last_cycle, last_period], [avg_cycle, avg_period]]]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        prediction = model.predict(input_tensor)[0][0]
        predicted_start = dates[-1][0] + timedelta(days=int(prediction))

        st.success(f"📅 Next period predicted to start: **{predicted_start.strftime('%B %d, %Y')}**")

        # Optional: Show current phase
        today = datetime.now().date()
        days_since_last = (today - dates[-1][0]).days
        if days_since_last <= avg_period:
            phase = "Menstrual Phase"
        elif days_since_last <= 14:
            phase = "Follicular Phase"
        elif days_since_last <= 21:
            phase = "Ovulation Phase"
        else:
            phase = "Luteal Phase"

        st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. Likely in: **{phase}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")
