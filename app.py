import streamlit as st
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model without compiling (avoids deserialization issues)
model_path = "./final_model.h5"
model = load_model(model_path, compile=False)

# --- App UI ---
st.title("🩸 Menstrual Cycle Predictor")
st.markdown("Please enter the start and end dates of your last 4 periods.")

periods = []

# Input for 4 periods
for i in range(4):
    st.subheader(f"Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    periods.append((start, end))

# Prediction logic
def predict_next_period(dates):
    cycle_lengths = []
    period_lengths = []

    for i in range(3):
        delta = (dates[i][0] - dates[i + 1][0]).days
        cycle_lengths.append(delta)

    for start, end in dates:
        period_lengths.append((end - start).days)

    avg_cycle = np.mean(cycle_lengths)
    avg_duration = np.mean(period_lengths)

    # Prepare model input
    X_input = np.array([[avg_cycle, avg_duration]])
    prediction = model.predict(X_input)

    return int(prediction[0][0])

# Determine phase logic
def determine_phase(start_date):
    today = datetime.date.today()
    days_since = (today - start_date).days

    if days_since < 0:
        return "Not started yet"
    elif days_since <= 5:
        return "Menstrual Phase"
    elif days_since <= 13:
        return "Follicular Phase"
    elif days_since <= 16:
        return "Ovulation Phase"
    elif days_since <= 28:
        return "Luteal Phase"
    else:
        return "Uncertain"

# Predict button
if st.button("Predict Next Period"):
    try:
        predicted_gap = predict_next_period(periods)
        latest_start = max([p[0] for p in periods])
        next_period_start = latest_start + datetime.timedelta(days=predicted_gap)

        st.success(f"📅 Next period predicted to start: **{next_period_start.strftime('%B %d, %Y')}**")

        today = datetime.date.today()
        phase = determine_phase(latest_start)
        st.info(f"📅 Today is {today.strftime('%B %d, %Y')}. Likely in: **{phase}**")

    except Exception as e:
        st.error(f"❌ An error occurred
