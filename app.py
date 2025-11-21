import streamlit as st
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta

def calculate_cycle_data(start_dates, end_dates):
    cycle_lengths = []
    period_lengths = []

    for i in range(1, len(start_dates)):
        cycle_length = (start_dates[i - 1] - start_dates[i]).days
        cycle_lengths.append(cycle_length)

    for i in range(len(start_dates)):
        period_length = (end_dates[i] - start_dates[i]).days
        period_lengths.append(period_length)

    avg_cycle_length = sum(cycle_lengths) / len(cycle_lengths)
    avg_period_length = sum(period_lengths) / len(period_lengths)

    return avg_cycle_length, avg_period_length

def determine_phase(today, last_period_start, avg_cycle_length, avg_period_length):
    days_since_last = (today - last_period_start).days

    if days_since_last < 0:
        return "Invalid"
    elif days_since_last <= avg_period_length:
        return "Menstrual Phase"
    elif days_since_last <= avg_period_length + 7:
        return "Follicular Phase"
    elif days_since_last <= avg_period_length + 14:
        return "Ovulation Phase"
    elif days_since_last < avg_cycle_length:
        return "Luteal Phase"
    else:
        return "Awaiting next cycle"

# Load the model
model = tf.keras.models.load_model("final_model.h5")

st.title("🩸 Menstrual Cycle Tracker")
st.markdown("Please enter the **start and end dates** of your last 4 periods.")

start_dates = []
end_dates = []

for i in range(4):
    st.subheader(f"Period {i + 1}")
    start_date = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end_date = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start_date)
    end_dates.append(end_date)

if st.button("Predict Next Period"):
    try:
        avg_cycle_length, avg_period_length = calculate_cycle_data(start_dates, end_dates)

        # Use most recent period start date (Period 1)
        last_period_start = start_dates[0]

        input_data = np.array([[avg_cycle_length, avg_period_length, (datetime.now().date() - last_period_start).days]])
        input_data = input_data.reshape((1, 3, 1)).astype(np.float32)

        predicted_gap = int(model.predict(input_data)[0][0])
        next_period_date = last_period_start + timedelta(days=predicted_gap)

        st.success(f"📅 Your next period is predicted to start on **{next_period_date.strftime('%B %d, %Y')}**")

        today = datetime.now().date()
        phase = determine_phase(today, last_period_start, avg_cycle_length, avg_period_length)
        st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
