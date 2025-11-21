import streamlit as st
import numpy as np
from datetime import datetime, timedelta, date
from tensorflow.keras.models import load_model

# Load the model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods (most recent included).")

start_dates = []
end_dates = []

# Input for 4 periods
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

def get_cycle_phase(days_since_start):
    if days_since_start < 5:
        return "Menstrual Phase"
    elif days_since_start < 13:
        return "Follicular Phase"
    elif days_since_start < 17:
        return "Ovulation Phase"
    elif days_since_start < 28:
        return "Luteal Phase"
    else:
        return "Post-cycle"

if st.button("Predict Next Period"):
    if any(d is None for d in start_dates + end_dates):
        st.warning("⚠️ Please ensure all 4 start and end dates are entered.")
    else:
        # Sort periods by start date (just in case)
        periods = sorted(zip(start_dates, end_dates), key=lambda x: x[0])
        sorted_starts = [p[0] for p in periods]
        sorted_ends = [p[1] for p in periods]

        # Calculate average cycle length from 3 intervals
        cycle_lengths = [(sorted_starts[i+1] - sorted_starts[i]).days for i in range(3)]
        avg_cycle = int(np.mean(cycle_lengths))

        # Last period duration (most recent)
        last_duration = (sorted_ends[-1] - sorted_starts[-1]).days
        input_data = np.array([[avg_cycle, last_duration]], dtype=np.float32)

        # Predict next period
        predicted_days_until_next = int(model.predict(input_data)[0][0])
        next_start = sorted_starts[-1] + timedelta(days=predicted_days_until_next)

        st.success(f"📅 Your next period is predicted to start on **{next_start.strftime('%B %d, %Y')}**")

        # Identify current phase
        today = date.today()
        last_start = sorted_starts[-1]
        days_since_last = (today - last_start).days

        if days_since_last < 0:
            st.info("📅 Today is before your most recent recorded period.")
        else:
            phase = get_cycle_phase(days_since_last)
            st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")
