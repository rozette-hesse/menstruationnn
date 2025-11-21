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

# Collect 4 periods of start and end dates
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

def get_cycle_phase(today, last_start, last_end, avg_cycle):
    period_length = (last_end - last_start).days
    ovulation_day = last_start + timedelta(days=avg_cycle - 14)

    if last_start <= today <= last_end:
        return "Menstrual Phase"
    elif last_end < today < ovulation_day - timedelta(days=3):
        return "Follicular Phase"
    elif ovulation_day - timedelta(days=3) <= today <= ovulation_day + timedelta(days=1):
        return "Ovulation Phase"
    elif ovulation_day + timedelta(days=1) < today < last_start + timedelta(days=avg_cycle):
        return "Luteal Phase"
    else:
        return "Unknown Phase"

if st.button("Predict Next Period"):
    try:
        if any(date is None for date in start_dates + end_dates):
            st.warning("⚠️ Please make sure all 4 start and end dates are filled in.")
        else:
            # Sort periods by start date descending
            sorted_periods = sorted(zip(start_dates, end_dates), key=lambda x: x[0])
            start_dates_sorted = [p[0] for p in sorted_periods]
            end_dates_sorted = [p[1] for p in sorted_periods]

            # Calculate average cycle length from start dates
            cycle_lengths = [(start_dates_sorted[i + 1] - start_dates_sorted[i]).days for i in range(3)]
            avg_cycle = int(np.mean(cycle_lengths))

            # Last period info
            last_start = start_dates_sorted[-1]
            last_end = end_dates_sorted[-1]
            last_duration = (last_end - last_start).days

            # Predict
            input_data = np.array([avg_cycle, last_duration], dtype=np.float32).reshape(1, 2)
            predicted_days_until_next = int(model.predict(input_data)[0][0])
            next_start_date = last_start + timedelta(days=predicted_days_until_next)

            st.success(f"\U0001F4C5 Your next period is predicted to start on **{next_start_date.strftime('%B %d, %Y')}**")

            # Determine today's phase
            today = datetime.today().date()
            phase = get_cycle_phase(today, last_start, last_end, avg_cycle)
            st.info(f"\U0001F4C5 Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
