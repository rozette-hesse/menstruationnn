import streamlit as st
import numpy as np
from datetime import datetime, timedelta, date
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods.")

start_dates = []
end_dates = []

# Collect period data from user
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

# Phase detection function
def get_cycle_phase(today, last_start, avg_cycle):
    day_in_cycle = (today - last_start).days % avg_cycle
    if day_in_cycle <= 5:
        return "Menstrual Phase"
    elif day_in_cycle <= 13:
        return "Follicular Phase"
    elif day_in_cycle <= 16:
        return "Ovulation Phase"
    elif day_in_cycle <= avg_cycle:
        return "Luteal Phase"
    else:
        return "Unknown Phase"

if st.button("Predict Next Period"):
    try:
        # Validation
        if any(date is None for date in start_dates + end_dates):
            st.warning("⚠️ Please fill in all 4 start and end dates.")
        else:
            # Calculate features for last 3 periods
            features = []
            for i in range(1, 4):
                cycle_length = (start_dates[i] - start_dates[i - 1]).days
                duration = (end_dates[i] - start_dates[i]).days
                features.append([cycle_length, duration])

            input_data = np.array(features).reshape(1, 3, 2)
            predicted_days_until_next = int(model.predict(input_data)[0][0])

            latest_start = max(start_dates)
            next_start_date = latest_start + timedelta(days=predicted_days_until_next)
            st.success(f"📅 Your next period is predicted to start on **{next_start_date.strftime('%B %d, %Y')}**")

            # Predict current phase
            today = date.today()
            avg_cycle_length = int(np.mean([(start_dates[i] - start_dates[i - 1]).days for i in range(1, 4)]))
            phase = get_cycle_phase(today, latest_start, avg_cycle_length)
            st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
