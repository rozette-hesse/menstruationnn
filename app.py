import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your **last 4 periods**.")

start_dates = []
end_dates = []

# Collect 4 start and end dates
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

if st.button("Predict Next Period"):
    try:
        if any(d is None for d in start_dates + end_dates):
            st.warning("⚠️ Please make sure all 4 start and end dates are filled.")
        else:
            # Sort by start date
            sorted_periods = sorted(zip(start_dates, end_dates), key=lambda x: x[0])
            start_dates, end_dates = zip(*sorted_periods)

            # Calculate cycle lengths
            cycle_lengths = [(start_dates[i + 1] - start_dates[i]).days for i in range(3)]
            durations = [(end_dates[i] - start_dates[i]).days for i in range(3)]
            avg_cycle = int(np.mean(cycle_lengths))
            avg_duration = int(np.mean(durations))

            # Reshape for model
            input_data = np.array([avg_cycle, avg_duration]).reshape(1, 2)

            predicted_days_until_next = int(model.predict(input_data)[0][0])
            last_start_date = max(start_dates)
            predicted_start = last_start_date + timedelta(days=predicted_days_until_next)

            st.success(f"📅 Your next period is predicted to start on **{predicted_start.strftime('%B %d, %Y')}**")

            # Identify today's phase
            today = datetime.today().date()
            current_period_start = max(start_dates)
            predicted_next = predicted_start
            cycle_length = (predicted_next - current_period_start).days

            days_into_cycle = (today - current_period_start).days
            phase = "Unknown Phase"

            if days_into_cycle < 0:
                phase = "Pre-cycle"
            elif 0 <= days_into_cycle <= avg_duration:
                phase = "Menstrual Phase"
            elif avg_duration < days_into_cycle <= 14:
                phase = "Follicular Phase"
            elif 14 < days_into_cycle <= 17:
                phase = "Ovulation Phase"
            elif 17 < days_into_cycle <= cycle_length:
                phase = "Luteal Phase"

            st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
