import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load the model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods.")

start_dates = []
end_dates = []

# Collect 4 periods
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

if st.button("Predict Next Period"):
    try:
        if any(date is None for date in start_dates + end_dates):
            st.warning("⚠️ Please fill all start and end dates.")
        else:
            # Sort periods by start date (latest last)
            periods = sorted(zip(start_dates, end_dates), key=lambda x: x[0])
            start_dates, end_dates = zip(*periods)

            # Compute 3 cycle lengths and 3 durations
            cycle_lengths = [(start_dates[i + 1] - start_dates[i]).days for i in range(3)]
            durations = [(end_dates[i] - start_dates[i]).days for i in range(3)]

            # Prepare model input
            input_data = np.array([[cycle_lengths[i], durations[i]] for i in range(3)], dtype=np.float32)
            input_data = input_data.reshape((1, 3, 2))  # Shape: (1, 3, 2)

            prediction = model.predict(input_data)
            predicted_days_until_next = int(prediction[0][0])

            # Use the most recent period's start date
            last_period_start = max(start_dates)
            next_start_date = last_period_start + timedelta(days=predicted_days_until_next)

            st.success(f"📅 Your next period is predicted to start on **{next_start_date.strftime('%B %d, %Y')}**")

            # Determine current phase
            today = datetime.today().date()
            days_since_last_period = (today - last_period_start).days
            avg_cycle_length = int(np.mean(cycle_lengths))

            if days_since_last_period < 0:
                phase = "Before Last Period"
            elif days_since_last_period <= 5:
                phase = "Menstrual Phase"
            elif days_since_last_period <= 13:
                phase = "Follicular Phase"
            elif days_since_last_period <= 16:
                phase = "Ovulation Phase"
            elif days_since_last_period <= avg_cycle_length:
                phase = "Luteal Phase"
            else:
                phase = "Awaiting Period"

            st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
