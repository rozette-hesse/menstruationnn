import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods.")

start_dates = []
end_dates = []

# Collect 4 periods from the user
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

if st.button("Predict Next Period"):
    try:
        if any(date is None for date in start_dates + end_dates):
            st.warning("⚠️ Please fill in all the dates.")
        else:
            # Create model input shape (3, 2): 3 cycles with [cycle_length, duration]
            input_data = np.array([
                [
                    [(start_dates[1] - start_dates[0]).days, (end_dates[0] - start_dates[0]).days],
                    [(start_dates[2] - start_dates[1]).days, (end_dates[1] - start_dates[1]).days],
                    [(start_dates[3] - start_dates[2]).days, (end_dates[2] - start_dates[2]).days],
                ]
            ])

            prediction = model.predict(input_data)
            predicted_days_until_next = int(prediction[0][0])

            # Use most recent period (start_dates[3]) to forecast
            next_start_date = start_dates[3] + timedelta(days=predicted_days_until_next)
            st.success(f"📅 Your next period is predicted to start on **{next_start_date.strftime('%B %d, %Y')}**")

            # Estimate today's phase based on last period
            today = datetime.now().date()
            last_start = start_dates[3]
            last_end = end_dates[3]
            cycle_length = predicted_days_until_next
            days_since_last_period = (today - last_start).days

            if days_since_last_period < 0:
                phase = "Before Period Start"
            elif days_since_last_period <= (last_end - last_start).days:
                phase = "Menstrual"
            elif days_since_last_period <= 13:
                phase = "Follicular"
            elif days_since_last_period <= 15:
                phase = "Ovulation"
            elif days_since_last_period < cycle_length:
                phase = "Luteal"
            else:
                phase = "Unknown"

            st.info(f"📅 Today is {today.strftime('%B %d, %Y')}. You are likely in the **{phase} Phase**.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
