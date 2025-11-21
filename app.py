import streamlit as st
import numpy as np
from datetime import date, timedelta
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor & Phase Tracker")
st.write("Enter the start and end dates of your last 4 periods. We'll predict your next start date and show your current cycle phase.")

start_dates = []
end_dates = []

# Collect 4 periods from user input
for i in range(4):
    st.subheader(f"Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}", value=None)
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}", value=None)
    start_dates.append(start)
    end_dates.append(end)

if st.button("Predict & Check Phase"):
    if None in start_dates + end_dates:
        st.error("⚠️ Please fill in **all** 4 start and end dates.")
    else:
        # Sort periods by start date to maintain order
        periods = sorted(zip(start_dates, end_dates), key=lambda x: x[0])
        last_start = periods[-1][0]  # Most recent period start

        # Prepare input for model: 3 cycle lengths and durations
        input_features = [
            [(periods[i+1][0] - periods[i][0]).days, (periods[i][1] - periods[i][0]).days]
            for i in range(3)
        ]
        input_array = np.array([input_features], dtype=np.float32)  # shape (1, 3, 2)

        try:
            prediction = model.predict(input_array)
            predicted_days = int(prediction[0][0])
            next_start = last_start + timedelta(days=predicted_days)

            st.success(f"🗓️ Predicted next period start date: **{next_start.strftime('%B %d, %Y')}**")

            # Determine today's cycle phase
            today = date.today()
            days_since_last = (today - last_start).days

            if days_since_last < 0:
                st.warning("⚠️ The last period start date is in the future.")
            else:
                period_duration = (periods[-1][1] - periods[-1][0]).days
                mid_cycle = predicted_days // 2

                if days_since_last < period_duration:
                    phase = "Menstrual Phase"
                elif days_since_last < mid_cycle:
                    phase = "Follicular Phase"
                elif days_since_last < predicted_days - 1:
                    phase = "Luteal Phase"
                else:
                    phase = "Next Period Incoming"

                st.write(f"Today is cycle day #{days_since_last + 1}.")
                st.subheader(f"🔍 Current Cycle Phase: **{phase}**")

        except Exception as e:
            st.error(f"An error occurred while predicting: {e}")
