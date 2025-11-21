import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Enter the **start and end dates** of your **last 4 periods**.")

start_dates = []
end_dates = []

# Get 4 periods from the user
for i in range(4):
    st.subheader(f"Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

if st.button("Predict Next Period"):
    try:
        if any(date is None for date in start_dates + end_dates):
            st.warning("⚠️ Please fill in all 4 start and end dates.")
        else:
            # Sort based on start dates
            sorted_data = sorted(zip(start_dates, end_dates))
            start_dates, end_dates = zip(*sorted_data)

            # Calculate average cycle length
            cycle_lengths = [(start_dates[i + 1] - start_dates[i]).days for i in range(3)]
            avg_cycle = int(np.mean(cycle_lengths))

            # Duration of most recent period
            last_duration = (end_dates[-1] - start_dates[-1]).days

            # Predict next period
            input_data = np.array([[[avg_cycle, last_duration],
                                    [avg_cycle, last_duration],
                                    [avg_cycle, last_duration]]])  # shape (1, 3, 2)
            prediction = model.predict(input_data)
            predicted_days = int(prediction[0][0])

            next_start_date = start_dates[-1] + timedelta(days=predicted_days)
            st.success(f"🗓️ Your next period is predicted to start on **{next_start_date.strftime('%B %d, %Y')}**")

            # Calculate today's cycle phase
            today = datetime.today().date()
            days_since_last_start = (today - start_dates[-1]).days

            if days_since_last_start < 0:
                phase = "Before current cycle started"
            elif days_since_last_start <= last_duration:
                phase = "Menstrual Phase"
            elif days_since_last_start <= 14:
                phase = "Follicular Phase"
            elif 14 < days_since_last_start <= avg_cycle:
                phase = "Luteal Phase"
            else:
                phase = "Next Period Incoming"

            st.info(f"📆 **Today is {today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
