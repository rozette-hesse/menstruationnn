import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load model
model = load_model("final_model.h5", compile=False)

st.title("🧨 Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods.")

start_dates = []
end_dates = []

# Collect 4 periods of dates
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

if st.button("Predict Next Period"):
    try:
        if any(d is None for d in start_dates + end_dates):
            st.warning("⚠️ Please fill in all 4 start and end dates.")
        else:
            # Sort by date to ensure correct order
            periods = sorted(zip(start_dates, end_dates), key=lambda x: x[0])
            start_dates, end_dates = zip(*periods)

            # Calculate cycle lengths
            cycle_lengths = [(start_dates[i + 1] - start_dates[i]).days for i in range(3)]
            avg_cycle_length = int(np.mean(cycle_lengths))
            last_period_duration = (end_dates[-1] - start_dates[-1]).days

            # Prepare input and predict
            input_data = np.array(cycle_lengths + [last_period_duration], dtype=np.float32).reshape(1, 3, 1)
            predicted_days_until_next = int(model.predict(input_data)[0][0])

            # Predict next period start date
            latest_start = max(start_dates)
            predicted_start = latest_start + timedelta(days=predicted_days_until_next)

            st.success(f"🗓️ Your next period is predicted to start on **{predicted_start.strftime('%B %d, %Y')}**")

            # Phase Calculation
            today = datetime.today().date()
            days_since_last_start = (today - latest_start).days

            if days_since_last_start < 0:
                phase = "Pre-cycle (future date entered)"
            elif days_since_last_start <= 5:
                phase = "Menstrual Phase"
            elif days_since_last_start <= 13:
                phase = "Follicular Phase"
            elif days_since_last_start == 14:
                phase = "Ovulation Phase"
            elif days_since_last_start <= avg_cycle_length:
                phase = "Luteal Phase"
            else:
                phase = "Late Phase / Possible Delay"

            st.info(f"🗓️ Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase}**.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
