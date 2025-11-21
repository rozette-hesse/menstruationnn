import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("final_model.h5", compile=False)

st.title("🧨 Menstrual Cycle Predictor")
st.write("Please enter the **start and end dates** of your last 4 periods.")

start_dates = []
end_dates = []

# Ask for 4 period start and end dates
for i in range(4):
    st.markdown(f"### Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    start_dates.append(start)
    end_dates.append(end)

if st.button("Predict Next Period"):
    try:
        if any(d is None for d in start_dates + end_dates):
            st.warning("⚠️ Please fill in all start and end dates.")
        else:
            # Sort periods based on start date (most recent last)
            periods = sorted(zip(start_dates, end_dates), key=lambda x: x[0])
            start_dates, end_dates = zip(*periods)

            # Calculate cycle lengths between start dates
            cycle_lengths = [(start_dates[i + 1] - start_dates[i]).days for i in range(3)]
            durations = [(end_dates[i] - start_dates[i]).days for i in range(3)]

            # Prepare input for model: shape must be (1, 3, 2)
            features = np.array([[cl, dur] for cl, dur in zip(cycle_lengths, durations)])
            input_data = features.reshape(1, 3, 2)

            prediction = model.predict(input_data)
            predicted_days_until_next = int(prediction[0][0])

            latest_start = max(start_dates)
            predicted_start = latest_start + timedelta(days=predicted_days_until_next)

            st.success(f"🗓️ Your next period is predicted to start on **{predicted_start.strftime('%B %d, %Y')}**")

            # Identify current cycle phase
            today = datetime.today().date()
            latest_duration = (end_dates[-1] - start_dates[-1]).days
            current_cycle_day = (today - start_dates[-1]).days

            if current_cycle_day < 0:
                phase = "Before cycle start"
            elif current_cycle_day <= latest_duration:
                phase = "Menstrual"
            elif current_cycle_day <= 13:
                phase = "Follicular"
            elif current_cycle_day <= 16:
                phase = "Ovulation"
            elif current_cycle_day <= predicted_days_until_next:
                phase = "Luteal"
            else:
                phase = "Unknown Phase"

            st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. You are likely in the **{phase} Phase**.")

    except Exception as e:
        st.error(f"An error occurred
