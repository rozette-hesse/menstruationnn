import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("final_model.h5", compile=False)

st.title("🩸 Menstrual Cycle Predictor")
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
        # Check for any missing input
        if any(date is None for date in start_dates + end_dates):
            st.warning("⚠️ Please make sure all 4 start and end dates are filled in.")
        else:
            # Calculate cycle lengths (difference between consecutive start dates)
            cycle_lengths = [(start_dates[i + 1] - start_dates[i]).days for i in range(3)]
            avg_cycle = int(np.mean(cycle_lengths))

            # Duration of the last period
            last_duration = (end_dates[-1] - start_dates[-1]).days

            # Prepare model input and predict
            input_data = np.array([[avg_cycle, last_duration]])
            prediction = model.predict(input_data)
            predicted_days_until_next = int(prediction[0][0])

            # Predict next start date
            next_start_date = start_dates[-1] + timedelta(days=predicted_days_until_next)

            st.success(f"🗓️ Your next period is predicted to start on **{next_start_date.strftime('%B %d, %Y')}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")
