import streamlit as st
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Menstrual Cycle Predictor")
st.title("🩸 Predict Your Next Period")

# Load model
MODEL_PATH = "final_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Helper to collect date input
def get_period_input(label_prefix):
    start = st.date_input(f"Start Date {label_prefix}", key=f"start_{label_prefix}")
    end = st.date_input(f"End Date {label_prefix}", key=f"end_{label_prefix}")
    return start, end

# Collect 4 periods
periods = []
for i in range(1, 5):
    st.subheader(f"Period {i}")
    start, end = get_period_input(str(i))
    periods.append((start, end))

if st.button("Predict Next Period"):
    try:
        # Convert to durations and cycle lengths
        durations = [(end - start).days for start, end in periods]
        cycle_lengths = [(periods[i+1][0] - periods[i][0]).days for i in range(3)]

        # Use last cycle length as estimate for 4th
        cycle_lengths.append(cycle_lengths[-1])

        X = np.array([[d, c] for d, c in zip(durations, cycle_lengths)])
        X = np.expand_dims(X, axis=0)  # Shape: (1, 4, 2)

        # Predict number of days until next period
        predicted_days = model.predict(X)[0][0]
        predicted_days = int(round(predicted_days))

        # Add to latest known start date
        latest_start = periods[-1][0]
        predicted_date = latest_start + timedelta(days=predicted_days)

        st.success(f"📅 Next period predicted to start: **{predicted_date.strftime('%B %d, %Y')}**")

        # Show current phase
        today = datetime.today().date()
        days_since = (today - periods[-1][0]).days
        if days_since <= durations[-1]:
            phase = "Menstrual Phase"
        elif days_since <= durations[-1] + 7:
            phase = "Follicular Phase"
        elif days_since <= durations[-1] + 14:
            phase = "Ovulation Phase"
        else:
            phase = "Luteal Phase"

        st.info(f"📅 Today is **{today.strftime('%B %d, %Y')}**. Likely in: **{phase}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
