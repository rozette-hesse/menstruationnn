import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import os
import pathlib

# Determine base directory (folder where this script lives)
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Build path to model file relative to this directory
model_path = BASE_DIR / "final_model.h5"

# Load model
if not model_path.exists():
    st.error(f"Model file not found at {model_path!s}")
    st.stop()

model = tf.keras.models.load_model(str(model_path), compile=False)

st.title("🩸 Menstrual Cycle Predictor")
st.write("Enter the **start and end dates** of your last 4 periods:")

periods = []
for i in range(4):
    st.subheader(f"Period {i + 1}")
    start = st.date_input(f"Start Date {i + 1}", key=f"start_{i}")
    end = st.date_input(f"End Date {i + 1}", key=f"end_{i}")
    periods.append((start, end))

if st.button("Predict Next Period"):
    try:
        # Ensure no missing values
        if any(s is None or e is None for s, e in periods):
            st.warning("⚠️ Please fill in all start and end dates for 4 periods.")
        else:
            # Sort by start date ascending (oldest → newest)
            periods_sorted = sorted(periods, key=lambda x: x[0])
            starts = [p[0] for p in periods_sorted]
            ends = [p[1] for p in periods_sorted]

            # Build input for model: 3 cycles (intervals) × 2 features (cycle_length, period_length)
            features = []
            for i in range(1, 4):
                cycle_len = (starts[i] - starts[i - 1]).days
                period_len = (ends[i] - starts[i]).days
                features.append([cycle_len, period_len])

            input_data = np.array(features, dtype=float).reshape((1, 3, 2))

            predicted_gap = int(model.predict(input_data)[0, 0])
            last_start = starts[-1]
            next_start = last_start + timedelta(days=predicted_gap)

            st.success(f"📅 Next period predicted to start: **{next_start.strftime('%B %d, %Y')}**")

            # Determine cycle phase
            today = datetime.today().date()
            days_since_last = (today - last_start).days
            avg_cycle = int(np.mean([f[0] for f in features]))
            avg_period = int(np.mean([f[1] for f in features]))

            if days_since_last < 0:
                phase = "Before most recent period"
            elif days_since_last <= avg_period:
                phase = "Menstrual Phase"
            elif days_since_last <= avg_period + 7:
                phase = "Follicular Phase"
            elif days_since_last <= avg_period + 14:
                phase = "Ovulation Phase"
            elif days_since_last <= avg_cycle:
                phase = "Luteal Phase"
            else:
                phase = "Awaiting next cycle"

            st.info(f"📅 Today is {today.strftime('%B %d, %Y')}. Likely in: **{phase}**")

    except Exception as e:
        st.error(f"Error: {e}")
