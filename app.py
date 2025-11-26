import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta

# Load the model with custom objects
@st.cache_resource
def load_model():
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'accuracy': tf.keras.metrics.Accuracy()
    }
    return tf.keras.models.load_model("final_model.h5", custom_objects=custom_objects)

model = load_model()

st.title("Menstrual Cycle Predictor")

# Input periods
periods = []
for i in range(1, 5):
    st.subheader(f"Period {i}")
    start_date = st.date_input(f"Start Date {i}", key=f"start_{i}")
    end_date = st.date_input(f"End Date {i}", key=f"end_{i}")
    periods.append((start_date, end_date))

# Calculate features
X = []
for i in range(len(periods) - 1):
    curr_start = periods[i][0]
    curr_end = periods[i][1]
    next_start = periods[i + 1][0]
    
    duration = (curr_end - curr_start).days
    cycle_length = (next_start - curr_start).days
    X.append([duration, cycle_length])

X = np.array(X)

if st.button("Predict Next Period"):
    try:
        # Normalize input
        X_input = X[-3:].reshape(1, 3, 2).astype(np.float32)
        predicted_gap = model.predict(X_input)[0][0]  # days to next period

        last_start = periods[-1][0]
        predicted_start = last_start + timedelta(days=int(predicted_gap))

        st.success(f"📅 Next period predicted to start: **{predicted_start.strftime('%B %d, %Y')}**")

        # Estimate phase
        today = datetime.today().date()
        days_since_last = (today - last_start).days

        if days_since_last <= 5:
            phase = "Menstrual Phase"
        elif days_since_last <= 13:
            phase = "Follicular Phase"
        elif days_since_last <= 20:
            phase = "Ovulation Phase"
        else:
            phase = "Luteal Phase"

        st.info(f"📅 Today is {today.strftime('%B %d, %Y')}. Likely in: **{phase}**")

    except Exception as e:
        st.error(f"❌ Failed to predict: {e}")
