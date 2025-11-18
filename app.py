import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from keras.models import load_model
from utils import calculate_lengths, create_dataset

# Load trained model
model = load_model("models/lstm_4000.h5")

st.title("Menstrual Cycle Predictor")
st.markdown("Enter at least 4 past menstrual cycles to get predictions.")

# Input table for cycle dates
st.subheader("Enter Past Periods")
dates_input = st.text_area("Format: YYYY-MM-DD,YYYY-MM-DD per line for Start and End", 
                           placeholder="2025-09-01,2025-09-05\n2025-10-01,2025-10-05")

if dates_input:
    try:
        periods = [tuple(line.strip().split(",")) for line in dates_input.strip().split("\n") if line]
        # Ensure chronological order
        periods.sort(key=lambda x: datetime.strptime(x[0], "%Y-%m-%d"))

        if len(periods) < 4:
            st.warning("Please enter at least 4 periods (start and end dates).")
        else:
            # Calculate lengths
            cycle_lengths, m_lengths = calculate_lengths(periods)
            data = np.array([[c, m] for c, m in zip(cycle_lengths, m_lengths)])
            train_x, _ = create_dataset(data, 3)

            # Predict next
            last_seq = np.expand_dims(data[-3:], axis=0)
            y_pred = model.predict(last_seq, verbose=0)[0]
            next_cycle = int(round(y_pred[0]))
            next_menstruation = int(round(y_pred[1]))

            last_period_start = datetime.strptime(periods[-1][0], "%Y-%m-%d")
            predicted_start = last_period_start + timedelta(days=next_cycle)
            predicted_end = predicted_start + timedelta(days=next_menstruation - 1)

            st.success(f"Predicted Next Period: {predicted_start.strftime('%Y-%m-%d')} to {predicted_end.strftime('%Y-%m-%d')}")
            st.info(f"Predicted Cycle Length: {next_cycle} days")
            st.info(f"Predicted Menstruation Length: {next_menstruation} days")

            # Phase indicator
            today = datetime.today()
            days_since_last_start = (today - last_period_start).days
            if days_since_last_start < m_lengths[-1]:
                phase = "Menstruation"
            elif days_since_last_start < 14:
                phase = "Follicular"
            elif days_since_last_start < next_cycle:
                phase = "Luteal"
            else:
                phase = "New cycle expected soon"

            st.warning(f"Today's cycle phase: {phase}")

    except Exception as e:
        st.error(f"Error parsing input: {e}")
