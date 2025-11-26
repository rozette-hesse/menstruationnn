import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf

def calculate_phase(today, last_start_date, average_cycle_length):
    days_since_last_period = (today - last_start_date).days
    cycle_day = days_since_last_period % average_cycle_length

    if cycle_day <= 5:
        return "Menstrual Phase"
    elif 6 <= cycle_day <= 13:
        return "Follicular Phase"
    elif 14 <= cycle_day <= 16:
        return "Ovulation Phase"
    elif 17 <= cycle_day <= average_cycle_length:
        return "Luteal Phase"
    else:
        return "Unknown Phase"

def get_date_input(label):
    return st.date_input(label, value=None, key=label)

def main():
    st.title("Menstrual Cycle Predictor")
    st.write("Please enter the **start and end dates** of your last 4 periods.")

    start_dates = []
    end_dates = []

    for i in range(1, 5):
        st.subheader(f"Period {i}")
        start = get_date_input(f"Start Date {i}")
        end = get_date_input(f"End Date {i}")
        if start and end:
            start_dates.append(start)
            end_dates.append(end)

    if st.button("Predict Next Period"):
        try:
            if len(start_dates) < 4:
                st.warning("Please enter all 4 periods.")
                return

            # Calculate cycle gaps between starts
            cycle_gaps = [(start_dates[i] - start_dates[i+1]).days for i in range(3)]
            avg_cycle_length = int(np.mean(cycle_gaps))

            # Load model
            model = tf.keras.models.load_model("/mnt/data/final_model.h5")

            # Prepare input shape
            input_data = np.array(cycle_gaps).reshape(1, 3, 1)
            predicted_gap = int(model.predict(input_data)[0][0])

            # Get latest start date
            last_start = max(start_dates)
            next_start = last_start + timedelta(days=predicted_gap)

            # If predicted date is in the past, suggest the next one
            if next_start <= datetime.today().date():
                next_start += timedelta(days=predicted_gap)

            st.success(f"📅 Next period predicted to start: **{next_start.strftime('%B %d, %Y')}**")

            # Phase tracking
            today = datetime.today().date()
            phase = calculate_phase(today, last_start, avg_cycle_length)
            st.info(f"📅 Today is {today.strftime('%B %d, %Y')}. Likely in: **{phase}**")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
