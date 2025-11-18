import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from keras.models import load_model

# === Utility Functions ===
def calculate_lengths(periods):
    cycle_lengths = []
    m_lengths = []
    periods.sort()  # Ensure chronological order
    for i in range(1, len(periods)):
        prev_start = datetime.strptime(periods[i-1][0], "%Y-%m-%d")
        curr_start = datetime.strptime(periods[i][0], "%Y-%m-%d")
        curr_end = datetime.strptime(periods[i][1], "%Y-%m-%d")

        cycle_lengths.append((curr_start - prev_start).days)
        m_lengths.append((curr_end - curr_start).days + 1)
    return np.array([[c, m] for c, m in zip(cycle_lengths, m_lengths)])

def create_dataset(sequence, steps=3):
    x = []
    for i in range(len(sequence) - steps):
        x.append(sequence[i:i+steps])
    return np.array(x)

def get_phase(start_date, today, cycle_len, men_len):
    days_since = (today - start_date).days % cycle_len
    if days_since < men_len:
        return "Menstruation"
    elif days_since < men_len + 5:
        return "Follicular"
    elif days_since < cycle_len - 14:
        return "Ovulation"
    else:
        return "Luteal"

# === Streamlit App ===
st.set_page_config(page_title="Menstrual Cycle Predictor")
st.title("🔮 Menstrual Cycle Predictor")

st.markdown("""
Enter **at least 4** previous cycle periods (start and end dates). The model will predict your next cycle and show your current cycle phase.
""")

with st.form("cycle_form"):
    num_cycles = st.number_input("How many cycles would you like to enter?", min_value=4, step=1, value=4)
    dates = []
    for i in range(num_cycles):
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input(f"Cycle {i+1} Start", key=f"start_{i}")
        with col2:
            end = st.date_input(f"Cycle {i+1} End", key=f"end_{i}")
        dates.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))

    submitted = st.form_submit_button("Predict Next Cycle")

if submitted:
    try:
        data = calculate_lengths(dates)
        x_input = create_dataset(data, 3)

        if len(x_input) == 0:
            st.error("Please enter at least 4 valid cycles.")
        else:
            model = load_model("models/lstm_model_4000.h5")
            pred = model.predict(np.expand_dims(x_input[-1], axis=0))[0]
            next_cycle, next_men = int(round(pred[0])), int(round(pred[1]))

            last_start = datetime.strptime(dates[-1][0], "%Y-%m-%d")
            pred_start = last_start + timedelta(days=next_cycle)
            pred_end = pred_start + timedelta(days=next_men - 1)

            st.success(f"**Predicted Next Period:** {pred_start.strftime('%Y-%m-%d')} to {pred_end.strftime('%Y-%m-%d')}\n")
            st.info(f"**Predicted Cycle Length:** {next_cycle} days | **Menstruation Length:** {next_men} days")

            today = datetime.today()
            phase = get_phase(last_start, today, next_cycle, next_men)
            st.warning(f"**Today's Cycle Phase:** {phase}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
