import streamlit as st
from datetime import datetime, timedelta

st.title("🩸 Menstrual Cycle Predictor")

# Input
periods = []
for i in range(4, 0, -1):
    st.subheader(f"Period {5 - i}")
    start = st.date_input(f"Start Date {5 - i}", key=f"start{i}")
    end = st.date_input(f"End Date {5 - i}", key=f"end{i}")
    if start and end:
        periods.append((start, end))

if st.button("Predict Next Period") and len(periods) == 4:
    try:
        # Calculate cycle lengths based on start dates
        start_dates = [p[0] for p in periods]
        cycle_lengths = [(start_dates[i] - start_dates[i+1]).days for i in range(3)]
        avg_cycle_length = round(sum(cycle_lengths) / len(cycle_lengths))

        # Predict next period
        last_start = start_dates[0]
        predicted_next = last_start + timedelta(days=avg_cycle_length)

        st.success(f"📅 Next period predicted to start: **{predicted_next.strftime('%B %d, %Y')}**")
        today = datetime.now().date()
        days_since = (today - last_start).days

        # Phase estimation (simplified)
        if days_since <= 7:
            phase = "Menstrual Phase"
        elif days_since <= 14:
            phase = "Follicular Phase"
        elif days_since <= 21:
            phase = "Ovulation Phase"
        else:
            phase = "Luteal Phase"

        st.info(f"📅 Today is {today.strftime('%B %d, %Y')}. Likely in: **{phase}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
