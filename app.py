import tensorflow as tf
import streamlit as st

# Path to model file
MODEL_PATH = "final_model.h5"

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "mse": tf.keras.metrics.MeanSquaredError(),
            "mean_squared_error": tf.keras.metrics.MeanSquaredError()  # in case it's needed
        }
    )
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()
