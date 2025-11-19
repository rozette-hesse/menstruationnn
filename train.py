import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load data from file
def load_period_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    starts = []
    ends = []
    for line in lines:
        if line.startswith("Start"):
            starts.append(line.split(":")[-1].strip())
        elif line.startswith("End"):
            ends.append(line.split(":")[-1].strip())

    dates = pd.DataFrame({"start": pd.to_datetime(starts), "end": pd.to_datetime(ends)})
    dates = dates.sort_values("start").reset_index(drop=True)
    dates["cycle_length"] = dates["start"].diff().dt.days
    dates["menstruation_length"] = (dates["end"] - dates["start"]).dt.days
    dates = dates.dropna()
    return dates[["cycle_length", "menstruation_length"]]

# Prepare dataset
def prepare_data(df):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    joblib.dump(scaler, "scaler.save")

    X, y = [], []
    for i in range(1, len(data_scaled)):
        X.append(data_scaled[i-1])
        y.append(data_scaled[i])
    return np.array(X), np.array(y), scaler

# Train the model
def train_model(X, y, epochs=500):
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dense(y.shape[1])
    ])
    model.compile(optimizer=Adam(), loss='mse', metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
    model.fit(X, y, epochs=epochs, verbose=1, validation_split=0.2)
    model.save("final_model.h5")
    print("Model saved as final_model.h5")

# Main
if __name__ == '__main__':
    df = load_period_data("calendar.txt")
    X, y, scaler = prepare_data(df)
    train_model(X, y)
    print("Training complete.")
