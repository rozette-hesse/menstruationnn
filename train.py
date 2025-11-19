import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from utils import read_period_file, make_train_test_sets
import os

# === LOAD DATA ===
data = read_period_file('calendar.txt')
train_X, train_y, test_X, test_y = make_train_test_sets(data)

# === NORMALIZE DATA ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
train_X = scaler_X.fit_transform(train_X)
test_X = scaler_X.transform(test_X)
train_y = scaler_y.fit_transform(train_y)
test_y = scaler_y.transform(test_y)

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# === BUILD MODEL ===
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(train_y.shape[1]))
model.compile(optimizer='adam', loss='mse', metrics=[mean_absolute_error, mean_absolute_percentage_error])

# === TRAIN MODEL ===
history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y), verbose=1)

# === EVALUATE MODEL ===
pred_y = model.predict(test_X)
pred_y = scaler_y.inverse_transform(pred_y)
test_y = scaler_y.inverse_transform(test_y)

mae = mean_absolute_error(test_y, pred_y)
mape = mean_absolute_percentage_error(test_y, pred_y)
print("Mean Absolute Error:", mae)
print("Mean Absolute Percentage Error:", mape)

# === SAVE MODEL ===
os.makedirs("models", exist_ok=True)
model.save("models/final_model.h5")
print("Model saved to models/final_model.h5")
