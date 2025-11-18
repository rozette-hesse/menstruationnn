# train.py

from utils import *
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import metrics
from keras.models import load_model
import numpy as np
import os

# Load synthetic data
train_x_syn, train_y_syn, _, _ = load_synthetic_data("synthetic_data.txt")

# Load real data
periods = read_period_file("calendar.txt")
train_x_real, train_y_real, test_x, test_y, last_known_period = make_train_test_sets(periods)

# Combine synthetic and real data
train_x = np.array(train_x_syn.tolist() + train_x_real.tolist()[-78:])
train_y = np.array(train_y_syn.tolist() + train_y_real.tolist()[-78:])

# Training configuration
n_epochs = 4000
n_steps = 3
n_features = train_x.shape[2]

# Build LSTM model
model = Sequential()
model.add(LSTM(100, activation="relu", return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation="relu"))
model.add(Dense(n_features))
model.compile(optimizer="adam", loss="mse", metrics=[metrics.mae, metrics.mape])

# Train the model
model.fit(train_x, train_y, epochs=n_epochs, verbose=2)

# Evaluate
model.evaluate(test_x, test_y)

# Save the model
os.makedirs("models", exist_ok=True)
model_path = f"models/lstm_model_{n_epochs}.h5"
model.save(model_path)
print(f"Model saved to {model_path}")

# Predict
y_preds = model.predict(test_x, verbose=0)
predictions = [[int(round(i[0])), int(round(i[1]))] for i in y_preds]
accuracies = evaluate_predictions(test_y, predictions)

print("Accuracy of menstrual cycle length prediction:", round(accuracies[0], 4))
print("Accuracy of menstruation length prediction:", round(accuracies[1], 4))
