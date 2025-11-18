# train.py

import numpy as np
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import metrics
from utils import load_synthetic_data, read_period_file, make_train_test_sets, evaluate_predictions

# 1. Load synthetic data
train_x_synth, train_y_synth, _, _ = load_synthetic_data("synthetic_data.txt")

# 2. Load real period data
periods_real = read_period_file("calendar.txt")
train_x_real, train_y_real, test_x_real, test_y_real, last_known_period = make_train_test_sets(periods_real)

# 3. Combine synthetic and real data
N_real = 78  # adjust if fewer real cycles
train_x = np.array(train_x_synth.tolist() + train_x_real.tolist()[-N_real:])
train_y = np.array(train_y_synth.tolist() + train_y_real.tolist()[-N_real:])

# 4. Define model architecture
n_steps = train_x.shape[1]
n_features = train_x.shape[2]

model = Sequential()
model.add(LSTM(100, activation="relu", return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation="relu"))
model.add(Dense(n_features))
model.compile(optimizer="adam", loss="mse", metrics=[metrics.mae, metrics.mape])

# 5. Train the model
n_epochs = 4000
model.fit(train_x, train_y, epochs=n_epochs, verbose=2)

# 6. Evaluate the model
y_preds = model.predict(test_x_real, verbose=0)
predictions = [[int(round(i[0])), int(round(i[1]))] for i in y_preds]
acc = evaluate_predictions(test_y_real, predictions)

print("Accuracy of cycle length prediction:", round(acc[0], 4))
print("Accuracy of menstruation length prediction:", round(acc[1], 4))

# 7. Save model
model.save("lstm_combined_model.h5")
print("Saved model to disk: lstm_combined_model.h5")
