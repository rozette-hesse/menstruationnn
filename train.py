import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from utils import read_period_file, make_train_test_sets, load_synthetic_data

# === Load real and synthetic data ===
real_periods = read_period_file("calendar.txt")
train_x_r, train_y_r, test_x_r, test_y_r, last_known_period = make_train_test_sets(real_periods)

train_x_s, train_y_s, test_x_s, test_y_s = load_synthetic_data("synthetic_data.txt")

# === Merge real + synthetic ===
train_x = np.concatenate((train_x_r, train_x_s), axis=0)
train_y = np.concatenate((train_y_r, train_y_s), axis=0)
test_x = np.concatenate((test_x_r, test_x_s), axis=0)
test_y = np.concatenate((test_y_r, test_y_s), axis=0)

# === Define Model ===
n_steps = train_x.shape[1]
n_features = train_x.shape[2]

model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
    LSTM(64, activation='relu'),
    Dense(n_features)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
)

# === Train ===
n_epochs = 5000
model.fit(train_x, train_y, epochs=n_epochs, verbose=2)

# === Evaluate ===
loss, mae, mape = model.evaluate(test_x, test_y)
print("Accuracy cycle length:", round(1 - mae, 4))
print("Accuracy menstruation length:", round(1 - mae, 4))

# === Save Model ===
model.save("final_model.h5")
print("Model saved to final_model.h5")
