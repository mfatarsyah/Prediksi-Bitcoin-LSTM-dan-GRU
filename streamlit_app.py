# =======================
# FILE: generate_dummy_model.py
# =======================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import joblib

# 1. Generate dummy Bitcoin-like data
np.random.seed(42)
data = np.cumsum(np.random.randn(2000) * 50 + 30000)  # Dummy Bitcoin prices
prices = pd.DataFrame(data, columns=['Close'])

# 2. Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(prices)

# 3. Create sequences (X, y) for training
sequence_length = 60
X = []
y = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4. Build simple Hybrid LSTM-GRU model
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(GRU(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# 5. Save the model and scaler
model.save("bitcoin_model.h5")
joblib.dump(scaler, "scaler.joblib")

print("Dummy model and scaler saved successfully.")
