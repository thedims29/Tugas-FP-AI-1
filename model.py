# Import Library
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('DataPupuk.csv', sep=';', encoding='latin-1')
X = data['Luas Tanah (m²)'].values.reshape(-1, 1)
y = data[['Banyak Pupuk (kg)', 'Air (liter)', 'Waktu (hari)']].values

# Normalisasi data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Membuat dataset untuk LSTM
X_lstm = []
y_lstm = []

for i in range(1, len(X_scaled)):
    X_lstm.append(X_scaled[i-1])
    y_lstm.append(y_scaled[i])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Reshape untuk LSTM [samples, time steps, features]
X_lstm = X_lstm.reshape((X_lstm.shape[0], 1, X_lstm.shape[1]))

# Membangun model LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dense(3))  # Output untuk 3 variabel
model.compile(optimizer='adam', loss='mse', metrics=['mae']) #Menambahkan MAE sebagai metrik

# Melatih model dan tampilkan epoch dan metrik
history = model.fit(X_lstm, y_lstm, epochs=200, verbose=1) # verbose=1 untuk menampilkan progress bar

# Plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot MAE
plt.plot(history.history['mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


# Menyimpan model
model.save('model_pupuk.h5')