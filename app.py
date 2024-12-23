from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model
model = load_model('model_pupuk.h5')

# Load data untuk scaler
data = pd.read_csv('DataPupuk.csv', sep=';', encoding='latin-1')
X = data['Luas Tanah (m²)'].values.reshape(-1, 1)
y = data[['Banyak Pupuk (kg)', 'Air (liter)', 'Waktu (hari)']].values

# Normalisasi data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(X)
scaler_y.fit(y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    luas_tanah = float(request.form['luas_tanah'])
    luas_tanah_scaled = scaler_X.transform([[luas_tanah]])
    
    # Reshape untuk LSTM
    luas_tanah_scaled = luas_tanah_scaled.reshape((1, 1, 1))
    
    # Prediksi
    prediksi_scaled = model.predict(luas_tanah_scaled)
    prediksi = scaler_y.inverse_transform(prediksi_scaled)
    
    banyak_pupuk = prediksi[0][0]
    air = prediksi[0][1]
    waktu = prediksi[0][2]
    
    return render_template('index.html', luas_tanah=luas_tanah, banyak_pupuk=banyak_pupuk, air=air, waktu=waktu)

if __name__ == '__main__':
    app.run(debug=True)