import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Harga Bitcoin",
    page_icon="🔋",
    layout="wide"
)

st.title("Prediksi Harga Bitcoin Menggunakan Hybrid LSTM-GRU")

# =====================================================================================
# Fungsi untuk memuat model dan scaler
# =====================================================================================
@st.cache_resource
def load_model_hybrid(path='bitcoin_model.h5'):
    return load_model(path)

@st.cache_resource
def load_scaler(path='scaler.joblib'):
    return joblib.load(path)

@st.cache_data
def load_bitcoin_data():
    start_date = '2020-01-01'
    end_date = date.today().strftime('%Y-%m-%d')
    data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    return data[['Close']]

# =====================================================================================
# Tampilan Visualisasi Historis
# =====================================================================================
st.subheader("Visualisasi Harga Bitcoin (2020 - Sekarang)")
data = load_bitcoin_data()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Harga Historis', line=dict(color='deepskyblue')))
fig.update_layout(xaxis_title='Tanggal', yaxis_title='Harga (USD)', yaxis_tickprefix='$', template='plotly_dark')
st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# Input Prediksi
# =====================================================================================
st.markdown("### Masukkan Tanggal untuk Prediksi")
future_date = st.date_input("Pilih Tanggal Prediksi", min_value=date.today() + timedelta(days=1), max_value=date.today() + timedelta(days=30))
predict_button = st.button("Click here to Predict")

if predict_button:
    model = load_model_hybrid()
    scaler = load_scaler()
    sequence_length = 60

    if len(data) < sequence_length:
        st.error("Data historis tidak cukup untuk prediksi.")
    else:
        last_sequence = data['Close'].values[-sequence_length:]
        scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
        current_batch = scaled_sequence.reshape(1, sequence_length, 1)

        days_to_predict = (future_date - data.index[-1].date()).days
        predictions = []

        for _ in range(days_to_predict):
            pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

        predicted_prices = scaler.inverse_transform(predictions)

        # Visualisasi Hasil
        st.subheader("Hasil Prediksi Harga Bitcoin")
        pred_dates = [data.index[-1] + timedelta(days=i+1) for i in range(days_to_predict)]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Harga Historis', line=dict(color='deepskyblue')))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=predicted_prices.flatten(), name='Prediksi', line=dict(color='tomato', dash='dash')))
        fig_pred.update_layout(xaxis_title='Tanggal', yaxis_title='Harga (USD)', yaxis_tickprefix='$', template='plotly_dark')
        st.plotly_chart(fig_pred, use_container_width=True)

        st.metric("Prediksi Harga pada {}".format(future_date.strftime('%d %B %Y')), f"${predicted_prices[-1][0]:,.2f}")
