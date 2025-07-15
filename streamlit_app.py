import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="💰", layout="wide")
st.title("Prediksi Harga Bitcoin Menggunakan Algoritma Hybrid LSTM-GRU")

# Fungsi data historis dengan CryptoCompare
@st.cache_data
def get_historical_cc():
    now = int(pd.Timestamp(date.today()).timestamp())
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=1500"
    resp = requests.get(url).json()
    df = pd.DataFrame(resp['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s') 
    return df.set_index('time')['close']

# Model & scaler dummy
@st.cache_resource
def load_model_scaler():
    model = load_model('bitcoin_model.h5')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# Ambil data
try:
    close = get_historical_cc()
    st.subheader("📈 Grafik Harga Bitcoin")
    fig = go.Figure([go.Scatter(x=close.index, y=close.values, mode='lines')])
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Gagal memuat data historis: {e}")

# Input tanggal untuk prediksi
st.markdown("### Pilih tanggal prediksi")
future = st.date_input("Tanggal", min_value=date.today() + timedelta(days=1),
                       max_value=date.today() + timedelta(days=365))
if st.button("Klik disini untuk Memprediksi"):
    with st.spinner("🔄 Sedang memproses prediksi..."):
        model, scaler = load_model_scaler()
        seq_len = 60
        if len(close) < seq_len:
            st.error("Data historis tidak cukup.")
        else:
            arr = close.values[-seq_len:].reshape(-1,1)
            scaled = scaler.transform(arr)
            batch = scaled.reshape(1, seq_len, 1)
            days = (future - close.index[-1].date()).days
            if days <= 0:
                st.warning("Tanggal harus setelah data historis terakhir.")
            else:
                preds = []
                for _ in range(days):
                    p = model.predict(batch, verbose=0)[0]
                    preds.append(p)
                    batch = np.append(batch[:,1:,:], [[p]], axis=1)
                us = scaler.inverse_transform(preds).flatten()
                dt = [close.index[-1] + timedelta(days=i+1) for i in range(days)]
            
            # Tampilkan grafik
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=close.index, y=close.values, name='Historis'))
                fig2.add_trace(go.Scatter(x=dt, y=us, name='Prediksi', line=dict(color='red')))
                st.plotly_chart(fig2, use_container_width=True)
            
            # Tampilkan harga terakhir prediksi
                st.metric(f"Harga diprediksi pada {future}", f"${us[-1]:,.2f}")
