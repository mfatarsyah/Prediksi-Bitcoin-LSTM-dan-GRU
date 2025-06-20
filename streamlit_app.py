import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import time # Diimpor untuk mendemonstrasikan progress bar

# =====================================================================================
# Konfigurasi Halaman Streamlit
# =====================================================================================
st.set_page_config(
    page_title="Prediksi Harga Bitcoin | Hybrid LSTM-GRU (Optimized)",
    page_icon="⚡",
    layout="wide"
)

# =====================================================================================
# Inisialisasi Session State untuk Caching Hasil
# Ini adalah kunci untuk performa tinggi. Hasil prediksi akan disimpan di sini.
# =====================================================================================
if 'prediction_state' not in st.session_state:
    st.session_state.prediction_state = {
        'results': None,
        'last_params': None
    }

# =====================================================================================
# Fungsi-fungsi Bantuan (dengan Caching untuk Performa)
# =====================================================================================
@st.cache_resource
def load_prediction_model(model_path='bitcoin_model.h5'):
    """Memuat model hybrid LSTM-GRU yang telah dilatih."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error memuat model dari '{model_path}': {e}", icon="🚨")
        st.info("Pastikan file model 'btc_model.h5' ada di direktori yang sama.")
        return None

@st.cache_resource
def load_price_scaler(scaler_path='scaler.joblib'):
    """Memuat scaler yang digunakan saat training."""
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Error memuat scaler dari '{scaler_path}': {e}", icon="🚨")
        st.info("Pastikan file 'scaler.joblib' ada di direktori yang sama.")
        return None

@st.cache_data
def load_bitcoin_data(start_date, end_date):
    """Mengunduh data harga Bitcoin (BTC-USD) dari Yahoo Finance."""
    try:
        data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
        if data.empty:
            st.warning("Tidak ada data yang ditemukan untuk rentang tanggal yang dipilih.", icon="⚠️")
            return None
        return data[['Close']]
    except Exception as e:
        st.error(f"Gagal mengunduh data dari Yahoo Finance: {e}", icon="🌐")
        return None

# =====================================================================================
# Tampilan Utama Aplikasi
# =====================================================================================
st.title("⚡ Dasbor Prediksi Harga Bitcoin (Versi Cepat)")
st.markdown("Aplikasi ini telah dioptimalkan untuk performa. Prediksi yang berat hanya dijalankan sekali dan hasilnya disimpan sementara.")
st.markdown("---")

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("⚙️ Panel Kontrol")

default_start = date.today() - timedelta(days=365*2)
default_end = date.today()

start_date = st.sidebar.date_input("🗓️ Tanggal Mulai Data", default_start, min_value=date(2014, 9, 17), max_value=default_end)
end_date = st.sidebar.date_input("🗓️ Tanggal Akhir Data", default_end, min_value=start_date, max_value=date.today())

n_days_to_predict = st.sidebar.slider("Forecasting (hari)", 1, 30, 7, help="Pilih berapa hari ke depan yang ingin Anda prediksi (Maks: 30 hari).")

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Judul Skripsi:**\n
    Implementasi Deep Learning Menggunakan Model Hybrid LSTM dan GRU untuk Memprediksi Harga Bitcoin Berbasis Streamlit.
    
    **Perbaikan:** Kode ini dioptimalkan untuk mengatasi loading lama.
    """
)

# =====================================================================================
# Logika Inti dan Visualisasi
# =====================================================================================
data = load_bitcoin_data(start_date, end_date)

if data is not None:
    # Buat placeholder untuk elemen UI agar bisa diupdate nanti
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    table_placeholder = st.empty()

    # Gabungkan parameter saat ini untuk perbandingan cache
    current_params = (start_date, end_date, n_days_to_predict)

    # --- Logika Prediksi ---
    if predict_button:
        model = load_prediction_model()
        scaler = load_price_scaler()

        if model is not None and scaler is not None:
            sequence_length = 60 # PENTING: Harus sama dengan yang digunakan saat training

            if len(data) < sequence_length:
                st.error(f"Error: Data historis tidak cukup. Dibutuhkan minimal {sequence_length} hari data. Pilih rentang tanggal mulai yang lebih awal.", icon="📉")
            else:
                # Tampilkan pesan proses dan progress bar
                progress_text = "Proses prediksi sedang berjalan. Mohon tunggu."
                my_bar = st.progress(0, text=progress_text)
                
                # Persiapan data untuk prediksi
                last_sequence = data['Close'].values[-sequence_length:]
                last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
                
                predictions = []
                current_batch = last_sequence_scaled.reshape(1, sequence_length, 1)

                # Loop prediksi yang memakan waktu
                for i in range(n_days_to_predict):
                    pred = model.predict(current_batch, verbose=0)[0]
                    predictions.append(pred)
                    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
                    # Update progress bar
                    my_bar.progress((i + 1) / n_days_to_predict, text=f"{progress_text} (Hari ke-{i+1}/{n_days_to_predict})")

                time.sleep(1) # Beri jeda agar animasi progress bar terlihat selesai
                my_bar.empty() # Hapus progress bar setelah selesai

                predicted_prices = scaler.inverse_transform(predictions)
                last_date = data.index[-1]
                prediction_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, n_days_to_predict + 1)])
                
                # Simpan hasil ke session state (CACHE)
                st.session_state.prediction_state['results'] = {
                    'predicted_prices': predicted_prices,
                    'prediction_dates': prediction_dates
                }
                st.session_state.prediction_state['last_params'] = current_params
                st.success('Prediksi berhasil diselesaikan!', icon="✅")

    # --- Logika Tampilan Hasil ---
    # Cek apakah ada hasil (dari perhitungan baru atau dari cache)
    if st.session_state.prediction_state['results'] and st.session_state.prediction_state['last_params'] == current_params:
        
        # Ambil hasil dari cache
        results = st.session_state.prediction_state['results']
        predicted_prices = results['predicted_prices']
        prediction_dates = results['prediction_dates']

        # Tampilkan metrik
        with metrics_placeholder.container():
            col1, col2 = st.columns(2)
            last_price = data['Close'].iloc[-1]
            predicted_tomorrow_price = predicted_prices[0][0]
            delta = predicted_tomorrow_price - last_price
            col1.metric("Harga Penutupan Terakhir", f"${last_price:,.2f}")
            col2.metric("Prediksi Harga Besok", f"${predicted_tomorrow_price:,.2f}", f"${delta:,.2f}", delta_color="inverse")
        
        # Tampilkan grafik
        with chart_placeholder.container():
            st.subheader("Visualisasi Harga: Historis vs. Prediksi")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Harga Historis', line=dict(color='deepskyblue', width=2)))
            fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices.flatten(), mode='lines', name='Harga Prediksi', line=dict(color='tomato', width=2, dash='dash')))
            fig.update_layout(xaxis_title='Tanggal', yaxis_title='Harga (USD)', yaxis_tickprefix='$', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Tampilkan tabel rincian
        with table_placeholder.container():
            st.markdown("#### Rincian Hasil Prediksi")
            df_predictions = pd.DataFrame({'Tanggal': prediction_dates.strftime('%A, %d %B %Y'),'Harga Prediksi (USD)': predicted_prices.flatten()})
            st.dataframe(df_predictions.set_index('Tanggal').style.format({'Harga Prediksi (USD)': "${:,.2f}"}), use_container_width=True)

    else:
        # Tampilan default jika belum ada prediksi atau parameter berubah
        with chart_placeholder.container():
            st.subheader("Visualisasi Harga Historis")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Harga Historis', line=dict(color='deepskyblue', width=2)))
            fig.update_layout(xaxis_title='Tanggal', yaxis_title='Harga (USD)', yaxis_tickprefix='$', template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)

    # Selalu tampilkan data mentah di expander
    with st.expander("Lihat Data Historis Mentah (Tabel)"):
        st.dataframe(data.sort_index(ascending=False).style.format({'Close': "${:,.2f}"}), use_container_width=True)

