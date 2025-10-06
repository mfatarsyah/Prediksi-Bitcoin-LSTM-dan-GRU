# Bitcoin Price Prediction Using Hybrid LSTM-GRU Model

## This project presents a deep learning-based approach to predict Bitcoin prices using a hybrid model that combines Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures. The model is deployed via Streamlit, offering an interactive dashboard for real-time visualization and forecasting.

### Features
🔁 Hybrid LSTM-GRU Model: Combines the strengths of both architectures for improved time series prediction.

📊 Streamlit Dashboard: Interactive web interface for visualizing historical data, predictions, and model performance.

🧠 Preprocessing Pipeline: Includes MinMaxScaler for normalization and joblib for model persistence.

📁 Modular Codebase: Clean separation of concerns between data handling, model logic, and UI components.


### 🧪 Requirements
``` pip install -r requirements.txt ```


### ▶️ How to Run
``` streamlit run streamlit_app.py ```

### 📌 Model Overview
The hybrid model is trained on historical Bitcoin price data to capture both short-term volatility and long-term trends. It uses a sequential architecture with stacked LSTM and GRU layers followed by dense output for regression.
