import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import requests
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Initialize prediction history in session state
#if 'predicted_history' not in st.session_state:
    #st.session_state['predicted_history'] = []


# --- Model Definition ---
class MultiTaskTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.output_layer(x[:, -1, :])

# --- Feature Preprocessing ---
def preprocess(df):
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Return'].rolling(window=10).std()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['BB_upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    df.dropna(inplace=True)

    features = [
        'Close', 'Volume', 'MA_20', 'MA_50', 'EMA_20',
        'BB_upper', 'BB_lower', 'RSI', 'Volatility',
        'Close_lag_1', 'Close_lag_2'
    ]
    return df[features].tail(60)

# --- Hardcoded MinMaxScaler Ranges ---
feature_mins = np.array([74610.0, 0.11258, 74859.2635, 74999.2088, 74907.9148, 75141.8374, 74493.6396, 0.0, 6.73e-08, 74610.0, 74610.0])
feature_maxs = np.array([88736.39, 1225.06388, 88471.6385, 88423.4862, 88446.4295, 88912.8396, 88388.0164, 99.9999801, 0.00728114838, 88736.39, 88736.39])

def scale_features(X):
    return (X - feature_mins) / (feature_maxs - feature_mins + 1e-8)

# --- Binance Live Fetcher ---
def fetch_binance_ohlcv(symbol='BTCUSDT', interval='1m', limit=120):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Vol", "Taker Buy Quote Vol", "Ignore"])
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df[["Open Time", "Close", "Volume"]]

# --- Load Model ---
@st.cache_resource
def load_model():
    model = MultiTaskTransformer(input_size=11)
    model.load_state_dict(torch.load("models/model_minutely_full_new.pth", map_location="cpu"))
    model.eval()
    return model

# --- Streamlit UI ---
st.set_page_config("Minutely BTC Prediction", layout="wide")
if 'predicted_history' not in st.session_state:
    st.session_state['predicted_history'] = []
st.title("Live Bitcoin Minutely Prediction")

model = load_model()
raw_df = fetch_binance_ohlcv()
X_raw = preprocess(raw_df)
X_scaled = scale_features(X_raw.values)

st.subheader("Recent BTC/USDT Minutely Data")
st.dataframe(X_raw.tail(), use_container_width=True)

st.markdown("### Model Prediction")
st.code(f"Fetched rows: {len(raw_df)}")
st.code(f"Preprocessed rows: {X_raw.shape[0]}")

if X_raw.shape[0] < 60:
    st.warning("Not enough data to make a prediction.")
else:
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor).squeeze().numpy()
        #close, vol, direction = prediction[0], prediction[1], int(prediction[2] > 0.5)
        close = prediction[0]
        vol = abs(prediction[1])  # Ensure non-negative volatility
        direction = int(prediction[2] > 0.5)
        next_time = raw_df["Open Time"].iloc[-1] + pd.Timedelta(minutes=1)
        st.session_state['predicted_history'].append((next_time, close))

    # Display prediction as big metrics
    # Centered and styled material-style prediction display
st.markdown("""
<style>
.metric-container {
    text-align: center;
    font-size: 22px;
    font-weight: 600;
}
.metric-value {
    font-size: 36px;
    font-weight: bold;
    margin-top: 4px;
}
.up {
    color: #2ECC71;  /* Green */
}
.down {
    color: #E74C3C;  /* Red */
}
</style>
""", unsafe_allow_html=True)

direction_icon = "⬆️" if direction else "⬇️"
direction_class = "up" if direction else "down"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-container">
        Predicted Close Price
        <div class="metric-value">${close:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container">
        Predicted Volatility
        <div class="metric-value">{vol:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        Predicted Direction
        <div class="metric-value {direction_class}">{direction_icon} {'Up' if direction else 'Down'}</div>
    </div>
    """, unsafe_allow_html=True)


# --- Live Updating Plot ---
st.markdown("### Actual vs Predicted Close Price (Auto-refreshes every 60s)")

timestamps = raw_df["Open Time"].tail(60).tolist()
actual_close = raw_df["Close"].tail(60).tolist()
predicted_closes = [close] * 60  # You can enhance this to keep running history later
next_time = timestamps[-1] + pd.Timedelta(minutes=1)

fig = go.Figure()

# Actual close line
fig.add_trace(go.Scatter(
    x=timestamps,
    y=actual_close,
    mode="lines+markers",
    name="Actual Close",
    line=dict(color="blue")
))

# Predicted close line (flat for now)
#fig.add_trace(go.Scatter(
#    x=timestamps,
#    y=predicted_closes,
#    mode="lines",
#    name="Predicted Close",
#    line=dict(color="orange", dash="dash")
#))

# Predicted next point
fig.add_trace(go.Scatter(
    x=[next_time],
    y=[close],
    mode="markers+text",
    marker=dict(color="red", size=10),
    #text=["Predicted"],
    textposition="top center",
    name="Next Prediction"
))

# Confidence band
#fig.add_trace(go.Scatter(
#    x=[next_time, next_time, next_time, next_time],
#    y=[close - vol * 500, close + vol * 500, close + vol * 500, close - vol * 500],
#    fill='toself',
#    fillcolor='rgba(255, 0, 0, 0.15)',
#    line=dict(color='rgba(255,0,0,0.2)'),
#    name='Confidence Band',
#    showlegend=True
#))

# Red line for all past predicted values
if len(st.session_state['predicted_history']) > 1:
    pred_times, pred_closes = zip(*st.session_state['predicted_history'])
    fig.add_trace(go.Scatter(
        x=list(pred_times),
        y=list(pred_closes),
        mode="lines+markers",
        line=dict(color="red", width=2, dash="dash"),
        name="Predicted Line"
))

fig.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=30, b=20),
    showlegend=True,
    xaxis_title="Time",
    yaxis_title="BTC/USDT Price",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="refresh")