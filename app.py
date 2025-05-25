import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Initialize the BingX exchange
exchange = ccxt.bingx()
exchange.load_markets()

# Function to fetch and prepare data
def get_ohlcv_data(symbol, timeframe, limit, since=None):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')  # Convert to IST
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Streamlit app
st.title("Interactive Trading Charts with BingX")
st.sidebar.header("Settings")

# Symbol selection
symbols = exchange.symbols
symbol = st.sidebar.selectbox("Select Symbol", symbols, index=symbols.index('BTC/USDT'))

# Timeframe selection
timeframes = list(exchange.timeframes.keys())
timeframe = st.sidebar.selectbox("Select Timeframe", timeframes, index=timeframes.index('1h'))

# Number of candles to fetch
limit = st.sidebar.slider("Number of Candles", min_value=50, max_value=500, value=150, step=50)

# Scroll functionality (start time)
scroll_days = st.sidebar.slider("Scroll Back (Days)", min_value=0, max_value=30, value=0, step=1)
since = None
if scroll_days > 0:
    since = int((datetime.utcnow() - timedelta(days=scroll_days)).timestamp() * 1000)

# Fetch data
st.write(f"Fetching {limit} candles for {symbol} at {timeframe} timeframe...")
df = get_ohlcv_data(symbol, timeframe, limit, since=since)

# Plot data
if not df.empty:
    st.write(f"Interactive Chart for {symbol} ({timeframe})")

    # Create candlestick chart using Plotly
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )
    ])

    # Customize layout
    fig.update_layout(
        title=f"{symbol} ({timeframe})",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,  # Add a range slider for scrolling
        template="plotly_dark",          # Dark theme (optional)
        height=600                       # Set figure height
    )

    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    st.write("Data Preview:")
    st.dataframe(df)
else:
    st.warning("No data available for the selected parameters.")