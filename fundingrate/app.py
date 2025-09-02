import time
import math
import random
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dateutil import tz
from datetime import datetime, timedelta, timezone

# =========================
# CONFIG
# =========================
BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_DATA = "https://fapi.binance.com/futures/data"  # open interest history
DEFAULT_SYMBOL = "BTCUSDT"   # Binance USDT-M perpetual symbol
DEFAULT_INTERVAL = "15m"     # 1m, 5m, 15m, 1h, 4h, 1d
DEFAULT_LIMIT = 500          # klines limit
LOCAL_TZ = tz.gettz("Asia/Kolkata")

# =========================
# SIMPLE RETRY (no tenacity)
# =========================
def get_json(url, params=None, timeout=15, attempts=4, base_sleep=0.7, max_sleep=3.0):
    """
    Simple exponential backoff with jitter.
    - attempts: total tries (>=1)
    - base_sleep: starting backoff
    - max_sleep: cap per sleep
    """
    last_err = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if i == attempts - 1:
                break
            # exponential backoff + jitter
            wait_s = min(max_sleep, base_sleep * (2 ** i)) + random.uniform(0, 0.3)
            time.sleep(wait_s)
    raise last_err

def ms_to_local_dt(ms):
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone(LOCAL_TZ)

# =========================
# DATA FETCHERS
# =========================
def fetch_klines(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """GET /fapi/v1/klines"""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    data = get_json(url, params={"symbol": symbol, "interval": interval, "limit": limit})
    cols = [
        "open_time","open","high","low","close","volume","close_time","quote_asset_volume",
        "number_of_trades","taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_open_interest_hist(symbol=DEFAULT_SYMBOL, period="15m", limit=500):
    """
    GET /futures/data/openInterestHist
    period: "5m","15m","30m","1h","2h","4h","6h","12h","1d"
    """
    url = f"{BINANCE_DATA}/openInterestHist"
    data = get_json(url, params={"symbol": symbol, "period": period, "limit": limit})
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    return df[["timestamp","sumOpenInterest"]].rename(columns={"sumOpenInterest":"open_interest"})

def fetch_funding_rates(symbol=DEFAULT_SYMBOL, limit=200):
    """GET /fapi/v1/fundingRate (historical funding per 8h)"""
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    data = get_json(url, params={"symbol": symbol, "limit": limit})
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["fundingTime","fundingRate"]].sort_values("fundingTime")

# =========================
# METRICS / SIGNALS
# =========================
def compute_oi_change_pct(oi_df, hours=24):
    if oi_df.empty:
        return np.nan
    end_time = oi_df["timestamp"].iloc[-1]
    start_cut = end_time - timedelta(hours=hours)
    recent = oi_df[oi_df["timestamp"] >= start_cut]
    if len(recent) < 2:
        return np.nan
    start = recent["open_interest"].iloc[0]
    end = recent["open_interest"].iloc[-1]
    if start == 0 or pd.isna(start):
        return np.nan
    return (end - start) / start * 100.0

def funding_extreme_signals(f_df, pos_thr=0.10/100, neg_thr=-0.05/100):
    """
    Heuristics:
      +0.10% (0.0010) every 8h => longs crowded
      -0.05% (-0.0005) every 8h => shorts crowded
    """
    if f_df.empty:
        return None
    last_rate = f_df.iloc[-1]["fundingRate"]
    status = "normal"
    note = "Funding within normal range."
    if pd.notna(last_rate):
        if last_rate >= pos_thr:
            status = "high_positive"
            note = "High positive funding → longs crowded; contrarian risk down."
        elif last_rate <= neg_thr:
            status = "high_negative"
            note = "High negative funding → shorts crowded; squeeze-up risk."
    return {"last_rate": last_rate, "status": status, "note": note}

def simple_market_structure(kl_df, lookback=80):
    """
    Light HH/HL detection on closes; fallback to slope.
    """
    if kl_df.empty:
        return {"trend": "unknown", "comment": "No data"}
    closes = kl_df["close"].tail(lookback).to_numpy()
    if len(closes) < 10:
        return {"trend":"unknown","comment":"Insufficient bars"}

    s = pd.Series(closes)
    highs = s[(s.shift(1) < s) & (s.shift(-1) < s)].dropna().tail(3)
    lows  = s[(s.shift(1) > s) & (s.shift(-1) > s)].dropna().tail(3)

    def inc(seq): return all(x < y for x, y in zip(seq, seq[1:]))
    def dec(seq): return all(x > y for x, y in zip(seq, seq[1:]))

    trend, comment = "range", "Range/indecision."
    if len(highs) >= 3 and len(lows) >= 3:
        if inc(highs.values) and inc(lows.values):
            trend, comment = "uptrend", "Higher Highs + Higher Lows."
        elif dec(highs.values) and dec(lows.values):
            trend, comment = "downtrend", "Lower Highs + Lower Lows."
    else:
        x = np.arange(len(closes))
        m = np.polyfit(x, closes, 1)[0]
        if m > 0:
            trend, comment = "uptrend", "Positive slope on closes."
        elif m < 0:
            trend, comment = "downtrend", "Negative slope on closes."
    return {"trend": trend, "comment": comment}

# =========================
# CHART
# =========================
def make_price_chart(kl, f_df=None, oi_df=None):
    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=kl["open_time"],
        open=kl["open"], high=kl["high"], low=kl["low"], close=kl["close"],
        name="Price"
    ))

    # Funding (secondary conceptually; displayed as %)
    if f_df is not None and not f_df.empty:
        fig.add_trace(go.Scatter(
            x=f_df["fundingTime"], y=f_df["fundingRate"]*100,
            mode="lines+markers", name="Funding (8h, %)"
        ))

    # Open Interest
    if oi_df is not None and not oi_df.empty:
        fig.add_trace(go.Scatter(
            x=oi_df["timestamp"], y=oi_df["open_interest"],
            mode="lines", name="Open Interest", fill="tozeroy"
        ))

    fig.update_layout(
        title=f"{DEFAULT_SYMBOL} — Price, Funding (%, 8h), Open Interest",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=720
    )
    return fig

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="BTC Perp Dashboard", layout="wide")

st.title("BTC Perpetual — Funding Rate + Open Interest + Price Action")
st.caption("Source: Binance Futures (public endpoints). Timezone: Asia/Kolkata.")

colA, colB, colC, colD = st.columns([1,1,1,1.2], vertical_alignment="center")
with colA:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL, help="Binance USDT-M perpetual symbol (e.g., BTCUSDT, ETHUSDT)")
with colB:
    interval = st.selectbox("Kline Interval", ["1m","5m","15m","1h","4h","1d"], index=2)
with colC:
    auto_refresh = st.checkbox("Auto-refresh", value=True)
with colD:
    refresh_sec = st.slider("Refresh every (seconds)", 5, 120, 20)

# Fetch data
with st.spinner("Fetching data …"):
    kl = fetch_klines(symbol=symbol, interval=interval, limit=DEFAULT_LIMIT)
    f_df = fetch_funding_rates(symbol=symbol, limit=200)
    oi_hist = fetch_open_interest_hist(symbol=symbol, period="15m", limit=500)

# Compute metrics
oi_pct_24h = compute_oi_change_pct(oi_hist, hours=24)
funding_sig = funding_extreme_signals(f_df)
mstruct = simple_market_structure(kl, lookback=80)

# KPI row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Last Price", f'{kl["close"].iloc[-1]:,.2f} USDT')
with k2:
    if funding_sig and funding_sig["last_rate"] is not None and pd.notna(funding_sig["last_rate"]):
        st.metric("Last Funding (8h)", f'{funding_sig["last_rate"]*100:.4f}%')
    else:
        st.metric("Last Funding (8h)", "—")
with k3:
    if oi_pct_24h is not None and not math.isnan(oi_pct_24h):
        st.metric("Open Interest (24h)", f"{oi_pct_24h:+.2f}%")
    else:
        st.metric("Open Interest (24h)", "—")
with k4:
    st.metric("Market Structure", mstruct["trend"].upper())

# Signals / Commentary
st.subheader("Signals & Commentary")
sig_col1, sig_col2 = st.columns(2)

with sig_col1:
    st.markdown("### Funding Context")
    if funding_sig:
        st.write(f"- **Status**: `{funding_sig['status']}` — {funding_sig['note']}")
        st.write("- Heuristic thresholds: +0.10% (longs crowded), -0.05% (shorts crowded).")
    else:
        st.write("No funding data.")

    st.markdown("### Market Structure")
    st.write(f"- {mstruct['comment']}")

with sig_col2:
    st.markdown("### OI + Funding Playbook (Heuristics)")
    st.write("- **Funding↑ + OI↑ + Price Flat** → Crowded side building. Watch for squeeze fade.")
    st.write("- **Funding↓ + OI↑ + Price Flat** → Shorts adding. Squeeze-up risk on breakout.")
    st.write("- **Funding extreme** → Use as contrarian context near key S/R, not standalone signals.")
    st.caption("Contextual reads only. Not financial advice.")

# Chart
fig = make_price_chart(kl, f_df, oi_hist)
st.plotly_chart(fig, use_container_width=True)

# Raw data
with st.expander("Raw: Funding History"):
    st.dataframe(f_df.tail(80), use_container_width=True)
with st.expander("Raw: Open Interest History"):
    st.dataframe(oi_hist.tail(200), use_container_width=True)
with st.expander("Raw: Klines"):
    st.dataframe(kl.tail(200), use_container_width=True)

# Auto-refresh
if auto_refresh:
    st.caption(f"Auto-refreshing every {refresh_sec}s …")
    time.sleep(refresh_sec)
    st.rerun()
