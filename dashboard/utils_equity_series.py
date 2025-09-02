# utils_equity_series.py
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Optional, Tuple, Dict

import pandas as pd

from delta_client import DeltaIndiaClient, DeltaAPIError

IST = ZoneInfo("Asia/Kolkata")

# transaction types that impact wallet equity (adjust if your ledger shows more)
P_EQUITY_TYPES = {
    # cash flows
    "deposit", "withdrawal", "transfer_in", "transfer_out", "bonus", "adjustment",
    # trading PnL & costs
    "settlement", "funding", "commission",
}

def _to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)

def _parse_dt_ist(s: str) -> datetime:
    # created_at is ISO; normalize to IST for daily bucketing
    return pd.to_datetime(s, utc=True).to_pydatetime().astimezone(IST)

def fetch_daily_equity_series(
    api_key: str,
    api_secret: str,
    start_date: str,  # 'YYYY-MM-DD' in IST
    end_date: str,    # 'YYYY-MM-DD' in IST
    base_ccy: str = "USDT",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['date', 'equity'] where 'date' is IST date
    and 'equity' is END-OF-DAY equity (wallet) in base_ccy, reconstructed from the ledger.
    """
    # Parse the requested window in IST
    t_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=IST)
    t_end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=IST) + timedelta(hours=23, minutes=59, seconds=59)
    if t_end < t_start:
        raise ValueError("end_date must be >= start_date")

    # Build a day index in IST
    all_days = pd.date_range(t_start.date(), t_end.date(), freq="D", tz=IST)

    client = DeltaIndiaClient(api_key, api_secret, verbose=verbose)

    # 1) Anchor: current equity NOW (in base_ccy), from balances
    balances = client.get_wallet_balances()
    equity_now = 0.0
    for b in balances:
        asset = str(b.get("asset", "")).upper()
        if asset == base_ccy and "equity" in b:
            equity_now = float(b["equity"])
            break
    if equity_now == 0.0:
        # fallback: sum per-asset equity if provided
        eq_sum, ok = 0.0, False
        for b in balances:
            if "equity" in b:
                ok = True
                eq_sum += float(b["equity"])
        if ok:
            equity_now = eq_sum

    # 2) Get wallet transactions from start_date 00:00 up to NOW (so we can back out equity at start)
    #    If you only need up to end_date (in the past), you can still use NOW and it will work;
    #    we’ll subtract all deltas after the day we care about.
    now_utc = datetime.now(IST).astimezone(ZoneInfo("UTC"))
    start_utc = t_start.astimezone(ZoneInfo("UTC"))
    txns = client.get_wallet_transactions(start_us=_to_us(start_utc), end_us=_to_us(now_utc), max_pages=200)

    tx = pd.DataFrame(txns)
    if tx.empty:
        # No ledger entries -> equity is flat over the window
        return pd.DataFrame({
            "date": all_days.date,
            "equity": [equity_now] * len(all_days),
        })

    # 3) Normalize and filter to equity-impacting entries
    tx["created_at"] = tx["created_at"].apply(_parse_dt_ist)
    tx["transaction_type"] = tx["transaction_type"].astype(str)
    tx["amount"] = tx["amount"].astype(float)

    tx = tx[tx["transaction_type"].str.lower().isin(P_EQUITY_TYPES)].copy()

    # 4) Bucket per IST day (end-of-day equity uses daily sum)
    tx["day"] = tx["created_at"].dt.date

    # daily deltas across full pulled window (start_date .. now)
    daily_delta = tx.groupby("day")["amount"].sum().sort_index()

    # 5) Compute equity at the START of window by backing out deltas *after* t_start - 1 day
    #    Let S be start_date's end-of-day equity we want to derive via forward cumulative sum.
    #    If we know current equity (E_now), then:
    #      E_now = S + sum(daily_deltas from start_date through today)
    #      => S = E_now - sum(daily_deltas from start_date through today)
    tail_sum = daily_delta.loc[
        daily_delta.index >= t_start.date()
    ].sum() if not daily_delta.empty else 0.0

    equity_start = float(equity_now - tail_sum)

    # 6) Build daily series within requested [t_start, t_end]:
    #    EOD equity for day D = equity_start + cumulative sum of deltas up to D (inclusive)
    # Ensure we have deltas for every date (fill missing with 0)
    window_days = pd.date_range(t_start.date(), t_end.date(), freq="D", tz=IST).date
    delta_window = pd.Series(0.0, index=pd.Index(window_days, name="day"))
    for d, v in daily_delta.items():
        if d in delta_window.index:
            delta_window.loc[d] = v

    equity_series = (delta_window.cumsum() + equity_start).astype(float)

    out = pd.DataFrame({"date": equity_series.index, "equity": equity_series.values})
    return out
