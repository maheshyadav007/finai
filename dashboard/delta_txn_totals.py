# delta_txn_totals.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from delta_client import DeltaIndiaClient, DeltaAPIError

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


# ---------- tiny helpers ----------
def _to_us(dt: datetime) -> int:
    """UTC datetime -> microseconds since epoch (Delta API uses this)."""
    return int(dt.timestamp() * 1_000_000)

def _ist_day_bounds(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    """Return IST-aware [start_of_day, end_of_day] datetimes for date strings (YYYY-MM-DD)."""
    t_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=IST)
    t_end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=IST) + timedelta(hours=23, minutes=59, seconds=59)
    if t_end < t_start:
        raise ValueError("end_date must be >= start_date")
    return t_start, t_end

def _to_decimal(x: Any) -> Decimal:
    if x is None or x == "":
        return Decimal("0")
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")

def _log(verbose: bool, *args):
    if verbose:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[Totals", ts, "]", *args)


# ---------- minimal model ----------
@dataclass
class DeltaTxn:
    amount: Decimal
    asset_id: Optional[int]
    asset_symbol: str
    balance: Decimal                # balance AFTER txn
    created_at: str                 # raw string (UTC ISO)
    transaction_type: str           # 'cashflow' | 'commission' | 'deposit' | 'funding' (as per your data)
    uuid: str
    user_id: Optional[int] = None
    product_id: Optional[int] = None
    fund_id: Optional[int] = None
    # convenient bits from meta_data (optional)
    product_symbol: Optional[str] = None
    fill_uuid: Optional[str] = None

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "DeltaTxn":
        meta = d.get("meta_data") or {}
        return cls(
            amount=_to_decimal(d.get("amount")),
            asset_id=(int(d["asset_id"]) if d.get("asset_id") is not None else None),
            asset_symbol=str(d.get("asset_symbol") or "").upper(),
            balance=_to_decimal(d.get("balance")),
            created_at=str(d.get("created_at") or ""),
            transaction_type=str(d.get("transaction_type") or "").lower(),
            uuid=str(d.get("uuid") or ""),
            user_id=(int(d["user_id"]) if d.get("user_id") is not None else None),
            product_id=(int(d["product_id"]) if d.get("product_id") is not None else None),
            fund_id=(int(d["fund_id"]) if d.get("fund_id") is not None else None),
            product_symbol=meta.get("product_symbol"),
            fill_uuid=meta.get("fill_uuid"),
        )


# ---------- main function you asked for ----------
def fetch_totals(
    api_key: str,
    api_secret: str,
    start_date: str,                 # 'YYYY-MM-DD' (IST)
    end_date: str,                   # 'YYYY-MM-DD' (IST)
    asset_symbol: Optional[str] = None,  # e.g. 'USD' or 'USDT'; if None, include all assets
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Fetch wallet transactions in [start_date, end_date] (IST) and compute totals for:
      - total_commission
      - total_funding
      - total_cashflow
      - total_deposit

    Returns floats (you can switch to Decimal easily if you prefer).
    """
    # 1) compute IST window -> UTC microseconds
    t_start_ist, t_end_ist = _ist_day_bounds(start_date, end_date)
    start_utc = t_start_ist.astimezone(UTC)
    end_utc = t_end_ist.astimezone(UTC)
    start_us = _to_us(start_utc)
    end_us = _to_us(end_utc)

    _log(verbose, f"Window IST: {t_start_ist} .. {t_end_ist}")
    _log(verbose, f"Window UTC: {start_utc} .. {end_utc}")
    _log(verbose, f"Micros: {start_us} .. {end_us}")

    # 2) pull txns
    client = DeltaIndiaClient(api_key, api_secret, verbose=verbose)
    try:
        rows = client.get_wallet_transactions(start_us=start_us, end_us=end_us, max_pages=200)
    except DeltaAPIError as e:
        _log(verbose, "API error:", e, "| payload:", e.payload, "| signed:", e.signature_data)
        raise
    except Exception as e:
        _log(verbose, "Unexpected error:", e)
        raise

    _log(verbose, f"Fetched {len(rows)} wallet txns")

    # 3) convert -> DeltaTxn (+ optional asset filter)
    txns: List[DeltaTxn] = []
    for r in rows:
        try:
            t = DeltaTxn.from_api(r)
            if asset_symbol:
                if t.asset_symbol.upper() != asset_symbol.upper():
                    continue
            txns.append(t)
        except Exception as e:
            _log(verbose, "Skip row:", e, r)

    _log(verbose, f"Kept {len(txns)} txns after filtering (asset={asset_symbol or 'ANY'})")

    # 4) single pass totals for your 4 types
    total_commission = Decimal("0")
    total_funding    = Decimal("0")
    total_cashflow   = Decimal("0")
    total_deposit    = Decimal("0")

    for t in txns:
        tt = t.transaction_type
        if tt == "commission":
            total_commission += t.amount
        elif tt == "funding":
            total_funding += t.amount
        elif tt == "cashflow":
            total_cashflow += t.amount
        elif tt == "deposit":      # deposit is its own type in your data
            total_deposit += t.amount
        else:
            # ignore anything else (or log in verbose mode)
            _log(verbose, "Ignore txn type:", tt)

    # 5) return as floats (still signed, as per API amounts)
    total_commission = -total_commission  # commissions are negative amounts
    result = {
        "total_commission": float(total_commission),
        "total_funding": float(total_funding),
        "total_cashflow": float(total_cashflow),
        "total_deposit": float(total_deposit),
        "count": len(txns),
        "asset": asset_symbol or "ANY",
        "window_ist": f"{t_start_ist.date()} .. {t_end_ist.date()}",
    }
    _log(verbose, "Totals:", result)
    return result


# ---------- quick manual test ----------
if __name__ == "__main__":
    # from CONSTANTS import API_KEY, API_SECRET
    API_KEY = "YOUR_KEY"
    API_SECRET = "YOUR_SECRET"
    out = fetch_totals(API_KEY, API_SECRET,
                       start_date="2025-08-01",
                       end_date="2025-08-29",
                       asset_symbol="USD",  # or None
                       verbose=True)
    print(out)
