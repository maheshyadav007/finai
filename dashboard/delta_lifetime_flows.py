# delta_lifetime_flows.py
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from delta_client import DeltaIndiaClient, DeltaAPIError
from delta_txn_models import (
    parse_all_txns,
    DepositTxn,
    WithdrawalTxn,
    SubAccountTransferTxn,
)

def _to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)

def _log(verbose: bool, *args):
    if verbose:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[Lifetime", ts, "]", *args)


def compute_lifetime_flows(
    api_key: str,
    api_secret: str,
    asset_symbol: Optional[str] = None,   # e.g. "USD" or "USDT"
    verbose: bool = False,
    max_pages: Optional[int] = None,      # None = all pages
) -> Dict[str, Any]:
    """
    Summarize external cash movements since account inception:
      - total_deposit (sum of deposits, positive)
      - total_withdrawal (sum of withdrawals, positive magnitude)
      - total_sub_transfer_in  (sum of positive sub-account transfers)
      - total_sub_transfer_out (sum of negative sub-account transfers, as positive magnitude)
      - net_external = (deposit + transfer_in) - (withdrawal + transfer_out)

    If `asset_symbol` is provided, only that asset is counted.

    Returns a dict with totals (floats) and diagnostic info.
    """
    client = DeltaIndiaClient(api_key, api_secret, verbose=verbose)

    # Pull from epoch to now (UTC). Server will cap range internally; we page until done.
    start_us = _to_us(datetime(1970, 1, 1, tzinfo=timezone.utc))
    end_us   = _to_us(datetime.now(tz=timezone.utc))

    _log(verbose, f"Fetching wallet transactions from epoch..now (micros: {start_us}..{end_us})")

    try:
        rows = client.get_wallet_transactions(start_us=start_us, end_us=end_us, max_pages=max_pages)
    except DeltaAPIError as e:
        _log(verbose, "API error:", e, "| payload:", e.payload, "| signed:", e.signature_data)
        raise
    except Exception as e:
        _log(verbose, "Unexpected error:", e)
        raise

    _log(verbose, f"Fetched {len(rows)} transactions (raw)")

    # Parse into typed txns
    txns = parse_all_txns(rows)

    # Optional asset filter
    if asset_symbol:
        a = asset_symbol.upper()
        txns = [t for t in txns if (t.asset_symbol or "").upper() == a]
        _log(verbose, f"Kept {len(txns)} after asset filter = {a}")

    # Totals (Decimals to avoid float drift, return floats at the end)
    dep  = Decimal("0")
    wdr  = Decimal("0")
    tin  = Decimal("0")
    tout = Decimal("0")

    # We only care about these movement types for lifetime flows
    for t in txns:
        if isinstance(t, DepositTxn):
            # deposits typically positive in ledger
            dep += t.amount
        elif isinstance(t, WithdrawalTxn):
            # withdrawals may be negative; store positive magnitude
            wdr += (-t.amount if t.amount < 0 else t.amount)
        elif isinstance(t, SubAccountTransferTxn):
            # positive = incoming, negative = outgoing
            if t.amount >= 0:
                tin += t.amount
            else:
                tout += (-t.amount)

    net_external = (dep + tin) - (wdr + tout)

    out = {
        "asset": asset_symbol or "ANY",
        "total_deposit": float(dep),
        "total_withdrawal": float(wdr),
        "total_sub_transfer_in": float(tin),
        "total_sub_transfer_out": float(tout),
        "net_external": float(net_external),
        "count_considered": len(txns),
    }
    _log(verbose, "Lifetime flows:", out)
    return out


# ---- quick manual test ----
if __name__ == "__main__":
    from CONSTANTS import API_KEY, API_SECRET
    print(compute_lifetime_flows(API_KEY, API_SECRET, asset_symbol="USD", verbose=True))
