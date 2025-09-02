# totals_service.py
from __future__ import annotations
from typing import Dict, Optional

_SYMBOL_FALLBACK = "$"

class TotalsService:
    """
    Computes totals (USD base via fetch_totals) and converts to target currencies.
    Currencies are configured via enabled_currencies (order matters).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        verbose: bool = False,
        enabled_currencies: Optional[Dict[str, Dict]] = None,
    ):
        """
        enabled_currencies example:
        {
          "USD": {"symbol": "$", "factor": 1.0,  "editable": False, "label": "USD"},
          "INR": {"symbol": "₹", "factor": 83.0, "editable": True,  "label": "INR per USD"},
          "EUR": {"symbol": "€", "factor": 0.92, "editable": True,  "label": "EUR per USD"},
        }
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.verbose = verbose

        self.currencies: Dict[str, Dict] = enabled_currencies or {
            "USD": {"symbol": "$", "factor": 1.0, "editable": False, "label": "USD"},
            "INR": {"symbol": "₹", "factor": 83.0, "editable": True, "label": "INR per USD"},
        }

    # --- compute (USD base) ---
    def compute(self, start_date: str, end_date: str, asset_symbol: Optional[str]) -> dict:
        from delta_txn_totals import fetch_totals
        return fetch_totals(
            api_key=self.api_key,
            api_secret=self.api_secret,
            start_date=start_date,
            end_date=end_date,
            asset_symbol=(asset_symbol or None),
            verbose=self.verbose,
        )

    # --- conversions ---
    def _factor(self, code: str) -> float:
        return float(self.currencies.get(code.upper(), {}).get("factor", 1.0))

    def _symbol(self, code: str) -> str:
        return str(self.currencies.get(code.upper(), {}).get("symbol", _SYMBOL_FALLBACK))

    def convert_amount(self, amount_usd: Optional[float], target_ccy: str) -> Optional[float]:
        if amount_usd is None:
            return None
        return float(amount_usd) * self._factor(target_ccy)

    def convert_totals(self, totals_usd: dict, target_ccy: str) -> dict:
        tgt = target_ccy.upper()
        f = self._factor(tgt)
        out = dict(totals_usd)
        for k in ("total_commission", "total_funding", "total_cashflow", "total_deposit"):
            v = totals_usd.get(k)
            if v is not None:
                out[k] = float(v) * f
        out["ccy"] = tgt
        out["symbol"] = self._symbol(tgt)
        return out

    def set_rates(self, **usd_to_target: float) -> None:
        """
        Update factors in-place. Pass kwargs like: set_rates(INR=83.2, EUR=0.91)
        (factor = units of target per 1 USD)
        """
        for code, factor in usd_to_target.items():
            code_u = code.upper()
            if code_u in self.currencies:
                self.currencies[code_u]["factor"] = float(factor)

    def enabled_codes(self) -> list[str]:
        """Ordered list of enabled currency codes (as configured)."""
        return list(self.currencies.keys())

    def editable_codes(self) -> list[str]:
        return [c for c, cfg in self.currencies.items() if bool(cfg.get("editable", False))]

    def label_for(self, code: str) -> str:
        return str(self.currencies.get(code, {}).get("label", code))



# # totals_service.py (or inside your app.py if you defined TotalsService there)
# from __future__ import annotations
# from typing import Optional, Dict

# # Keep your existing imports above…

# _SYMBOL = {"USD": "$", "INR": "₹", "EUR": "€"}

# class TotalsService:
#     """
#     Thin wrapper around fetch_totals() with built-in currency conversion.
#     All computations are done in USD first; conversions are derived from a USD base.
#     """
#     def __init__(self, api_key: str, api_secret: str, verbose: bool = False):
#         self.api_key = api_key
#         self.api_secret = api_secret
#         self.verbose = verbose

#         # factors = <target units per 1 USD>
#         # i.e., 1 USD = 83 INR, 1 USD = 0.92 EUR
#         self.factors: Dict[str, float] = {
#             "USD": 1.0,
#             "INR": 85.0,   # edit as you like
#             "EUR": 0.92,   # edit as you like
#         }

#     # --- existing method you already had ---
#     def compute(self, start_date: str, end_date: str, asset_symbol: Optional[str]) -> dict:
#         # import here to avoid circulars if needed
#         from delta_txn_totals import fetch_totals
#         return fetch_totals(
#             api_key=self.api_key,
#             api_secret=self.api_secret,
#             start_date=start_date,
#             end_date=end_date,
#             asset_symbol=(asset_symbol or None),
#             verbose=self.verbose,
#         )

#     # Optional helper if you also compute end-of-window balance in USD elsewhere
#     def convert_amount(self, amount_usd: Optional[float], target_ccy: str) -> Optional[float]:
#         if amount_usd is None:
#             return None
#         f = self.factors.get(target_ccy.upper())
#         if not f:
#             f = 1.0  # fallback
#         return float(amount_usd) * f

#     def convert_totals(self, totals_usd: dict, target_ccy: str) -> dict:
#         """
#         Convert a totals dict that is in USD to target currency in a copy—does not mutate input.
#         Expects keys: total_commission, total_funding, total_cashflow, total_deposit
#         """
#         tgt = target_ccy.upper()
#         f = self.factors.get(tgt, 1.0)
#         out = dict(totals_usd)  # shallow copy
#         for k in ("total_commission", "total_funding", "total_cashflow", "total_deposit"):
#             v = totals_usd.get(k)
#             if v is not None:
#                 out[k] = float(v) * f
#         out["asset"] = totals_usd.get("asset", "ANY")
#         out["window_ist"] = totals_usd.get("window_ist")
#         out["ccy"] = tgt
#         out["symbol"] = _SYMBOL.get(tgt, "$")
#         return out

#     def set_rates(self, usd_inr: Optional[float] = None, usd_eur: Optional[float] = None):
#         """Update conversion factors (units of target per 1 USD)."""
#         if usd_inr is not None:
#             self.factors["INR"] = float(usd_inr)
#         if usd_eur is not None:
#             self.factors["EUR"] = float(usd_eur)
