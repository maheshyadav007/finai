# delta_client.py
import time
import hmac
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple, Sequence, Union
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

import requests

BASE_URL = "https://api.india.delta.exchange"
DEFAULT_TIMEOUT = 30

ParamsType = Optional[Union[Sequence[Tuple[str, Any]], Dict[str, Any]]]

@dataclass
class DeltaAPIError(Exception):
    status_code: int
    code: Optional[str]
    message: str
    payload: Optional[Dict[str, Any]] = None
    signature_data: Optional[str] = None

    def __str__(self) -> str:
        code_str = f" [{self.code}]" if self.code else ""
        return f"DeltaAPIError{code_str}: {self.message} (HTTP {self.status_code})"


class DeltaIndiaClient:
    """
    Robust client with:
    - signature that INCLUDES '?' before query (when present)
    - param ORDER preserved (no sorting); we pass a list of (key, value) tuples
    - matching signature string and actual request URL
    - verbose diagnostics
    """

    def __init__(self, api_key: str, api_secret: str, user_agent: str = "NiceGUI-DeltaDash/1.4", verbose: bool = False):
        self.api_key = api_key
        self.api_secret_bytes = api_secret.encode()
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": user_agent})

    # ---------- logging ----------
    def _log(self, msg: str) -> None:
        if self.verbose:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DeltaClient {ts}] {msg}")

    # ---------- helpers ----------
    @staticmethod
    def _now_epoch_seconds_str() -> str:
        return str(int(time.time()))

    @staticmethod
    def _qs_from(params: ParamsType) -> str:
        """Build query string WITHOUT leading '?' using the provided order."""
        if params is None:
            return ""
        if isinstance(params, dict):
            # dict preserves insertion order in py3.7+, but we prefer explicit tuples elsewhere in this file
            return urlencode(params, doseq=True)
        # sequence of tuples -> preserves order
        return urlencode(list(params), doseq=True)

    @staticmethod
    def _canonical_body(payload: Optional[Dict[str, Any]]) -> str:
        if not payload:
            return ""
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def _sign(self, method: str, path: str, params: ParamsType, payload: Optional[Dict[str, Any]]) -> Tuple[Dict[str, str], str, str]:
        ts = self._now_epoch_seconds_str()
        qs = self._qs_from(params)
        body = self._canonical_body(payload)
        # IMPORTANT: include '?' between path and query when query exists
        sig = method.upper() + ts + path + (f"?{qs}" if qs else "") + body
        signature = hmac.new(self.api_secret_bytes, sig.encode(), hashlib.sha256).hexdigest()
        headers = {"api-key": self.api_key, "timestamp": ts, "signature": signature}
        return headers, sig, qs  # return qs for logging/preview

    def _request(self, method: str, path: str, params: ParamsType = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers, signature_data, qs = self._sign(method, path, params, payload)
        url = f"{BASE_URL}{path}"

        # requests can take params as list of tuples to preserve order
        self._log(f"REQUEST {method.upper()} {path}{('?' + qs) if qs else ''} | signed='{signature_data}'")

        try:
            if method.upper() == "GET":
                r = self.session.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
            else:
                r = self.session.request(method.upper(), url, params=params, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        except requests.Timeout as e:
            self._log(f"❌ Timeout calling {path}")
            raise DeltaAPIError(0, "timeout", f"Timeout calling {path}", None, signature_data) from e
        except requests.ConnectionError as e:
            self._log(f"❌ Network error calling {path}: {e}")
            raise DeltaAPIError(0, "connection_error", f"Network error calling {path}", None, signature_data) from e
        except Exception as e:
            self._log(f"❌ Unexpected error calling {path}: {e}")
            raise DeltaAPIError(0, "unknown_error", f"Unexpected error calling {path}: {e}", None, signature_data) from e

        self._log(f"RESPONSE {r.status_code} for {path}")

        if r.status_code >= 400:
            try:
                detail = r.json()
                code = (detail.get("error") or {}).get("code") if isinstance(detail, dict) else None
                msg = (detail.get("error") or {}).get("message") if isinstance(detail, dict) else None
            except Exception:
                detail, code, msg = {"text": r.text}, None, None
            self._log(f"❌ HTTP {r.status_code} {path} | code={code} msg={msg} | signed='{signature_data}'")
            raise DeltaAPIError(r.status_code, code, msg or "HTTP error", detail, signature_data)

        try:
            data = r.json()
            self._log(f"✅ OK {path}")
            return data
        except Exception:
            self._log(f"❌ Non-JSON response from {path}: {r.text[:200]}")
            raise DeltaAPIError(r.status_code, "bad_json", "Response not JSON", {"text": r.text}, signature_data)

    # ---------- pagination (uses ordered tuples) ----------
    def paginate(self, path: str, params: ParamsType = None, page_size: int = 500, max_pages: Optional[int] = None):
        # start with ordered tuple list; if params is dict, convert preserving insertion order
        if params is None:
            p_list: List[Tuple[str, Any]] = []
        elif isinstance(params, dict):
            p_list = list(params.items())
        else:
            p_list = list(params)

        # page_size LAST, to match Delta's examples: start_time & end_time first, then page_size
        p_list.append(("page_size", page_size))

        pages = 0
        while True:
            data = self._request("GET", path, params=p_list)
            yield data
            pages += 1
            if max_pages and pages >= max_pages:
                break
            meta = (data or {}).get("meta") or {}
            after = meta.get("after")
            if not after:
                break
            # append or replace 'after' at the end; keep start/end/page_size ordering intact
            # remove previous 'after' if present
            p_list = [(k, v) for (k, v) in p_list if k != "after"]
            p_list.append(("after", after))
            self._log(f"→ paginate next after={after}")

    # ---------- endpoints ----------
    def get_wallet_balances(self) -> List[Dict[str, Any]]:
        path = "/v2/wallet/balances"
        data = self._request("GET", path)
        result = data.get("result", []) if isinstance(data, dict) else []
        self._log(f"Balances count={len(result)}")
        return result

    def get_wallet_transactions(
        self,
        start_us: Optional[int] = None,
        end_us: Optional[int] = None,
        txn_types: Optional[List[str]] = None,
        asset_ids: Optional[List[int]] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # Build ordered params: start_time, end_time, (optionals...), page_size is added in paginate()
        p_list: List[Tuple[str, Any]] = []
        if start_us is not None:
            p_list.append(("start_time", start_us))
        if end_us is not None:
            p_list.append(("end_time", end_us))
        if txn_types:
            p_list.append(("transaction_types", ",".join(txn_types)))
        if asset_ids:
            p_list.append(("asset_ids", ",".join(map(str, asset_ids))))

        out: List[Dict[str, Any]] = []
        for page in self.paginate("/v2/wallet/transactions", params=p_list, max_pages=max_pages):
            out.extend((page or {}).get("result") or [])
        self._log(f"Wallet txns fetched={len(out)}")
        return out

    def get_fills(
        self,
        start_us: int,
        end_us: int,
        product_ids: Optional[List[int]] = None,
        contract_types: Optional[List[str]] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        p_list: List[Tuple[str, Any]] = [
            ("start_time", start_us),
            ("end_time", end_us),
        ]
        if product_ids:
            p_list.append(("product_ids", ",".join(map(str, product_ids))))
        if contract_types:
            p_list.append(("contract_types", ",".join(contract_types)))

        out: List[Dict[str, Any]] = []
        for page in self.paginate("/v2/fills", params=p_list, max_pages=max_pages):
            out.extend((page or {}).get("result") or [])
        self._log(f"Fills fetched={len(out)}")
        return out

# -----------------------
# Diagnostics helper
# -----------------------
def run_diagnostics(api_key: str, api_secret: str, minutes_back: int = 30, verbose: bool = True) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "clock_unix_seconds": int(time.time()),
        "public_products_ok": False,
        "balances_ok": False,
        "wallet_txns_ok": False,
        "fills_ok": False,
        "errors": [],
        "notes": [],
    }
    report["notes"].append("Ensure your system clock is NTP-synced; large skew can break signatures.")

    # Public check
    products_url = f"{BASE_URL}/v2/products"
    try:
        if verbose:
            print(f"[Diag] GET {products_url}")
        r = requests.get(products_url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        _ = r.json()
        report["public_products_ok"] = True
        if verbose:
            print("[Diag] ✅ Public products reachable")
    except Exception as e:
        report["errors"].append(f"Public products error: {e}")
        if verbose:
            print(f"[Diag] ❌ Public products error: {e}")

    client = DeltaIndiaClient(api_key, api_secret, verbose=verbose)

    # Balances
    try:
        _ = client.get_wallet_balances()
        report["balances_ok"] = True
        if verbose:
            print("[Diag] ✅ Balances OK")
    except DeltaAPIError as e:
        report["errors"].append(f"Balances API error: {e} | signed='{e.signature_data}' | payload={e.payload}")
        if verbose:
            print(f"[Diag] ❌ Balances API error: {e} | signed='{e.signature_data}' | payload={e.payload}")
    except Exception as e:
        report["errors"].append(f"Balances unknown error: {e}")
        if verbose:
            print(f"[Diag] ❌ Balances unknown error: {e}")

    # Time window
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(minutes=minutes_back)
    start_us = int(start_dt.timestamp() * 1_000_000)
    end_us = int(end_dt.timestamp() * 1_000_000)

    # Wallet txns
    try:
        _ = client.get_wallet_transactions(start_us=start_us, end_us=end_us, max_pages=2)
        report["wallet_txns_ok"] = True
        if verbose:
            print("[Diag] ✅ Wallet transactions OK")
    except DeltaAPIError as e:
        report["errors"].append(f"Wallet txns API error: {e} | signed='{e.signature_data}' | payload={e.payload}")
        if verbose:
            print(f"[Diag] ❌ Wallet txns API error: {e} | signed='{e.signature_data}' | payload={e.payload}")
    except Exception as e:
        report["errors"].append(f"Wallet txns unknown error: {e}")
        if verbose:
            print(f"[Diag] ❌ Wallet txns unknown error: {e}")

    # Fills
    try:
        _ = client.get_fills(start_us=start_us, end_us=end_us, max_pages=2)
        report["fills_ok"] = True
        if verbose:
            print("[Diag] ✅ Fills OK")
    except DeltaAPIError as e:
        report["errors"].append(f"Fills API error: {e} | signed='{e.signature_data}' | payload={e.payload}")
        if verbose:
            print(f"[Diag] ❌ Fills API error: {e} | signed='{e.signature_data}' | payload={e.payload}")
    except Exception as e:
        report["errors"].append(f"Fills unknown error: {e}")
        if verbose:
            print(f"[Diag] ❌ Fills unknown error: {e}")

    if verbose:
        print(f"[Diag Summary] public={report['public_products_ok']} balances={report['balances_ok']} txns={report['wallet_txns_ok']} fills={report['fills_ok']}")
        if report["errors"]:
            print("[Diag Errors]")
            for err in report["errors"]:
                print(" -", err)
    return report


if __name__ == "__main__":
    KEY = "YOUR_API_KEY_HERE"
    SEC = "YOUR_API_SECRET_HERE"
    print("Running Delta India diagnostics...")
    diag = run_diagnostics(KEY, SEC, minutes_back=30, verbose=True)
    print("\nDiagnostics dict:\n", json.dumps(diag, indent=2, default=str))
