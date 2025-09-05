# delta_txn_models.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, List, Type, TypeVar, Union
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

# -------------------------
# small parse helpers
# -------------------------
def _to_decimal(x: Any) -> Decimal:
    if x is None or x == "":
        return Decimal("0")
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")

def _parse_dt(s: Optional[str]) -> datetime:
    # API gives ISO UTC like '2025-09-03T05:05:28.010068Z'
    if not s:
        return datetime.now(UTC)
    try:
        # fast path: remove 'Z', parse as UTC
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(UTC)
    except Exception:
        return datetime.now(UTC)


# ============================================================
# Base class with common properties across ALL transaction types
# ============================================================
@dataclass
class DeltaTxnBase:
    amount: Decimal
    asset_id: Optional[int]
    asset_symbol: str
    balance: Decimal                 # balance AFTER txn (as provided by API)
    created_at: datetime             # parsed to UTC datetime
    fund_id: Optional[int]
    product_id: Optional[int]
    transaction_type: str            # lowercased canonical type
    user_id: Optional[int]
    uuid: str

    # raw meta (always present but contents differ by type)
    meta_data: Dict[str, Any] = field(default_factory=dict)

    # ----- convenience -----
    @property
    def created_at_ist(self) -> datetime:
        return self.created_at.astimezone(IST)

    def to_row(self) -> Dict[str, Any]:
        """Flat dict ideal for DataFrame creation / logging."""
        row = asdict(self)
        # make datetimes serializable
        row["created_at"] = self.created_at.isoformat()
        return row

    # ----- core parser shared by subclasses -----
    @classmethod
    def _from_api_common(cls, d: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "amount": _to_decimal(d.get("amount")),
            "asset_id": int(d["asset_id"]) if d.get("asset_id") is not None else None,
            "asset_symbol": str(d.get("asset_symbol") or "").upper(),
            "balance": _to_decimal(d.get("balance")),
            "created_at": _parse_dt(d.get("created_at")),
            "fund_id": int(d["fund_id"]) if d.get("fund_id") is not None else None,
            "product_id": int(d["product_id"]) if d.get("product_id") is not None else None,
            "transaction_type": str(d.get("transaction_type") or "").lower(),
            "user_id": int(d["user_id"]) if d.get("user_id") is not None else None,
            "uuid": str(d.get("uuid") or ""),
            "meta_data": dict(d.get("meta_data") or {}),
        }


# =========================================
# Concrete types (one class per transaction)
# =========================================
@dataclass
class FundingTxn(DeltaTxnBase):
    # meta
    product_symbol: Optional[str] = None

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "FundingTxn":
        base = cls._from_api_common(d)
        meta = base["meta_data"]
        return cls(**base, product_symbol=meta.get("product_symbol"))


@dataclass
class CommissionTxn(DeltaTxnBase):
    # meta
    amount_without_gst: Optional[Decimal] = None
    gst: Optional[Decimal] = None
    fill_uuid: Optional[str] = None
    product_symbol: Optional[str] = None

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "CommissionTxn":
        base = cls._from_api_common(d)
        meta = base["meta_data"]
        return cls(
            **base,
            amount_without_gst=_to_decimal(meta.get("amount_without_gst")),
            gst=_to_decimal(meta.get("gst")),
            fill_uuid=meta.get("fill_uuid"),
            product_symbol=meta.get("product_symbol"),
        )


@dataclass
class WithdrawalTxn(DeltaTxnBase):
    # meta
    withdrawal_id: Optional[str] = None

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "WithdrawalTxn":
        base = cls._from_api_common(d)
        meta = base["meta_data"]
        return cls(**base, withdrawal_id=meta.get("withdrawal_id"))


@dataclass
class CashflowTxn(DeltaTxnBase):
    # meta
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    position_size: Optional[Decimal] = None
    product_symbol: Optional[str] = None

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "CashflowTxn":
        base = cls._from_api_common(d)
        meta = base["meta_data"]
        return cls(
            **base,
            entry_price=_to_decimal(meta.get("entry_price")),
            exit_price=_to_decimal(meta.get("exit_price")),
            position_size=_to_decimal(meta.get("position_size")),
            product_symbol=meta.get("product_symbol"),
        )


@dataclass
class SubAccountTransferTxn(DeltaTxnBase):
    # meta
    transferee_id: Optional[int] = None
    transferrer_id: Optional[int] = None

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "SubAccountTransferTxn":
        base = cls._from_api_common(d)
        meta = base["meta_data"]
        try:
            transferee_id = int(meta["transferee_id"]) if meta.get("transferee_id") is not None else None
        except Exception:
            transferee_id = None
        try:
            transferrer_id = int(meta["transferrer_id"]) if meta.get("transferrer_id") is not None else None
        except Exception:
            transferrer_id = None
        return cls(**base, transferee_id=transferee_id, transferrer_id=transferrer_id)


@dataclass
class DepositTxn(DeltaTxnBase):
    # meta
    deposit_id: Optional[str] = None
    transaction_id: Optional[str] = None

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "DepositTxn":
        base = cls._from_api_common(d)
        meta = base["meta_data"]
        return cls(
            **base,
            deposit_id=meta.get("deposit_id"),
            transaction_id=meta.get("transaction_id"),
        )


@dataclass
class LiquidationFeeTxn(DeltaTxnBase):
    # meta
    amount_without_gst: Optional[Decimal] = None
    gst: Optional[Decimal] = None
    liquidation_order_id: Optional[str] = None
    margin_mode: Optional[str] = None
    product_symbol: Optional[str] = None

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "LiquidationFeeTxn":
        base = cls._from_api_common(d)
        meta = base["meta_data"]
        return cls(
            **base,
            amount_without_gst=_to_decimal(meta.get("amount_without_gst")),
            gst=_to_decimal(meta.get("gst")),
            liquidation_order_id=meta.get("liquidation_order_id"),
            margin_mode=meta.get("margin_mode"),
            product_symbol=meta.get("product_symbol"),
        )


# =========================================
# Factory + batch parse
# =========================================
TxnUnion = Union[
    FundingTxn,
    CommissionTxn,
    WithdrawalTxn,
    CashflowTxn,
    SubAccountTransferTxn,
    DepositTxn,
    LiquidationFeeTxn,
]

_FACTORY_MAP: Dict[str, Type[DeltaTxnBase]] = {
    "funding": FundingTxn,
    "commission": CommissionTxn,
    "withdrawal": WithdrawalTxn,
    "cashflow": CashflowTxn,
    "sub_account_transfer": SubAccountTransferTxn,
    "deposit": DepositTxn,
    "liquidation_fee": LiquidationFeeTxn,
}

def parse_txn(d: Dict[str, Any]) -> TxnUnion:
    """Return the right dataclass instance based on transaction_type."""
    t = str(d.get("transaction_type") or "").lower()
    cls = _FACTORY_MAP.get(t)
    if not cls:
        # fallback to base if some new type appears; keeps pipeline resilient
        base = DeltaTxnBase._from_api_common(d)
        return DeltaTxnBase(**base)  # type: ignore[return-value]
    return cls.from_api(d)  # type: ignore[return-value]

def parse_all_txns(rows: List[Dict[str, Any]]) -> List[TxnUnion]:
    """Convenience to parse a full API page/list into typed objects."""
    out: List[TxnUnion] = []
    for r in rows:
        try:
            out.append(parse_txn(r))
        except Exception as e:
            # keep robust: skip a bad row but continue
            # (you can log the offending row here if you want)
            # print("parse error:", e, r)
            pass
    return out
