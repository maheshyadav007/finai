# app.py
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from zoneinfo import ZoneInfo

import pandas as pd
from nicegui import ui

from delta_client import DeltaIndiaClient, DeltaAPIError
from CONSTANTS import API_KEY, API_SECRET

IST = ZoneInfo("Asia/Kolkata")

# ======================
# Global settings
# ======================
VERBOSE_DEFAULT = False   # default verbosity at app start (you can toggle in UI)


# ======================
# Config: Accounts
# ======================
ACCOUNTS = [
    {
        "name": "Delta India — Main",
        "exchange": "delta_india",
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "base_ccy": "USDT",
        "verbose": VERBOSE_DEFAULT,  # per-account verbose (inherits global toggle at runtime)
    },
    # Add more accounts with the same shape if needed
]


# ======================
# Data & Error Models
# ======================
@dataclass
class SnapshotResult:
    success: bool
    error: Optional[str]
    name: str
    base_ccy: str
    equity_now: float = 0.0
    invested: float = 0.0
    current_pnl: float = 0.0
    realized_t: float = 0.0
    total_trades_t: int = 0
    win_rate_t: float = 0.0
    monthly_pnl: Dict[str, float] = None
    debug_ctx: Dict[str, Any] = None  # extra context for debugging

    @staticmethod
    def error_result(name: str, base_ccy: str, err: str, debug_ctx: Optional[Dict[str, Any]] = None) -> 'SnapshotResult':
        return SnapshotResult(
            success=False,
            error=err,
            name=name,
            base_ccy=base_ccy,
            monthly_pnl={},
            debug_ctx=debug_ctx or {},
        )

    @staticmethod
    def ok_result(name: str, base_ccy: str, payload: Dict[str, Any], debug_ctx: Optional[Dict[str, Any]] = None) -> 'SnapshotResult':
        return SnapshotResult(
            success=True,
            error=None,
            name=name,
            base_ccy=base_ccy,
            equity_now=float(payload.get("equity_now", 0.0)),
            invested=float(payload.get("invested", 0.0)),
            current_pnl=float(payload.get("current_pnl", 0.0)),
            realized_t=float(payload.get("realized_t", 0.0)),
            total_trades_t=int(payload.get("total_trades_t", 0)),
            win_rate_t=float(payload.get("win_rate_t", 0.0)),
            monthly_pnl=dict(payload.get("monthly_pnl", {})),
            debug_ctx=debug_ctx or {},
        )


# ======================
# Utilities
# ======================
def to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)

def parse_dt(s: str) -> datetime:
    return pd.to_datetime(s, utc=True).to_pydatetime().astimezone(IST)

def ym_key(ts: datetime) -> str:
    return ts.strftime("%Y-%m")

def split_pos_neg(values: List[float]) -> tuple[List[float], List[float]]:
    pos = [v if v >= 0 else 0 for v in values]
    neg = [v if v < 0 else 0 for v in values]
    return pos, neg

def update_monthly_chart(chart, months: List[str], values: List[float]):
    pos, neg = split_pos_neg(values)
    chart.options.update({
        'xAxis': {'type': 'category', 'data': months},
        'yAxis': {'type': 'value'},
        'tooltip': {'trigger': 'axis'},
        'series': [
            {'name': 'Positive', 'type': 'bar', 'data': pos, 'itemStyle': {'color': '#4ade80'}},
            {'name': 'Negative', 'type': 'bar', 'data': neg, 'itemStyle': {'color': '#f87171'}},
        ],
        'legend': {'show': True},
    })
    chart.update()

def tlog(verbose: bool, *msg):
    """Timestamped console log controlled by verbose flag."""
    if verbose:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[APP {ts}]", *msg)


# ======================
# Adapter Layer
# ======================
class BaseAccountAdapter:
    def get_snapshot(self, cfg: Dict[str, Any], t_start: datetime, t_end: datetime, verbose: bool = False) -> SnapshotResult:
        raise NotImplementedError


class DeltaIndiaAdapter(BaseAccountAdapter):
    """Wraps DeltaIndiaClient with robust try/except and returns SnapshotResult."""

    def get_snapshot(self, cfg: Dict[str, Any], t_start: datetime, t_end: datetime, verbose: bool = False) -> SnapshotResult:
        name = cfg.get("name", "Delta India")
        base = cfg.get("base_ccy", "USDT")
        client_verbose = bool(cfg.get("verbose", False) or verbose)

        debug_ctx: Dict[str, Any] = {
            "exchange": "delta_india",
            "t_start": t_start.isoformat(),
            "t_end": t_end.isoformat(),
        }

        try:
            start_wall = time.perf_counter()
            client = DeltaIndiaClient(cfg["api_key"], cfg["api_secret"], verbose=client_verbose)

            # balances
            try:
                t0 = time.perf_counter()
                balances = client.get_wallet_balances()
                debug_ctx["balances_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            except DeltaAPIError as e:
                debug_ctx["balances_error"] = {"message": str(e), "payload": e.payload, "signed": e.signature_data}
                return SnapshotResult.error_result(name, base, f"Balances error: {e}", debug_ctx)
            except Exception as e:
                debug_ctx["balances_error"] = {"message": str(e)}
                return SnapshotResult.error_result(name, base, f"Balances error: {e}", debug_ctx)

            equity_now = 0.0
            for b in balances:
                asset = str(b.get("asset", "")).upper()
                if asset == base and "equity" in b:
                    equity_now = float(b["equity"])
                    break
            if equity_now == 0.0:
                eq_sum, ok = 0.0, False
                for b in balances:
                    if "equity" in b:
                        ok = True
                        eq_sum += float(b["equity"])
                if ok:
                    equity_now = eq_sum

            # txns
            try:
                t0 = time.perf_counter()
                txns = client.get_wallet_transactions(max_pages=40)
                debug_ctx["wallet_txns_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                debug_ctx["wallet_txns_count"] = len(txns)
            except DeltaAPIError as e:
                debug_ctx["wallet_txns_error"] = {"message": str(e), "payload": e.payload, "signed": e.signature_data}
                return SnapshotResult.error_result(name, base, f"Transactions error: {e}", debug_ctx)
            except Exception as e:
                debug_ctx["wallet_txns_error"] = {"message": str(e)}
                return SnapshotResult.error_result(name, base, f"Transactions error: {e}", debug_ctx)

            tx_df = pd.DataFrame(txns)
            if not tx_df.empty:
                tx_df["created_at"] = tx_df["created_at"].apply(parse_dt)
                tx_df["amount"] = tx_df["amount"].astype(float)
                tx_df["transaction_type"] = tx_df["transaction_type"].astype(str)
            else:
                tx_df = pd.DataFrame(columns=["created_at", "amount", "transaction_type", "product_id"])

            # Invested
            deposits = tx_df[tx_df["transaction_type"].str.lower().str.contains("deposit")]["amount"].sum()
            withdrawals = tx_df[tx_df["transaction_type"].str.lower().str.contains("withdraw")]["amount"].sum()
            invested = float(deposits - withdrawals)

            # Monthly realized PnL
            pnl_df = tx_df[tx_df["transaction_type"].isin(["settlement", "funding", "commission"])].copy()
            monthly_pnl: Dict[str, float] = defaultdict(float)
            if not pnl_df.empty:
                for row in pnl_df.itertuples(index=False):
                    monthly_pnl[ym_key(row.created_at)] += float(row.amount)

            # realized in t
            t_mask = (tx_df["created_at"] >= t_start) & (tx_df["created_at"] <= t_end)
            realized_t = tx_df.loc[
                t_mask & tx_df["transaction_type"].isin(["settlement", "funding", "commission"]),
                "amount",
            ].sum()

            # fills for trades/win-rate
            try:
                t0 = time.perf_counter()
                fills = client.get_fills(start_us=to_us(t_start), end_us=to_us(t_end), max_pages=40)
                debug_ctx["fills_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                debug_ctx["fills_count"] = len(fills)
            except DeltaAPIError as e:
                debug_ctx["fills_error"] = {"message": str(e), "payload": e.payload, "signed": e.signature_data}
                return SnapshotResult.error_result(name, base, f"Fills error: {e}", debug_ctx)
            except Exception as e:
                debug_ctx["fills_error"] = {"message": str(e)}
                return SnapshotResult.error_result(name, base, f"Fills error: {e}", debug_ctx)

            fills_df = pd.DataFrame(fills)
            total_trades, win_rate = 0, 0.0
            if not fills_df.empty:
                fills_df["created_at"] = fills_df["created_at"].apply(parse_dt)
                fills_df["side"] = fills_df["side"].astype(str)
                fills_df["size"] = fills_df["size"].astype(int)
                total_trades, win_rate = self._compute_trades_and_win_rate(fills_df, tx_df, t_start, t_end)

            current_pnl = float(equity_now - invested)

            debug_ctx["wall_ms"] = round((time.perf_counter() - start_wall) * 1000, 2)
            return SnapshotResult.ok_result(name, base, {
                "equity_now": equity_now,
                "invested": invested,
                "current_pnl": current_pnl,
                "realized_t": realized_t,
                "total_trades_t": total_trades,
                "win_rate_t": win_rate,
                "monthly_pnl": dict(monthly_pnl),
            }, debug_ctx)

        except DeltaAPIError as e:
            debug_ctx["fatal_error"] = {"message": str(e), "payload": e.payload, "signed": e.signature_data}
            return SnapshotResult.error_result(name, base, f"API error: {e}", debug_ctx)
        except Exception as e:
            debug_ctx["fatal_error"] = {"message": str(e)}
            return SnapshotResult.error_result(name, base, f"Unexpected: {e}", debug_ctx)

    @staticmethod
    def _compute_trades_and_win_rate(fills_df: pd.DataFrame, tx_df: pd.DataFrame, t_start: datetime, t_end: datetime) -> Tuple[int, float]:
        w = tx_df[tx_df["transaction_type"].isin(["settlement", "funding", "commission"])].copy()
        total = 0
        wins = 0
        for pid, g in fills_df.sort_values("created_at").groupby("product_id", dropna=False):
            pos = 0
            entry_time = None
            for row in g.itertuples(index=False):
                side_sign = 1 if row.side == "buy" else -1
                size = int(row.size or 0)
                t = row.created_at
                prev = pos
                pos += side_sign * size
                if prev == 0 and pos != 0:
                    entry_time = t
                if entry_time is not None and pos == 0:
                    exit_time = t
                    if exit_time >= t_start and entry_time <= t_end:
                        total += 1
                        sub = w[
                            (w["product_id"] == pid) &
                            (w["created_at"] >= entry_time) &
                            (w["created_at"] <= exit_time)
                        ]
                        if sub["amount"].sum() > 0:
                            wins += 1
                    entry_time = None
        win_rate = (wins / total * 100.0) if total else 0.0
        return total, win_rate


# Registry (add more adapters later)
ADAPTERS: Dict[str, BaseAccountAdapter] = {
    "delta_india": DeltaIndiaAdapter(),
}


# ======================
# Dashboard Controller
# ======================
class Dashboard:
    def __init__(self):
        self.global_verbose = VERBOSE_DEFAULT
        self.last_errors: Dict[str, str] = {}   # per-account error strings for the Debug drawer

        # Header
        with ui.header().classes('items-center justify-between'):
            ui.label('📊 Multi-Account Trading Dashboard').classes('text-xl font-semibold')
            tz_now = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')
            self.clock_label = ui.label(f'IST: {tz_now}').classes('text-sm text-gray-500')

        # Controls & KPIs
        with ui.grid(columns=3).classes('gap-4 w-full'):
            with ui.card().classes('p-4'):
                ui.label('Date Range (t) • IST').classes('text-base font-medium')
                today = datetime.now(IST).date()
                default_start = (datetime.now(IST) - timedelta(days=30)).date()
                self.start_input = ui.input('Start date (YYYY-MM-DD)', value=str(default_start))
                self.end_input = ui.input('End date (YYYY-MM-DD)', value=str(today))
                self.refresh_btn = ui.button('Refresh', color='primary')

            with ui.card().classes('p-4'):
                ui.label('All Accounts — KPIs').classes('text-base font-medium')
                self.kpi_invested = ui.label('Invested: —')
                self.kpi_equity = ui.label('Equity: —')
                self.kpi_current = ui.label('Current PnL: —')
                self.kpi_realized_t = ui.label('Realized PnL (t): —')
                self.kpi_trades = ui.label('Trades / Win-rate (t): —')

            with ui.card().classes('p-4'):
                ui.label('Monthly PnL — All Accounts').classes('text-base font-medium')
                self.chart_all = ui.echart({'xAxis': {'type': 'category', 'data': []},
                                            'yAxis': {'type': 'value'},
                                            'series': [],
                                            'tooltip': {'trigger': 'axis'},
                                            'legend': {'show': True}}).classes('w-full h-64')

        # Debug drawer
        with ui.expansion('🛠 Debug').classes('w-full'):
            with ui.row().classes('items-center gap-4'):
                self.verbose_switch = ui.switch('Verbose logs', value=self.global_verbose)
                ui.label('Shows extra logs in terminal and captures timing/error context.')
            self.debug_area = (
                ui.textarea('Last errors / diagnostics (per account)', value='')
                .props('autogrow')        # Quasar prop for auto-resizing
                .classes('w-full')
            )

        ui.separator()
        ui.label('Accounts').classes('text-lg font-semibold')

        # Per-account UI
        self.account_cards: List[Dict[str, Any]] = []
        self.monthly_charts: List[Any] = []
        with ui.grid(columns=3).classes('gap-4 w-full'):
            for acc in ACCOUNTS:
                with ui.card().classes('p-4'):
                    ui.label(acc["name"]).classes('text-base font-medium')
                    status = ui.label('⏳ Waiting...').classes('text-sm')
                    lbl_inv = ui.label('Invested: —')
                    lbl_eq = ui.label('Equity: —')
                    lbl_cur = ui.label('Current PnL: —')
                    lbl_real = ui.label('Realized PnL (t): —')
                    lbl_trd = ui.label('Trades / Win-rate (t): —')
                    self.account_cards.append({
                        "name": acc["name"],
                        "status": status,
                        "labels": (lbl_inv, lbl_eq, lbl_cur, lbl_real, lbl_trd),
                        "cfg": acc,
                    })
                    ui.label('Monthly PnL').classes('text-sm text-gray-500')
                    chart = ui.echart({'xAxis': {'type': 'category', 'data': []},
                                       'yAxis': {'type': 'value'},
                                       'series': [],
                                       'tooltip': {'trigger': 'axis'},
                                       'legend': {'show': True}}).classes('w-full h-48')
                    self.monthly_charts.append(chart)

        # Bindings
        self.refresh_btn.on_click(self._on_refresh_click)
        self.verbose_switch.on('update:model-value', self._on_verbose_toggle)

        # timers
        ui.timer(0.5, lambda: asyncio.create_task(self.refresh()), once=True)
        ui.timer(30.0, lambda: asyncio.create_task(self.refresh()))
        ui.timer(1.0, self._tick_clock)


    def _on_verbose_toggle(self, _e):
        # read directly from the component; works across NiceGUI versions
        self.global_verbose = bool(self.verbose_switch.value)
        tlog(self.global_verbose, "Verbose toggle set to", self.global_verbose)
        for card in self.account_cards:
            card["cfg"]["verbose"] = self.global_verbose


    def _tick_clock(self):
        self.clock_label.text = f'IST: {datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")}'

    async def _on_refresh_click(self):
        await self.refresh()

    async def refresh(self):
        # Parse dates safely
        try:
            t_start = datetime.strptime(self.start_input.value.strip(), '%Y-%m-%d').replace(tzinfo=IST)
            t_end = datetime.strptime(self.end_input.value.strip(), '%Y-%m-%d').replace(tzinfo=IST) + timedelta(hours=23, minutes=59, seconds=59)
        except Exception:
            ui.notify('Invalid date format. Use YYYY-MM-DD.', color='negative')
            return

        # Fetch all accounts concurrently (threaded, since requests is blocking)
        async def one(card_idx: int):
            cfg = self.account_cards[card_idx]["cfg"]
            adapter = ADAPTERS.get(cfg["exchange"])
            if not adapter:
                return SnapshotResult.error_result(cfg["name"], cfg.get("base_ccy", "USDT"), "No adapter for exchange")
            # Pass global verbose down (per-account cfg may also set verbose)
            return await asyncio.to_thread(adapter.get_snapshot, cfg, t_start, t_end, self.global_verbose)

        try:
            results = await asyncio.gather(*(one(i) for i in range(len(self.account_cards))), return_exceptions=True)
        except Exception as e:
            # truly unexpected (shouldn't happen often)
            tlog(self.global_verbose, "Refresh gather error:", e)
            ui.notify(f'Refresh failed: {e}', color='negative')
            return

        # Aggregate & update UI
        total_invested = 0.0
        total_equity = 0.0
        total_realized_t = 0.0
        total_trades = 0
        sum_win_rate_weighted = 0.0
        monthly_agg: Dict[str, float] = defaultdict(float)

        self.last_errors.clear()

        for idx, res in enumerate(results):
            status_lbl = self.account_cards[idx]["status"]
            labels = self.account_cards[idx]["labels"]
            chart = self.monthly_charts[idx]
            acc_name = self.account_cards[idx]["name"]

            # Handle exceptions at task level
            if isinstance(res, Exception):
                status_lbl.text = f'❌ Error: {res}'
                self.last_errors[acc_name] = str(res)
                labels[0].text = 'Invested: —'
                labels[1].text = 'Equity: —'
                labels[2].text = 'Current PnL: —'
                labels[3].text = 'Realized PnL (t): —'
                labels[4].text = 'Trades / Win-rate (t): —'
                chart.options.update({'xAxis': {'data': []}, 'series': []})
                chart.update()
                continue

            # SnapshotResult branch
            if not res.success:
                status_lbl.text = f'❌ Error: {res.error}'
                # capture formatted debug context for the drawer
                self.last_errors[acc_name] = self._fmt_debug(res.error, res.debug_ctx)
                labels[0].text = 'Invested: —'
                labels[1].text = 'Equity: —'
                labels[2].text = 'Current PnL: —'
                labels[3].text = 'Realized PnL (t): —'
                labels[4].text = 'Trades / Win-rate (t): —'
                chart.options.update({'xAxis': {'data': []}, 'series': []})
                chart.update()
                continue

            # Success
            status_lbl.text = '✅ Success'
            ccy = res.base_ccy
            invested = res.invested
            equity = res.equity_now
            current = res.current_pnl
            realized_t = res.realized_t
            trades_t = res.total_trades_t
            win_rate = res.win_rate_t

            total_invested += invested
            total_equity += equity
            total_realized_t += realized_t
            total_trades += trades_t
            if trades_t > 0:
                sum_win_rate_weighted += win_rate * trades_t

            # Per-account monthly chart
            months_sorted = sorted(res.monthly_pnl.keys())
            data_vals = [round(res.monthly_pnl[m], 4) for m in months_sorted]
            update_monthly_chart(chart, months_sorted, data_vals)

            # Labels
            labels[0].text = f'Invested: {invested:,.2f} {ccy}'
            labels[1].text = f'Equity: {equity:,.2f} {ccy}'
            labels[2].text = f'Current PnL: {current:,.2f} {ccy}'
            labels[3].text = f'Realized PnL (t): {realized_t:,.2f} {ccy}'
            labels[4].text = f'Trades / Win-rate (t): {trades_t} / {win_rate:.1f}%'

            # Aggregate monthly
            for m, v in res.monthly_pnl.items():
                monthly_agg[m] += v

        # Update global KPIs
        self.kpi_invested.text = f'Invested: {total_invested:,.2f}'
        self.kpi_equity.text = f'Equity: {total_equity:,.2f}'
        self.kpi_current.text = f'Current PnL: {(total_equity - total_invested):,.2f}'
        self.kpi_realized_t.text = f'Realized PnL (t): {total_realized_t:,.2f}'
        if total_trades > 0:
            self.kpi_trades.text = f'Trades / Win-rate (t): {total_trades} / {(sum_win_rate_weighted/total_trades):.1f}%'
        else:
            self.kpi_trades.text = 'Trades / Win-rate (t): 0 / 0.0%'

        # Global monthly chart
        months_all = sorted(monthly_agg.keys())
        data_all = [round(monthly_agg[m], 4) for m in months_all]
        update_monthly_chart(self.chart_all, months_all, data_all)

        # Update debug drawer contents
        self._refresh_debug_drawer()

    def _fmt_debug(self, head: str, ctx: Optional[Dict[str, Any]]) -> str:
        parts = [f"{head}"]
        if not ctx:
            return head
        for k in ["balances_error", "wallet_txns_error", "fills_error", "fatal_error"]:
            if k in ctx and ctx[k]:
                entry = ctx[k]
                msg = entry.get("message")
                signed = entry.get("signed")
                payload = entry.get("payload")
                seg = f"\n— {k}: {msg}"
                if signed:
                    seg += f"\n   signed: {signed}"
                if payload is not None:
                    seg += f"\n   payload: {payload}"
                parts.append(seg)
        # timings
        timings = []
        for k in ["balances_ms", "wallet_txns_ms", "fills_ms", "wall_ms"]:
            if k in ctx:
                timings.append(f"{k}={ctx[k]}ms")
        if timings:
            parts.append("\nTimings: " + ", ".join(timings))
        return "\n".join(parts)

    def _refresh_debug_drawer(self):
        if not self.last_errors:
            self.debug_area.value = "No errors. All good ✅"
            return
        chunks = []
        for acc, err in self.last_errors.items():
            chunks.append(f"[{acc}]\n{err}\n")
        self.debug_area.value = "\n".join(chunks)


# Instantiate and run
Dashboard()
ui.run(title='Trading Monitor', favicon='📊', reload=False)
