# app.py
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict
from zoneinfo import ZoneInfo

import pandas as pd
from nicegui import ui

from CONSTANTS import API_KEY, API_SECRET
from totals_service import TotalsService                 # currency conversion helpers
from delta_txn_totals import fetch_metrics               # window (pnl / fees / net)
from delta_lifetime_flows import compute_lifetime_flows  # lifetime (deposit/withdraw/transfer/net)
from delta_client import DeltaIndiaClient                # for current balance

IST = ZoneInfo("Asia/Kolkata")


def _fmt(symbol: str, value: Optional[float]) -> str:
    if value is None:
        return f'{symbol} —'
    return f'{symbol} {value:,.2f}'


# ---------- CONFIG: choose which currencies to expose ----------
CURRENCY_CONFIG: Dict[str, Dict] = {
    # code:   symbol  factor (code per USD)  editable?   label for inline editor
    "USD": {"symbol": "$", "factor": 1.0,  "editable": False, "label": "USD"},
    "INR": {"symbol": "₹", "factor": 85.0, "editable": False, "label": "INR per USD"},
    # "EUR": {"symbol": "€", "factor": 0.92, "editable": True, "label": "EUR per USD"},
}
DEFAULT_CCY = "INR"


# ---------- tiny UI helpers ----------
def set_amount_with_color(label: ui.label, amount: Optional[float], symbol: str):
    """Set formatted amount and colorize: green>0, red<0, gray==0/None."""
    if amount is None:
        label.text = _fmt(symbol, None)
        label.style('color:#6b7280')  # gray-500
        return
    label.text = _fmt(symbol, amount)
    if amount > 0:
        label.style('color:#16a34a')  # green-600
    elif amount < 0:
        label.style('color:#dc2626')  # red-600
    else:
        label.style('color:#374151')  # gray-700


def set_percent_with_color(label: ui.label, pct: Optional[float]):
    """Show 'xx.xx%' with color; green if >=0, red if <0, gray if None."""
    if pct is None:
        label.text = '—'
        label.style('color:#6b7280')  # gray-500
        return
    label.text = f'{pct:.2f}%'
    if pct >= 0:
        label.style('color:#16a34a')  # green-600
    else:
        label.style('color:#dc2626')  # red-600


class App:
    def __init__(self):
        # currency service only (we don't use svc.compute now)
        self.svc = TotalsService(API_KEY, API_SECRET, verbose=False, enabled_currencies=CURRENCY_CONFIG)
        self.currency = DEFAULT_CCY

        # caches (always stored in USD base)
        self.last_summary_usd: Optional[dict] = None           # window summary
        self.last_daily_df_usd: Optional[pd.DataFrame] = None  # window daily (future charts)
        self.last_lifetime_usd: Optional[dict] = None          # lifetime totals
        self.last_balance_usd: Optional[float] = None          # current balance (USD asset, or fallback)

        # === Header ===
        with ui.header().classes('items-center justify-between'):
            ui.label('📊 Wallet KPIs').classes('text-lg font-semibold')

        # === Row 1: Lifetime KPIs (+ current balance with % of net external) ===
        with ui.grid(columns=6).classes('gap-4 w-full'):
            self.kpi_l_dep   = self._kpi_card('Lifetime Deposits')
            self.kpi_l_wdr   = self._kpi_card('Lifetime Withdrawals')
            self.kpi_l_tin   = self._kpi_card('Lifetime Transfer In')
            self.kpi_l_tout  = self._kpi_card('Lifetime Transfer Out')
            self.kpi_l_net   = self._kpi_card('Lifetime Net External')
            # Current balance has a subline for % of net external
            self.kpi_balance_val, self.kpi_balance_pct = self._kpi_card_with_sub(
                'Current Balance', sublabel='— of Net External'
            )

        # === Row 2: Controls (time range, currency) ===
        with ui.card().classes('p-4 w-full'):
            with ui.row().classes('items-end gap-3'):
                today = datetime.now(IST).date()
                default_start = (datetime.now(IST) - timedelta(days=7)).date()
                self.start_input = ui.input('Start date', value=str(default_start)).props('type=date')
                self.end_input   = ui.input('End date', value=str(today)).props('type=date')

                # (optional) asset filter; leave visible so you can change balance asset preference if needed
                self.asset_input = ui.input('Asset for Balance/Filters (e.g., USD / USDT)', value='USD')

                with ui.column().classes('gap-2'):
                    ui.label('Currency')
                    # Toggle built from config keys
                    choices = {c: c for c in self.svc.enabled_codes()}
                    self.ccy_toggle = ui.toggle(choices, value=DEFAULT_CCY)

                    # Editable rate inputs generated from config
                    self.rate_inputs: Dict[str, ui.number] = {}
                    for code in self.svc.editable_codes():
                        label = self.svc.label_for(code)
                        factor = self.svc.currencies[code]["factor"]
                        step = 0.01 if code in ("EUR",) else 0.1
                        self.rate_inputs[code] = ui.number(label, value=factor, step=step).classes('min-w-[12rem]')

                ui.button('Sync', icon='sync', color='primary', on_click=self.on_sync)

        # === Row 3: Window KPIs (pnl row by convention) ===
        with ui.grid(columns=5).classes('gap-4 w-full'):
            self.kpi_pnl   = self._kpi_card('PnL (cashflow)')
            self.kpi_fees  = self._kpi_card('Fees (all-in)')        # fees shown neutral color
            self.kpi_net   = self._kpi_card('Net Result (PnL - Fees)')
            self.kpi_dep   = self._kpi_card('Deposits (Window)')
            self.kpi_wdr   = self._kpi_card('Withdrawals (Window)')

        # === Debug ===
        with ui.card().classes('p-4 w-full'):
            ui.label('Debug').classes('text-base font-medium')
            self.debug = ui.textarea('', value='Ready').props('rows=8').classes('w-full')

        # events
        self.ccy_toggle.on('update:model-value', self._on_currency_change)
        for code, inp in self.rate_inputs.items():
            inp.on('update:model-value', self._on_rate_change)

    # ----- components -----
    def _kpi_card(self, title: str):
        with ui.card().classes('p-4 rounded-2xl shadow-sm'):
            ui.label(title).classes('text-sm text-gray-500')
            val = ui.label('—').classes('text-2xl font-semibold')
            return val

    def _kpi_card_with_sub(self, title: str, sublabel: str = '—'):
        with ui.card().classes('p-4 rounded-2xl shadow-sm'):
            ui.label(title).classes('text-sm text-gray-500')
            main = ui.label('—').classes('text-2xl font-semibold')
            sub  = ui.label(sublabel).classes('text-xs')
            return main, sub

    # ----- event handlers -----
    def _on_currency_change(self, _e):
        self.currency = str(self.ccy_toggle.value).upper()
        self.apply_currency()  # re-render without refetch

    def _on_rate_change(self, _e):
        # Collect editable rates and update service
        updates = {}
        for code, inp in self.rate_inputs.items():
            try:
                updates[code] = float(inp.value)
            except Exception:
                pass
        if updates:
            self.svc.set_rates(**updates)
            self.apply_currency()

    # ----- helpers -----
    async def _fetch_current_balance_usd(self, preferred_asset: Optional[str]) -> float:
        """
        Pull balances and return a USD-base number for the requested asset:
        - if list of dicts -> pick matching asset_symbol or first item
        - if single dict     -> use that
        Fallback: 0.0
        """
        def _sync_pull() -> float:
            c = DeltaIndiaClient(API_KEY, API_SECRET, verbose=False)
            try:
                bals = c.get_wallet_balances()
            except Exception:
                return 0.0

            if isinstance(bals, dict):
                return float(bals.get('balance', 0.0) or 0.0)

            if isinstance(bals, list) and bals:
                asset = (preferred_asset or '').upper().strip()
                chosen = None
                if asset:
                    for b in bals:
                        if str(b.get('asset_symbol', '')).upper() == asset:
                            chosen = b
                            break
                if not chosen:
                    chosen = bals[0]  # fallback
                return float(chosen.get('balance', 0.0) or 0.0)

            return 0.0

        return await asyncio.to_thread(_sync_pull)

    # ----- sync -----
    async def on_sync(self):
        start = (self.start_input.value or '').strip()
        end   = (self.end_input.value or '').strip()
        asset = (self.asset_input.value or '').strip().upper() or None   # e.g., 'USD' or 'USDT'

        self.debug.value = 'Running...'
        try:
            # WINDOW: One call → summary + daily (USD base numbers)
            out = await asyncio.to_thread(fetch_metrics, API_KEY, API_SECRET, start, end, asset, False)
            self.last_summary_usd = out["summary"]
            self.last_daily_df_usd = pd.DataFrame(out["daily"]) if out.get("daily") else None

            # LIFETIME: Overall flows since inception (USD base)
            life = await asyncio.to_thread(compute_lifetime_flows, API_KEY, API_SECRET, asset, False, None)
            self.last_lifetime_usd = life

            # CURRENT BALANCE (USD asset or fallback)
            self.last_balance_usd = await self._fetch_current_balance_usd(asset or "USD")

            # Render in selected currency
            self.apply_currency()

            # Debug text
            lines = []
            if self.last_summary_usd:
                s = self.last_summary_usd
                lines.append(
                    f"Window: {s.get('window_ist','—')} | Asset: {s.get('asset','—')} | Txn Count: {s.get('count','—')}"
                )
                lines.append(
                    f"Window Transfers: in={s.get('transfer_in',0.0):,.2f} out={s.get('transfer_out',0.0):,.2f} (USD base)"
                )
            if self.last_lifetime_usd:
                l = self.last_lifetime_usd
                lines.append(
                    f"Lifetime: dep={l.get('total_deposit',0.0):,.2f}, wdr={l.get('total_withdrawal',0.0):,.2f}, "
                    f"tin={l.get('total_sub_transfer_in',0.0):,.2f}, tout={l.get('total_sub_transfer_out',0.0):,.2f}, "
                    f"net={l.get('net_external',0.0):,.2f} (USD base)"
                )
            lines.append(f"Current Balance (USD asset pref): {self.last_balance_usd:,.2f} (USD base)")
            lines.append("Convention: pnl=cashflow · fees=commission+liquidation_fee−funding · net=pnl−fees")
            self.debug.value = '✅ Done\n' + "\n".join(lines)

        except Exception as e:
            self.debug.value = f'❌ {e}'

    # ----- render -----
    def apply_currency(self):
        """Render cached USD results in the currently selected currency, with colors."""
        tgt = self.currency
        symbol = self.svc.currencies.get(tgt, {}).get("symbol", "$")

        # --- Lifetime row ---
        if self.last_lifetime_usd:
            L = self.last_lifetime_usd
            dep_usd  = float(L.get('total_deposit', 0.0) or 0.0)
            wdr_usd  = float(L.get('total_withdrawal', 0.0) or 0.0)
            tin_usd  = float(L.get('total_sub_transfer_in', 0.0) or 0.0)
            tout_usd = float(L.get('total_sub_transfer_out', 0.0) or 0.0)
            netx_usd = float(L.get('net_external', 0.0) or 0.0)

            # Lifetime totals (keep neutral color)
            self.kpi_l_dep.text  = _fmt(symbol, self.svc.convert_amount(dep_usd,  tgt));  self.kpi_l_dep.style('color:#374151')
            self.kpi_l_wdr.text  = _fmt(symbol, self.svc.convert_amount(wdr_usd,  tgt));  self.kpi_l_wdr.style('color:#374151')
            self.kpi_l_tin.text  = _fmt(symbol, self.svc.convert_amount(tin_usd,  tgt));  self.kpi_l_tin.style('color:#374151')
            self.kpi_l_tout.text = _fmt(symbol, self.svc.convert_amount(tout_usd, tgt));  self.kpi_l_tout.style('color:#374151')
            self.kpi_l_net.text  = _fmt(symbol, self.svc.convert_amount(netx_usd, tgt));  self.kpi_l_net.style('color:#374151')
        else:
            for lbl in (self.kpi_l_dep, self.kpi_l_wdr, self.kpi_l_tin, self.kpi_l_tout, self.kpi_l_net):
                lbl.text = _fmt(symbol, None)
                lbl.style('color:#6b7280')

        # Current Balance + % of Lifetime Net External
        if self.last_balance_usd is not None:
            bal_tgt = self.svc.convert_amount(float(self.last_balance_usd), tgt)
            set_amount_with_color(self.kpi_balance_val, bal_tgt, symbol)

            # compute % of lifetime net external in USD (ratio is currency invariant)
            pct = None
            if self.last_lifetime_usd:
                netx = float(self.last_lifetime_usd.get('net_external', 0.0) or 0.0)
                if netx != 0:
                    pct = ((float(self.last_balance_usd) / (netx )) * 100.0) - 100.0
            set_percent_with_color(self.kpi_balance_pct, pct)
        else:
            set_amount_with_color(self.kpi_balance_val, None, symbol)
            set_percent_with_color(self.kpi_balance_pct, None)

        # --- Window row (colorize pnl & net) ---
        if self.last_summary_usd:
            s = self.last_summary_usd
            pnl_usd  = float(s.get('pnl',  0.0) or 0.0)
            fees_usd = float(s.get('fees', 0.0) or 0.0)
            net_usd  = float(s.get('net_balance', 0.0) or 0.0)
            dep_usd  = float(s.get('deposit', 0.0) or 0.0)
            wdr_usd  = float(s.get('withdrawal', 0.0) or 0.0)

            pnl_tgt  = self.svc.convert_amount(pnl_usd,  tgt)
            fees_tgt = self.svc.convert_amount(fees_usd, tgt)
            net_tgt  = self.svc.convert_amount(net_usd,  tgt)
            dep_tgt  = self.svc.convert_amount(dep_usd,  tgt)
            wdr_tgt  = self.svc.convert_amount(wdr_usd,  tgt)

            # PnL & Net with color; Fees neutral gray
            set_amount_with_color(self.kpi_pnl, pnl_tgt, symbol)
            self.kpi_fees.text = _fmt(symbol, fees_tgt); self.kpi_fees.style('color:#374151')  # gray-700
            set_amount_with_color(self.kpi_net, net_tgt, symbol)

            # Deposits/Withdrawals neutral
            self.kpi_dep.text = _fmt(symbol, dep_tgt); self.kpi_dep.style('color:#374151')
            self.kpi_wdr.text = _fmt(symbol, wdr_tgt); self.kpi_wdr.style('color:#374151')
        else:
            for lbl in (self.kpi_pnl, self.kpi_fees, self.kpi_net, self.kpi_dep, self.kpi_wdr):
                lbl.text = _fmt(symbol, None)
                lbl.style('color:#6b7280')


# run
App()
ui.run(title='Wallet KPIs', favicon='📊', reload=True)
