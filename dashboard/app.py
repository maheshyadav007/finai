# app.py
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict
from zoneinfo import ZoneInfo

from nicegui import ui
from CONSTANTS import API_KEY, API_SECRET
from totals_service import TotalsService

IST = ZoneInfo("Asia/Kolkata")

def _fmt(symbol: str, value: Optional[float]) -> str:
    if value is None:
        return f'{symbol} —'
    return f'{symbol} {value:,.2f}'


# ---------- CONFIG: choose which currencies to expose ----------
CURRENCY_CONFIG: Dict[str, Dict] = {
    "USD": {"symbol": "$", "factor": 1.0,  "editable": False, "label": "USD"},
    "INR": {"symbol": "₹", "factor": 85.0, "editable": False,  "label": "INR per USD"},
    # "EUR": {"symbol": "€", "factor": 0.92, "editable": True, "label": "EUR per USD"},  # <- enable by uncommenting
}
DEFAULT_CCY = "INR"   # which one is selected on load


class App:
    def __init__(self):
        # service uses the config above
        self.svc = TotalsService(API_KEY, API_SECRET, verbose=False, enabled_currencies=CURRENCY_CONFIG)
        self.currency = DEFAULT_CCY
        self.last_usd_totals = None
        self.last_end_balance_usd = None  # optional if you calculate it later

        with ui.header().classes('items-center justify-between'):
            ui.label('📊 Wallet KPIs').classes('text-lg font-semibold')

        # Controls
        with ui.card().classes('p-4 w-full'):
            with ui.row().classes('items-end gap-3'):
                today = datetime.now(IST).date()
                default_start = (datetime.now(IST) - timedelta(days=7)).date()
                self.start_input = ui.input('Start date', value=str(default_start)).props('type=date')
                self.end_input   = ui.input('End date', value=str(today)).props('type=date')
                self.asset_input = ui.input('Asset (USD / USDT)', value='USD')
                self.asset_input.props('hidden')  # hide for now; future use

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
                        self.rate_inputs[code] = ui.number(label, value=factor, step=0.1 if code != "EUR" else 0.01).classes('min-w-[12rem]')

                ui.button('Sync', icon='sync', color='primary', on_click=self.on_sync)

        # KPI grid
        with ui.grid(columns=5).classes('gap-4 w-full'):
            self.kpi_invested = self._kpi_card('Invested Capital')    # total_deposit
            self.kpi_current  = self._kpi_card('Current Capital')     # total_cashflow
            self.kpi_netcur   = self._kpi_card('Net Current Capital') # cashflow - commission + funding
            self.kpi_comm     = self._kpi_card('Total Commission')
            self.kpi_fund     = self._kpi_card('Total Funding')

        # Debug
        with ui.card().classes('p-4 w-full'):
            ui.label('Debug / Errors').classes('text-base font-medium')
            self.debug = ui.textarea('', value='Ready').props('rows=8').classes('w-full')

        # Events
        self.ccy_toggle.on('update:model-value', self._on_currency_change)
        for code, inp in self.rate_inputs.items():
            inp.on('update:model-value', self._on_rate_change)

    def _kpi_card(self, title: str):
        with ui.card().classes('p-4 rounded-2xl shadow-sm'):
            ui.label(title).classes('text-sm text-gray-500')
            val = ui.label('—').classes('text-2xl font-semibold')
            return val

    def _on_currency_change(self, _e):
        self.currency = str(self.ccy_toggle.value).upper()
        self.apply_currency()  # re-render without refetch

    def _on_rate_change(self, _e):
        # Collect all editable rates and update service
        updates = {}
        for code, inp in self.rate_inputs.items():
            try:
                updates[code] = float(inp.value)
            except Exception:
                pass
        if updates:
            self.svc.set_rates(**updates)
            self.apply_currency()

    async def on_sync(self):
        start = (self.start_input.value or '').strip()
        end   = (self.end_input.value or '').strip()
        asset = (self.asset_input.value or '').strip().upper() or None

        self.debug.value = 'Running...'
        try:
            self.last_usd_totals = await asyncio.to_thread(self.svc.compute, start, end, asset)
        except Exception as e:
            self.debug.value = f'❌ {e}'
            return

        self.apply_currency()
        self.debug.value = (
            '✅ Done\n'
            f"Window: {self.last_usd_totals.get('window_ist','—')} | Asset: {self.last_usd_totals.get('asset','—')}\n"
            f"Txn Count: {self.last_usd_totals.get('count','—')}\n"
            "Formulas: Invested=Deposit | Current=Cashflow | NetCurrent=Cashflow-Commission+Funding"
        )

    def apply_currency(self):
        if not self.last_usd_totals:
            return
        tgt = self.currency
        symbol = self.svc.convert_totals(self.last_usd_totals, tgt).get("symbol", "$")

        dep_usd   = float(self.last_usd_totals.get('total_deposit',    0.0) or 0.0)
        cash_usd  = float(self.last_usd_totals.get('total_cashflow',   0.0) or 0.0)
        comm_usd  = float(self.last_usd_totals.get('total_commission', 0.0) or 0.0)
        fund_usd  = float(self.last_usd_totals.get('total_funding',    0.0) or 0.0)

        invested_usd    = dep_usd
        current_usd     = cash_usd
        net_current_usd = cash_usd - comm_usd + fund_usd

        self.kpi_invested.text = _fmt(symbol, self.svc.convert_amount(invested_usd, tgt))
        self.kpi_current.text  = _fmt(symbol, self.svc.convert_amount(current_usd,  tgt))
        self.kpi_netcur.text   = _fmt(symbol, self.svc.convert_amount(net_current_usd, tgt))
        self.kpi_comm.text     = _fmt(symbol, self.svc.convert_amount(comm_usd,   tgt))
        self.kpi_fund.text     = _fmt(symbol, self.svc.convert_amount(fund_usd,   tgt))


# run
App()
ui.run(title='Wallet KPIs', favicon='📊', reload=True)
