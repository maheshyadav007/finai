# app.py
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from nicegui import ui

from CONSTANTS import API_KEY, API_SECRET
from delta_txn_totals import fetch_totals  # <-- use your working function

IST = ZoneInfo("Asia/Kolkata")


# -----------------------------
# Small wrapper service
# -----------------------------
class TotalsService:
    """Thin wrapper around your fetch_totals so UI code stays clean."""
    def __init__(self, api_key: str, api_secret: str, verbose: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.verbose = verbose

    def compute(self, start_date: str, end_date: str, asset_symbol: Optional[str]) -> dict:
        return fetch_totals(
            api_key=self.api_key,
            api_secret=self.api_secret,
            start_date=start_date,
            end_date=end_date,
            asset_symbol=(asset_symbol or None),
            verbose=self.verbose,
        )


# -----------------------------
# Minimal NiceGUI app
# -----------------------------
class App:
    def __init__(self):
        self.verbose = False
        self.svc = TotalsService(API_KEY, API_SECRET, verbose=self.verbose)

        # Header
        with ui.header().classes('items-center justify-between'):
            ui.label('📊 Delta Wallet Totals').classes('text-lg font-semibold')
            self.verbose_switch = ui.switch('Verbose logs', value=self.verbose)

        # Inputs
        with ui.card().classes('p-4 w-full'):
            with ui.row().classes('items-end gap-3'):
                today = datetime.now(IST).date()
                default_start = (datetime.now(IST) - timedelta(days=7)).date()
                self.start_input = (
                                        ui.input('Start date', value=str(default_start))
                                        .props('type=date')                # ← native calendar
                                        .classes('min-w-[14rem]')
                                    )

                self.end_input = (
                                    ui.input('End date', value=str(today))
                                    .props('type=date')                # ← native calendar
                                    .classes('min-w-[14rem]')
                                )

                self.asset_input = ui.input('Asset (optional: USD / USDT)', value='USD').classes('min-w-[14rem]')
                ui.button('Sync', icon='sync', color='primary', on_click=self.on_compute)


        # Results
        with ui.grid(columns=2).classes('gap-4 w-full'):
            with ui.card().classes('p-4'):
                ui.label('Results').classes('text-base font-medium')
                self.lbl_window = ui.label('Window: —')
                self.lbl_asset  = ui.label('Asset: —')
                self.lbl_count  = ui.label('Txn Count: —')
                self.lbl_comm   = ui.label('Total Commission: —')
                self.lbl_fund   = ui.label('Total Funding: —')
                self.lbl_cash   = ui.label('Total Cashflow: —')
                self.lbl_dep    = ui.label('Total Deposit: —')

            with ui.card().classes('p-4'):
                ui.label('Debug / Errors').classes('text-base font-medium')
                self.debug_area = (
                                    ui.textarea('Debug / Errors', value='Ready')
                                    .props('rows=8')        # ✅ correct way in NiceGUI
                                    .classes('w-full')
                                )


        # Bindings
        self.verbose_switch.on('update:model-value', self._on_verbose_toggle)

    def _on_verbose_toggle(self, _e):
        self.verbose = bool(self.verbose_switch.value)
        self.svc.verbose = self.verbose
        if self.verbose:
            print(f"[APP] Verbose set to {self.verbose}")

    async def on_compute(self):
        # validate dates
        start = (self.start_input.value or '').strip()
        end   = (self.end_input.value or '').strip()
        try:
            datetime.strptime(start, '%Y-%m-%d')
            datetime.strptime(end, '%Y-%m-%d')
        except Exception:
            ui.notify('Invalid date(s). Use YYYY-MM-DD.', color='negative')
            return

        asset = (self.asset_input.value or '').strip().upper() or None

        self.debug_area.value = "Running..."
        try:
            # run compute in a thread (since HTTP inside is blocking)
            totals = await asyncio.to_thread(self.svc.compute, start, end, asset)
        except Exception as e:
            self.debug_area.value = f"❌ {e}"
            return

        # Update labels
        self.lbl_window.text = f"Window: {totals.get('window_ist','—')}"
        self.lbl_asset.text  = f"Asset: {totals.get('asset','—')}"
        self.lbl_count.text  = f"Txn Count: {totals.get('count','—')}"
        self.lbl_comm.text   = f"Total Commission: {totals.get('total_commission', 0.0):.6f}"
        self.lbl_fund.text   = f"Total Funding: {totals.get('total_funding', 0.0):.6f}"
        self.lbl_cash.text   = f"Total Cashflow: {totals.get('total_cashflow', 0.0):.6f}"
        self.lbl_dep.text    = f"Total Deposit: {totals.get('total_deposit', 0.0):.6f}"
        self.debug_area.value = "✅ Done"


# Run app
App()
ui.run(title='Delta Wallet Totals', favicon='📊', reload=True)
