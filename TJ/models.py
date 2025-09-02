"""
models.py
---------

Defines the core data model for a trade. A trade represents a single
transaction in the user's trading journal. Keeping this in a separate
module improves modularity and allows the model to be reused by other
components, such as the database layer and API or exchange integrations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """Represents a single trade entry.

    Attributes
    ----------
    id: Optional[int]
        Database primary key (None for new trades before insertion).
    instrument: str
        Symbol or name of the traded instrument (e.g. 'BTCUSDT', 'AAPL').
    date_time: datetime
        Timestamp of when the trade was executed.
    position: str
        Either 'long' or 'short'. Determines how PnL is calculated.
    entry_price: float
        Price at which the position was opened.
    exit_price: float
        Price at which the position was closed.
    quantity: float
        Quantity of the instrument traded. For stock this might be the
        number of shares; for crypto this could be contract size or units.
    notes: str
        User provided notes or comments about the trade.
    """

    id: Optional[int]
    instrument: str
    date_time: datetime
    position: str
    entry_price: float
    exit_price: float
    quantity: float
    notes: str = ""

    @property
    def pnl(self) -> float:
        """Compute profit or loss (PnL) for this trade.

        For long positions, PnL = (exit_price - entry_price) * quantity.
        For short positions, PnL = (entry_price - exit_price) * quantity.
        """
        if self.position.lower() == "long":
            return (self.exit_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.exit_price) * self.quantity