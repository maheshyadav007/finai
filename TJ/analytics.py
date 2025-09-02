"""
analytics.py
-------------

This module contains functions to compute performance metrics from a list
of Trade objects. Splitting analytics into its own module makes it easy
to reuse these functions in different contexts (command‑line tool,
Flask app, future API) without coupling them to UI or storage concerns.
"""

from typing import List, Dict, Any

from .models import Trade


def compute_metrics(trades: List[Trade]) -> Dict[str, Any]:
    """Compute performance statistics for the given trades.

    Parameters
    ----------
    trades: List[Trade]
        List of Trade instances for which to compute metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed metrics. Keys include:
        - total_trades: int
        - total_pnl: float
        - average_pnl: float
        - win_rate: float (percentage)
        - average_win: float
        - average_loss: float
        - largest_win: float
        - largest_loss: float
        - profit_factor: float
        - expectancy: float
    """
    metrics = {
        "total_trades": 0,
        "total_pnl": 0.0,
        "average_pnl": 0.0,
        "win_rate": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
    }
    if not trades:
        return metrics

    pnls = [trade.pnl for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    total_trades = len(trades)
    total_pnl = sum(pnls)
    average_pnl = total_pnl / total_trades
    win_rate = len(wins) / total_trades * 100
    average_win = sum(wins) / len(wins) if wins else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    largest_win = max(wins) if wins else 0.0
    largest_loss = min(losses) if losses else 0.0
    total_wins = sum(wins)
    total_losses = -sum(losses)  # sum of absolute loss values
    profit_factor = total_wins / total_losses if total_losses != 0 else 0.0
    loss_rate = (len(losses) / total_trades) * 100
    # expectancy per trade (in the units of PnL) = expected value of win/loss distribution
    expectancy = (win_rate / 100 * average_win) - (loss_rate / 100 * (-average_loss))

    metrics.update(
        {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "average_pnl": average_pnl,
            "win_rate": win_rate,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
        }
    )
    return metrics