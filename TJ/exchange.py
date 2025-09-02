"""
exchange.py
-----------

Stub module for future integration with cryptocurrency exchanges or
brokerage APIs. This module provides placeholder functions illustrating
how you might modularize external API interactions separately from your
core business logic (models, database and analytics). Keeping exchange
logic decoupled allows you to easily swap out or support multiple
exchanges without affecting other parts of the application.

To implement exchange integration:
1. Obtain API credentials from your chosen exchange (e.g. Binance, Bybit).
2. Use HTTP libraries such as `requests` or official SDKs provided by
   the exchange to fetch data (trades, account balances, market prices).
3. Parse the response into your internal data model (`Trade`) and store
   in the database using the methods defined in `database.py`.

This file currently contains no real logic; it simply defines an
interface and example stubs to guide future development.
"""

from datetime import datetime
from typing import List

from .models import Trade


def fetch_recent_trades(api_key: str, api_secret: str) -> List[Trade]:
    """Fetch recent trades from the connected exchange.

    Parameters
    ----------
    api_key: str
        The API key for authenticating with the exchange.
    api_secret: str
        The API secret corresponding to the API key.

    Returns
    -------
    List[Trade]
        A list of Trade objects populated from the exchange. For now,
        this function returns an empty list. Implement this function
        according to the exchange's API documentation.
    """
    # TODO: Implement API calls to fetch trades
    # Example stub demonstrating how returned data might be transformed
    # into a list of Trade objects. Remove or replace with actual
    # implementation when connecting to real exchange APIs.
    return []


def fetch_account_balance(api_key: str, api_secret: str) -> dict:
    """Fetch account balance from the connected exchange.

    Placeholder function to demonstrate where balance retrieval logic
    would reside. You might use this to display account equity or risk
    metrics alongside trade performance.

    Parameters
    ----------
    api_key: str
        The API key for authenticating with the exchange.
    api_secret: str
        The API secret corresponding to the API key.

    Returns
    -------
    dict
        A dictionary containing account balance information. Currently
        returns an empty dict; implement as needed.
    """
    # TODO: Implement API call to fetch account balance
    return {}