import os

import fmpsdk
from dotenv import load_dotenv
from ib_insync import MarketOrder, Stock

from src.utils.ib import get_capital, ib_connect_once, market_status
from src.utils.printlog import PrintLogNone

# Load environment variables from .env file
# This ensures sensitive information like API keys is securely loaded into the environment
load_dotenv()
FMP_APIKEY = os.getenv("FMP_APIKEY")


# Callbacks for Interactive Broker
def callback_cash_iteractive_broker(log=PrintLogNone()):
    """
    Retrieve total cash value from Interactive Broker account.

    Args:
        log: Logger instance

    Returns:
        Total cash value in USD
    """
    while True:
        try:
            ib = ib_connect_once()
            assert ib.isConnected()
            ret = get_capital(ib)
            if ret["currency"] == "USD":
                with log:
                    print(f"totalCashValue: {ret['totalCashValue']} USD")
                return ret["totalCashValue"]
        except Exception as e:
            with log:
                print(f"Error disconnecting: {e}")
            pass


def callback_net_liquidation_iteractive_broker(log=PrintLogNone()):
    """
    Retrieve net liquidation value from Interactive Broker account.

    Args:
        log: Logger instance

    Returns:
        Net liquidation value in USD
    """
    while True:
        try:
            ib = ib_connect_once()
            assert ib.isConnected()
            ret = get_capital(ib)
            if ret["currency"] == "USD":
                with log:
                    print(f"netLiquidation: {ret['netLiquidation']} USD")
                return ret["netLiquidation"]
        except Exception as e:
            with log:
                print(f"Error disconnecting: {e}")
            pass


def callback_open_positions_iteractive_broker(open_position, log=PrintLogNone()):
    """
    Execute an order to open a new position on Interactive Broker.

    Args:
        open_position: Position details (ticker, type, size, etc.)
        log: Logger instance
    """
    with log:
        print("Opening position:")
        print(f"{open_position}")
    trade = None
    # Get current market price from FMP
    ret = fmpsdk.quote(symbol=open_position["ticker"], apikey=FMP_APIKEY)
    last_quote = float(ret[0]["price"])
    stock_count = int(open_position["size"] / last_quote)
    ib = ib_connect_once()
    with log:
        print("placeOrder:")
    # Place BUY or SELL order based on position type
    if open_position["type"] == "long":
        trade = ib.placeOrder(
            Stock(open_position["ticker"], "SMART", "USD"),
            MarketOrder("BUY", stock_count),
        )
    elif open_position["type"] == "short":
        trade = ib.placeOrder(
            Stock(open_position["ticker"], "SMART", "USD"),
            MarketOrder("SELL", stock_count),
        )
    # Wait for order to complete
    if trade is not None:
        while not trade.isDone():
            with log:
                print("waitOnUpdate...")
            ib.waitOnUpdate()
    # Update position with execution details
    open_position["open_price"] = trade.orderStatus.avgFillPrice
    open_position["stock_count"] = stock_count
    open_position["size"] = stock_count * trade.orderStatus.avgFillPrice
    with log:
        print(str(trade))


def callback_close_positions_iteractive_broker(close_position, log=PrintLogNone()):
    """
    Execute an order to close an existing position on Interactive Broker.

    Args:
        close_position: Position details to close
        log: Logger instance
    """
    with log:
        print("Closing position:")
        print(f"{close_position}")
    ib = ib_connect_once()
    trade = None
    # Qualify contract to ensure proper exchange assignment
    contract = Stock(
        close_position["ticker"].strip().upper(),
        "SMART",
        "USD",
        primaryExchange="NASDAQ",
    )
    ib.qualifyContracts(contract)
    stock_count = close_position["stock_count"]
    with log:
        print("placeOrder:")
    # Place SELL or BUY order to close the position
    if close_position["type"] == "long":
        trade = ib.placeOrder(
            Stock(close_position["ticker"], "SMART", "USD"),
            MarketOrder("SELL", stock_count),
        )
    elif close_position["type"] == "short":
        trade = ib.placeOrder(
            Stock(close_position["ticker"], "SMART", "USD"),
            MarketOrder("BUY", stock_count),
        )
    # Wait for order to complete
    if trade is not None:
        while not trade.isDone():
            with log:
                print("waitOnUpdate...")
            ib.waitOnUpdate()
    # Update position with execution details
    close_position["close_price"] = trade.orderStatus.avgFillPrice
    with log:
        print(str(trade))


def callback_paris_open(log=PrintLogNone()):
    """
    Check if Paris market is open (for European trading session).

    Args:
        log: Logger instance

    Returns:
        Boolean indicating if market is open during regular trading hours
    """
    ib = ib_connect_once()
    rth = market_status(ib, mode="rth", symbol="AAPL", primaryExchange="NASDAQ")
    return rth["openNow"]


def callback_us_open(log=PrintLogNone()):
    """
    Check if US market is open (including pre-market and after-hours).

    Args:
        log: Logger instance

    Returns:
        Boolean indicating if market is open
    """
    ib = ib_connect_once()
    rth = market_status(ib, mode="", symbol="AAPL", primaryExchange="NASDAQ")
    return rth["openNow"]


def callback_ib_connected(log=PrintLogNone()):
    """
    Check if Interactive Broker connection is active and healthy.

    Args:
        log: Logger instance

    Returns:
        Boolean indicating connection status
    """
    try:
        ib = ib_connect_once()
        market_status(ib, mode="", symbol="AAPL", primaryExchange="NASDAQ")
        return True
    except Exception as e:
        with log:
            print(f"IB connection error: {e}")
        return False
