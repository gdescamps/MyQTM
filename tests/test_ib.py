"""Unit tests for Interactive Broker (IB) integration.

This module tests IB connectivity, order placement, position management,
and market status queries. Tests should only run when IB Gateway is running.
"""

from ib_insync import MarketOrder, Stock

from src.config import TRADE_STOCKS
from src.utils.ib import get_capital, ib_connect_once, ib_reboot_docker, market_status


def test_ib_connect():
    """Verify that Interactive Broker connection is established successfully."""
    ib = ib_connect_once()
    assert ib.isConnected()


def test_ib_positions():
    """Verify that current portfolio positions can be retrieved from IB.

    Fetches and displays all open positions including account, symbol, quantity,
    and average cost.
    """
    print("\n")
    ib = ib_connect_once()
    positions = ib.reqPositions()

    # Display each position's details
    for p in positions:
        print(p.account, p.contract.symbol, p.position, p.avgCost)
    assert ib.isConnected()
    assert positions


def test_ib_close_positions():
    """Verify that open positions can be closed via market orders.

    Closes all open positions by:
    1. Long positions (positive quantity): sell at market
    2. Short positions (negative quantity): buy at market
    Waits for each order to complete before proceeding.
    """
    ib = ib_connect_once()
    positions = ib.reqPositions()
    for p in positions:
        print(p.account, p.contract.symbol, p.position, p.avgCost)
        # Close long positions by selling
        if p.position > 0.5:
            trade = ib.placeOrder(
                Stock(p.contract.symbol, "SMART", "USD"),
                MarketOrder("SELL", abs(p.position)),
            )
            while not trade.isDone():
                ib.waitOnUpdate()
            assert trade.isDone()
        # Close short positions by buying
        elif p.position < -0.5:
            trade = ib.placeOrder(
                Stock(p.contract.symbol, "SMART", "USD"),
                MarketOrder("BUY", abs(p.position)),
            )
            while not trade.isDone():
                ib.waitOnUpdate()
            assert trade.isDone()


def test_ib_status():
    """Verify market status query in regular trading hours (RTH) mode.

    Queries whether AAPL is currently open during regular trading hours
    on NASDAQ.
    """
    ib = ib_connect_once()
    assert ib.isConnected()
    rth = market_status(ib, mode="rth", symbol="AAPL", primaryExchange="NASDAQ")
    print(rth)
    assert rth["openNow"]


def test_ib_status_empty_mode():
    """Verify market status query with empty mode (all trading hours).

    Queries whether AAPL is currently open, including extended trading hours.
    """
    ib = ib_connect_once()
    assert ib.isConnected()
    rth = market_status(ib, mode="", symbol="AAPL", primaryExchange="NASDAQ")
    print(rth)
    assert rth["openNow"]


def test_ib_buy():
    """Verify that a market buy order can be placed and executed.

    Places a buy order for 15 shares of AAPL at market price and waits
    for execution.
    """
    ib = ib_connect_once()
    assert ib.isConnected()
    trade = ib.placeOrder(Stock("AAPL", "SMART", "USD"), MarketOrder("BUY", 15))
    while not trade.isDone():
        ib.waitOnUpdate()
    assert trade.isDone()


def test_ib_sell():
    """Verify that a market sell order can be placed and executed.

    Places a sell order for 15 shares of AAPL at market price and waits
    for execution.
    """
    ib = ib_connect_once()
    assert ib.isConnected()
    trade = ib.placeOrder(Stock("AAPL", "SMART", "USD"), MarketOrder("SELL", 15))
    while not trade.isDone():
        ib.waitOnUpdate()
    assert trade.isDone()


def test_ib_capital():
    """Retrieve and display the current account capital information.

    Queries the account for capital metrics including cash, net liquidation value,
    and buying power.
    """
    ib = ib_connect_once()
    assert ib.isConnected()
    ret = get_capital(ib)
    print(ret)


def test_ib_symbols():
    """Verify that all configured trading symbols are valid and resolvable by IB.

    Qualifies each symbol in TRADE_STOCKS to confirm:
    1. Contract exists and is unambiguous in IB database
    2. Contract ID (conId) is valid and positive
    """
    ib = ib_connect_once()
    for symbol in TRADE_STOCKS:
        contract = Stock(
            symbol.strip().upper(), "SMART", "USD", primaryExchange="NASDAQ"
        )
        # Qualify contract to resolve it in IB database
        qualified = ib.qualifyContracts(contract)
        assert qualified, f"{symbol}: contract not found or ambiguous in IB"
        assert (
            qualified[0].conId > 0
        ), f"{symbol}: invalid contract ID after qualification"
        print(symbol)


def test_reboot_ib_docker():
    """Verify that IB Gateway Docker container can be rebooted and reconnected.

    Tests the complete cycle of:
    1. Verifying initial connection
    2. Rebooting the Docker container
    3. Reconnecting to the new container instance
    """
    ib = ib_connect_once()
    assert ib.isConnected()
    # Reboot IB Docker container
    assert ib_reboot_docker()
    # Reconnect after reboot
    ib = ib_connect_once()
    # Verify connection is stable with multiple assertions
    assert ib.isConnected()
    assert ib.isConnected()
    assert ib.isConnected()
    assert ib.isConnected()
