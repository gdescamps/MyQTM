"""Unit tests for robot trading functionality and market availability checks.

This module tests Interactive Broker (IB) connectivity, account metrics,
and market open/close detection for Paris and US stock exchanges.
"""

import datetime

from numpy import isnan

from src.ib import (
    callback_cash_iteractive_broker,
    callback_close_positions_iteractive_broker,
    callback_ib_connected,
    callback_net_liquidation_iteractive_broker,
    callback_open_positions_iteractive_broker,
    callback_paris_open,
    callback_us_open,
    ib_connect_once,
)


def test_robot_ib_connected():
    """Verify that the Interactive Broker connection is active."""
    assert callback_ib_connected()


def test_robot_cash():
    """Verify that cash balance is accessible and contains a valid numeric value."""
    value = callback_cash_iteractive_broker()
    assert value is not None
    assert not isnan(value)


def test_robot_net_liquidation():
    """Verify that net liquidation value is accessible and contains a valid numeric value."""
    value = callback_net_liquidation_iteractive_broker()
    assert value is not None
    assert not isnan(value)


def is_weekday():
    """Determine if the current day is a trading weekday.

    Returns:
        bool: True if today is Monday through Friday (0-4), False otherwise.
    """
    today = datetime.datetime.now().weekday()
    # weekday() returns: 0=Monday, 1=Tuesday, ..., 4=Friday, 5=Saturday, 6=Sunday
    return 0 <= today <= 4


def is_paris_open():
    """Determine if the Paris Stock Exchange is currently open.

    Paris Stock Exchange trading hours are 9:00 to 17:30 CET (9:00 to 22:00 in local timezone).
    The exchange is closed on weekends and public holidays.

    Returns:
        bool: True if current time is within trading hours on a weekday, False otherwise.
    """
    now = datetime.datetime.now()
    weekday = now.weekday()
    # Check if today is a trading day (Monday through Friday)
    if 0 <= weekday <= 4:
        # Paris Stock Exchange trading hours: 9:00 to 22:00 (adjusted for local timezone)
        open_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
        close_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
        return open_time <= now <= close_time
    return False


def is_us_open():
    """Determine if the US stock market is currently open.

    US market trading hours are 9:30 to 16:00 EST, which corresponds to
    15:30 to 22:00 CET (Paris time).
    The exchange is closed on weekends and US public holidays.

    Returns:
        bool: True if current time is within trading hours on a weekday, False otherwise.
    """
    import datetime

    now = datetime.datetime.now()
    weekday = now.weekday()
    # Check if today is a trading day (Monday through Friday)
    if 0 <= weekday <= 4:
        # US market hours expressed in Paris time: 15:30 to 22:00 CET
        open_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        close_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
        return open_time <= now <= close_time
    return False


def test_robot_paris_open():
    """Verify that the Paris market open status matches local calculation.

    Compares the callback result with the local is_weekday() and is_paris_open()
    helper functions to ensure consistency.
    """
    is_open = callback_paris_open()
    if is_weekday() and is_paris_open():
        assert is_open is True
    else:
        assert is_open is False


def test_robot_us_open():
    """Verify that the US market open status matches local calculation.

    Compares the callback result with the local is_weekday() and is_us_open()
    helper functions to ensure consistency.
    """
    is_open = callback_us_open()
    if is_weekday() and is_us_open():
        assert is_open is True
    else:
        assert is_open is False


def test_robot_open_close_positions():
    """Verify that opening and closing positions through IB callbacks work correctly.

    This test:
    1. Checks if the US market is currently open; skips if market is closed
    2. Records AAPL position state before opening a position
    3. Opens an AAPL long position via callback
    4. Closes the AAPL position via callback
    5. Verifies final position state matches initial state

    Test is skipped outside US market trading hours to avoid errors.
    """
    is_open = callback_us_open()
    if is_weekday() and is_us_open():
        assert is_open is True
    else:
        # Skip test if market is closed to avoid unnecessary errors
        assert is_open is False
        return

    # Record AAPL position before any trades
    has_aapl_before = False
    ib = ib_connect_once()
    positions = ib.reqPositions()
    for p in positions:
        if p.contract.symbol == "AAPL" and p.position > 0.5:
            has_aapl_before = True
            break

    # Execute open and close operations
    positions = {"ticker": "AAPL", "type": "long", "size": 1000}
    callback_open_positions_iteractive_broker(positions)
    callback_close_positions_iteractive_broker(positions)

    # Verify AAPL position state matches initial state
    has_aapl_after = False
    positions = ib.reqPositions()
    for p in positions:
        if p.contract.symbol == "AAPL" and p.position > 0.5:
            has_aapl_after = True
            break

    # Position state should be unchanged after opening and closing
    assert has_aapl_before == has_aapl_after
