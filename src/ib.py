import os
import random
import subprocess
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import fmpsdk
from dotenv import load_dotenv
from ib_insync import IB, MarketOrder, Stock

from src.printlog import PrintLogNone

# Load environment variables from .env file
# This ensures sensitive information like API keys is securely loaded into the environment
load_dotenv()
FMP_APIKEY = os.getenv("FMP_APIKEY")

# Global IB instance
ib = IB()


def ib_connect_reset(host="127.0.0.1", port=4002, timeout=30):
    global ib
    ib = IB()
    clientId = random.randint(1, 10000)
    ib.connect(host, port, clientId=clientId, timeout=timeout)
    # Abonne un logger d'erreurs pour voir les codes IB
    ib.errorEvent += lambda reqId, code, msg, advanced: print(
        f"[IB ERROR] reqId={reqId} code={code} msg={msg} advanced={advanced}"
    )
    return ib


def ib_connect_once(host="127.0.0.1", port=4002, timeout=30):
    global ib
    if not ib.isConnected():
        ib.disconnect()
        ib = IB()
        clientId = random.randint(1, 10000)
        ib.connect(host, port, clientId=clientId, timeout=timeout)
        # Abonne un logger d'erreurs pour voir les codes IB
        ib.errorEvent += lambda reqId, code, msg, advanced: print(
            f"[IB ERROR] reqId={reqId} code={code} msg={msg} advanced={advanced}"
        )
    return ib


def ib_disconnect():
    global ib
    ib.disconnect()
    ib = IB()


def _parse_ib_hours(hours_str: str, tz: ZoneInfo):
    """
    Parse les chaînes IB d'horaires (tradingHours/liquidHours) en intervalles timezone-aware.
    Gère :
      - 'YYYYMMDD:HHMM-HHMM'
      - 'YYYYMMDD:HHMM-YYYYMMDD:HHMM'
      - plusieurs sessions par jour séparées par des virgules
      - 'CLOSED'
    Retourne: liste triée [(start_dt, end_dt), ...] en fuseau `tz`.
    """

    def parse_token(token: str, default_date: str):
        # token peut être 'HHMM' ou 'YYYYMMDD:HHMM'
        token = token.strip()
        if ":" in token:
            dpart, tpart = token.split(":", 1)
        else:
            dpart, tpart = default_date, token
        if len(dpart) != 8 or len(tpart) != 4:
            raise ValueError(f"Format IB inattendu: {token!r}")
        y, m, d = int(dpart[:4]), int(dpart[4:6]), int(dpart[6:8])
        h, mi = int(tpart[:2]), int(tpart[2:])
        return datetime(y, m, d, h, mi, tzinfo=tz)

    intervals = []
    if not hours_str:
        return intervals

    for day_block in hours_str.split(";"):
        day_block = day_block.strip()
        if not day_block:
            continue
        # Exemple de block:
        # '20251009:0400-20251009:2000'  ou  '20251009:0930-1600'  ou  '20251011:CLOSED'
        if ":" not in day_block:
            continue  # format inattendu, on ignore prudemment

        date_part, times_part = day_block.split(":", 1)
        date_part = date_part.strip()
        times_part = times_part.strip()

        if times_part.upper() == "CLOSED":
            continue

        for span in times_part.split(","):
            span = span.strip()
            if not span:
                continue
            try:
                start_tok, end_tok = span.split("-", 1)
            except ValueError:
                raise ValueError(f"Plage horaire IB invalide: {span!r}")

            start_dt = parse_token(start_tok, date_part)
            end_dt = parse_token(end_tok, date_part)

            # IB ne traverse normalement pas minuit sur actions US, mais on filtre par sécurité
            if end_dt > start_dt:
                intervals.append((start_dt, end_dt))

    return sorted(intervals, key=lambda x: x[0])


def market_status(
    ib: IB, mode: str = "rth", symbol: str = "AAPL", primaryExchange: str = "NASDAQ"
):
    """
    Retourne ouverture/fermeture en ET et en heure de Paris pour le marché US du symbole.
    mode = 'rth' (9:30-16:00 ET) | 'extended' (liquidHours: pré/after-market inclus)
    """
    c = Stock(symbol, "SMART", "USD", primaryExchange=primaryExchange)
    c = ib.qualifyContracts(c)[0]
    cd = ib.reqContractDetails(c)[0]

    tzid = cd.timeZoneId or "America/New_York"
    try:
        et = ZoneInfo(tzid)
    except Exception:
        et = ZoneInfo("America/New_York")
    paris = ZoneInfo("Europe/Paris")

    now_utc = ib.reqCurrentTime()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    now_et = now_utc.astimezone(et)

    raw = cd.tradingHours if mode.lower() == "rth" else cd.liquidHours
    intervals = _parse_ib_hours(raw, et)

    open_now = False
    current_open = current_close = None
    next_open = next_close = None

    for start, end in intervals:
        if start <= now_et < end:
            open_now = True
            current_open, current_close = start, end
            break
        if now_et < start and next_open is None:
            next_open, next_close = start, end
            # on peut break car la liste est triée
            break

    to_paris = lambda dt: (dt.astimezone(paris) if dt else None)

    now_paris = now_et.astimezone(paris)

    return {
        "mode": "RTH" if mode.lower() == "rth" else "Extended",
        "symbolUsed": c.symbol,
        "exchangeTz": tzid,
        "nowET": now_et,
        "nowParis": now_paris,
        "openNow": open_now,
        "currentOpenET": current_open,
        "currentCloseET": current_close,
        "currentOpenParis": to_paris(current_open),
        "currentCloseParis": to_paris(current_close),
        "nextOpenET": next_open,
        "nextCloseET": next_close,
        "nextOpenParis": to_paris(next_open),
        "nextCloseParis": to_paris(next_close),
        "rawHoursString": raw,
    }


def get_capital(ib: IB):
    netLiquidation = None
    totalCashValue = None
    for v in ib.accountValues():
        if v.tag == "NetLiquidationByCurrency" and v.currency == "USD":
            netLiquidation = float(v.value)

    for v in ib.accountValues():
        if v.tag == "TotalCashBalance" and v.currency == "USD":
            totalCashValue = float(v.value)

    data = {
        "currency": "USD",
        "totalCashValue": totalCashValue,
        "netLiquidation": netLiquidation,
    }
    return data


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
        time.sleep(5)
        return True
    except Exception as e:
        with log:
            print(f"IB connection error: {e}")
        return False
