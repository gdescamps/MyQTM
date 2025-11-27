import os
import random
import subprocess
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ib_insync import IB, Stock

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


def ib_reboot_docker():
    try:
        cwd = os.getcwd()
        print(f"Rebooting IB Docker container in {cwd}")
        ib_disconnect()
        subprocess.run(["docker", "compose", "down"], check=True)
        time.sleep(5)
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
        time.sleep(20)
        ib_connect_reset()
        return True
    except Exception as e:
        print(f"Error rebooting IB Docker container: {e}")
        return False


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
