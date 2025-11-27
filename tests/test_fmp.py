"""Unit tests for Financial Modeling Prep (FMP) API integration.

This module verifies that FMP API calls successfully retrieve stock quote data.
"""

import os

import fmpsdk
from dotenv import load_dotenv

from src.config import TRADE_STOCKS
from src.fmp import (
    fmp_historical_price_eod,
    fmp_key_metrics,
    fmp_profile,
    fmp_ratings,
    fmp_stock_news,
)

load_dotenv()

FMP_APIKEY = os.getenv("FMP_APIKEY")


def test_fmp_quotes():
    """Verify that FMP API returns valid price quotes for all configured symbols.

    Queries the quote endpoint for each trading symbol in TRADE_STOCKS
    and confirms that price data is present in the response.
    """
    for symbol in TRADE_STOCKS:
        # Retrieve quote data from FMP API
        ret = fmpsdk.quote(symbol=symbol, apikey=FMP_APIKEY)
        # Verify that price field is present in response
        assert "price" in ret[0].keys()


def test_fmp_price_historical_eod():
    """Verify that historical end-of-day price data is retrieved and correctly ordered.

    Queries historical AAPL prices from 2018-01-01 to 2019-01-31 and confirms:
    1. Results are sorted in descending date order (most recent first)
    2. First record (index 0) contains January 2019 data
    3. Last record (index -1) contains January 2018 data
    """
    # Retrieve historical end-of-day price data for AAPL
    ret = fmp_historical_price_eod(
        apikey=FMP_APIKEY, symbol="AAPL", from_date="2018-01-01", to_date="2019-01-31"
    )
    # Verify data is sorted with oldest dates at the end
    assert "2018-01" in ret[-1]["date"]
    # Verify data is sorted with newest dates at the beginning
    assert "2019-01" in ret[0]["date"]


def test_fmp_profile():
    """Verify that company profile data is retrieved from FMP API.

    Queries the profile endpoint for AAPL and confirms that the response
    contains the correct symbol identifier.
    """
    # Retrieve company profile information for AAPL
    ret = fmp_profile(apikey=FMP_APIKEY, symbol="AAPL")
    # Verify that the returned profile matches the requested symbol
    assert ret[0]["symbol"] == "AAPL"


def test_fmp_stock_news():
    """
    Verify that stock news data is retrieved and contains valid publication dates.

    Queries the stock news endpoint for AAPL within a specific date range and confirms:
    1. The "publishedDate" field is present in the response.
    2. The earliest and latest news items fall within the specified date range.
    """
    ret = fmp_stock_news(
        apikey=FMP_APIKEY, symbol="AAPL", from_date="2023-01-01", to_date="2023-01-05"
    )
    # Verify that the "publishedDate" field exists in the first news item
    assert "publishedDate" in ret[0]
    # Verify that the dates in the response match the specified range
    assert "2023-01" in ret[-1]["publishedDate"]
    assert "2023-01" in ret[0]["publishedDate"]


def test_fmp_key_metrics():
    """
    Verify that key financial metrics are retrieved for the specified symbol.

    Queries the key metrics endpoint for AAPL with quarterly data and a limit of 30 records.
    Confirms:
    1. The "symbol" field is present in the response.
    2. The "symbol" field matches the requested symbol (AAPL).
    3. The number of records returned matches the specified limit (30).
    """
    ret = fmp_key_metrics(apikey=FMP_APIKEY, symbol="AAPL", period="quarter", limit=30)
    print(ret)
    # Verify that the "symbol" field exists in the first record
    assert "symbol" in ret[0]
    # Verify that the "symbol" field matches the requested symbol
    assert ret[0]["symbol"] == "AAPL"
    # Verify that the number of records matches the specified limit
    assert len(ret) == 30


def test_fmp_ratings():
    """
    Verify that stock ratings data is retrieved for the specified symbol.

    Queries the ratings endpoint for AAPL with a limit of 2000 records.
    Confirms:
    1. The number of records returned matches the specified limit (2000).
    2. The "symbol" field in the first record matches the requested symbol (AAPL).
    """
    ret = fmp_ratings(apikey=FMP_APIKEY, symbol="AAPL", limit=2000)
    # Verify that the number of records matches the specified limit
    assert len(ret) == 2000
    # Verify that the "symbol" field matches the requested symbol
    assert ret[0]["symbol"] == "AAPL"
