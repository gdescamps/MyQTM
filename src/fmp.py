import json
import typing
from urllib.request import urlopen

import certifi
from fmpsdk.url_methods import __return_json_v3


def get_jsonparsed_data(url):
    """
    Fetch JSON data from a URL and parse it.

    Args:
        url: The URL to fetch data from

    Returns:
        Parsed JSON data as a Python object
    """
    # Open URL with SSL certificate verification
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def fmp_analyst_stock_recommendations(
    apikey: str,
    symbol: str,
) -> typing.Optional[typing.List[typing.Dict]]:
    """
    Fetch analyst stock recommendations from Financial Modeling Prep API.

    Args:
        apikey: API key for FMP
        symbol: Stock ticker symbol

    Returns:
        List of analyst recommendations or None
    """
    path = "analyst-stock-recommendations"
    query_vars = {"apikey": apikey}
    if symbol:
        path = f"{path}/{symbol}"
    return __return_json_v3(path=path, query_vars=query_vars)


def fmp_economic_indicators(
    apikey: str, name: str, from_date: str = None, to_date: str = None
) -> typing.Optional[typing.List[typing.Dict]]:
    """
    Fetch economic indicators from Financial Modeling Prep API.

    Args:
        apikey: API key for FMP
        name: Name of the economic indicator
        from_date: Start date for data range (optional)
        to_date: End date for data range (optional)

    Returns:
        List of economic indicator data or None
    """
    url = f"https://financialmodelingprep.com/stable/economic-indicators?name={name}&apikey={apikey}"
    if from_date:
        url += f"&from={from_date}"
    if to_date:
        url += f"&to={to_date}"
    return get_jsonparsed_data(url)


def fmp_profile(apikey: str, symbol: str):
    """
    Fetch company profile information from Financial Modeling Prep API.

    Args:
        apikey: API key for FMP
        ticker: Stock ticker symbol

    Returns:
        Company profile data
    """
    url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={apikey}"
    return get_jsonparsed_data(url)


def fmp_historical_price_eod(
    apikey: str, symbol: str, from_date: str = None, to_date: str = None
):
    """
    Fetch historical end-of-day price data for a given stock symbol from Financial Modeling Prep API.

    Args:
        apikey (str): API key for FMP.
        symbol (str): Stock ticker symbol.
        from_date (str, optional): Start date for the data range in YYYY-MM-DD format.
        to_date (str, optional): End date for the data range in YYYY-MM-DD format.

    Returns:
        list: A list of dictionaries containing historical price data.
    """
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&apikey={apikey}"
    if from_date:
        url += f"&from={from_date}"
    if to_date:
        url += f"&to={to_date}"
    return get_jsonparsed_data(url)


def fmp_stock_news(
    apikey: str,
    symbol: str,
    from_date: str = None,
    to_date: str = None,
    page: int = None,
    limit: int = None,
):
    """
    Fetch stock news data for a given stock symbol from Financial Modeling Prep API.

    Args:
        apikey (str): API key for FMP.
        symbol (str): Stock ticker symbol.
        from_date (str, optional): Start date for the news range in YYYY-MM-DD format.
        to_date (str, optional): End date for the news range in YYYY-MM-DD format.
        page (int, optional): Page number for paginated results.
        limit (int, optional): Maximum number of news items per page.

    Returns:
        list: A list of dictionaries containing stock news data.
    """
    url = f"https://financialmodelingprep.com/stable/news/stock?symbol={symbol}&apikey={apikey}"
    if from_date:
        url += f"&from={from_date}"
    if to_date:
        url += f"&to={to_date}"
    if page:
        url += f"&page={page}"
    if limit:
        url += f"&limit={limit}"
    return get_jsonparsed_data(url)


def fmp_key_metrics(apikey: str, symbol: str, limit: int = None, period: str = None):
    """
    Fetch key financial metrics for a given stock symbol from Financial Modeling Prep API.

    Args:
        apikey (str): API key for FMP.
        symbol (str): Stock ticker symbol.
        limit (int, optional): Maximum number of records to retrieve.
        period (str, optional): Time period for the metrics (e.g., "quarter", "annual").

    Returns:
        list: A list of dictionaries containing key financial metrics.
    """
    url = f"https://financialmodelingprep.com/stable/ratios?symbol={symbol}&apikey={apikey}"
    if limit:
        url += f"&limit={limit}"
    if period:
        url += f"&period={period}"
    return get_jsonparsed_data(url)


def fmp_historical_rating(apikey: str, symbol: str, limit: int = None):
    """
    Fetch historical stock ratings for a given stock symbol from Financial Modeling Prep API.

    Args:
        apikey (str): API key for FMP.
        symbol (str): Stock ticker symbol.
        limit (int, optional): Maximum number of records to retrieve.

    Returns:
        list: A list of dictionaries containing historical stock ratings.
    """
    url = f"https://financialmodelingprep.com/stable/ratings-historical?symbol={symbol}&apikey={apikey}"
    if limit:
        url += f"&limit={limit}"
    return get_jsonparsed_data(url)
