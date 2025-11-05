import json
import typing
from urllib.request import urlopen

import certifi
from fmpsdk.url_methods import __return_json_v3


def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def fmp_grades_news(
    apikey: str,
    symbol: str,
    limit: int = 1,
    page: int = 0,
) -> typing.Optional[typing.List[typing.Dict]]:
    url = f"https://financialmodelingprep.com/stable/grades-news?symbol={symbol}&page={page}&limit={limit}&apikey={apikey}"
    return get_jsonparsed_data(url)


def fmp_analyst_stock_recommendations(
    apikey: str,
    symbol: str,
) -> typing.Optional[typing.List[typing.Dict]]:
    path = "analyst-stock-recommendations"
    query_vars = {"apikey": apikey}
    if symbol:
        path = f"{path}/{symbol}"
    return __return_json_v3(path=path, query_vars=query_vars)


def fmp_economic_indicators(
    apikey: str, name: str, from_date: str = None, to_date: str = None
) -> typing.Optional[typing.List[typing.Dict]]:
    url = f"https://financialmodelingprep.com/stable/economic-indicators?name={name}&apikey={apikey}"
    if from_date:
        url += f"&from={from_date}"
    if to_date:
        url += f"&to={to_date}"
    return get_jsonparsed_data(url)


def fmp_treasury_rates(
    apikey: str, from_date: str, to_date: str
) -> typing.Optional[typing.List[typing.Dict]]:
    url = f"https://financialmodelingprep.com/stable/treasury-rates?from={from_date}&to={to_date}&apikey={apikey}"
    return get_jsonparsed_data(url)


def fmp_historical_price_eod(
    apikey: str, symbol: str, from_date: str, to_date: str
) -> typing.Optional[typing.List[typing.Dict]]:
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&from={from_date}&to={to_date}&apikey={apikey}"
    return get_jsonparsed_data(url)


def fmp_profile(apikey: str, ticker: str):
    url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={apikey}"
    return get_jsonparsed_data(url)
