import json
import time
from pathlib import Path

import fmpsdk
from tqdm import tqdm

import src.config as config
from src.fmp import (
    fmp_analyst_stock_recommendations,
    fmp_economic_indicators,
    fmp_historical_price_eod,
    fmp_historical_rating,
    fmp_key_metrics,
    fmp_profile,
    fmp_stock_news,
)
from src.utils.path import get_project_root


def download_analyst_stock_recommendations(
    trade_stocks=None, trade_end_date=None, apikey=None, data_path=None
):
    trade_stocks = trade_stocks or config.TRADE_STOCKS
    trade_end_date = trade_end_date or config.TRADE_END_DATE
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    for stock in tqdm(trade_stocks):
        out_file = (
            data_path / f"{stock}_{trade_end_date}_analyst_stock_recommendations.json"
        )
        if not out_file.exists():
            ret = fmp_analyst_stock_recommendations(apikey=apikey, symbol=stock)
            with open(out_file, "w") as f:
                json.dump(ret, f, indent=2)


def download_historical_price(
    trade_stocks=None,
    trade_start_date=None,
    trade_end_date=None,
    apikey=None,
    data_path=None,
):
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    for stock in tqdm(trade_stocks):
        out_file = data_path / f"{stock}_{trade_end_date}_historical_price_full.json"
        if not out_file.exists():
            ret = fmp_historical_price_eod(
                apikey=apikey,
                symbol=stock,
                from_date=trade_start_date,
                to_date=trade_end_date,
            )
            with open(out_file, "w") as f:
                json.dump(ret, f, indent=2)


def download_stock_news(
    trade_stocks=None,
    trade_start_date=None,
    trade_end_date=None,
    apikey=None,
    data_path=None,
):
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    for stock in tqdm(trade_stocks):
        out_file = data_path / f"{stock}_{trade_end_date}_stock_news.json"
        if not out_file.exists():
            news = []
            page_index = 0
            while True:
                try:
                    ret = fmp_stock_news(
                        apikey=apikey,
                        from_date=trade_start_date,
                        to_date=trade_end_date,
                        symbol=stock,
                        limit=1000,
                        page=page_index,
                    )
                except Exception as e:
                    # print(f"Error: {e}")
                    # print("Retrying...")
                    # time.sleep(2)
                    break
                if len(ret) == 0:
                    break
                page_index += 1
                news.extend(ret)
            with open(out_file, "w") as f:
                json.dump(news, f, indent=2)


def download_key_metrics(
    trade_stocks=None, trade_end_date=None, apikey=None, data_path=None
):
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    for stock in tqdm(trade_stocks):
        out_file = data_path / f"{stock}_{trade_end_date}_key_metrics.json"
        if not out_file.exists():
            key_metrics = []
            try:
                ret = fmp_key_metrics(
                    apikey=apikey, symbol=stock, period="quarter", limit=30
                )
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                time.sleep(2)
                continue
            key_metrics.extend(ret)
            with open(out_file, "w") as f:
                json.dump(key_metrics, f, indent=2)


def download_ratings(
    trade_stocks=None, trade_end_date=None, apikey=None, data_path=None
):
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    for stock in tqdm(trade_stocks):
        out_file = data_path / f"{stock}_{trade_end_date}_ratings.json"
        if not out_file.exists():
            key_metrics = []
            try:
                ret = fmp_historical_rating(apikey=apikey, symbol=stock, limit=2000)
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                time.sleep(2)
                continue
            key_metrics.extend(ret)
            with open(out_file, "w") as f:
                json.dump(key_metrics, f, indent=2)


def download_economic_indicators(
    trade_start_date=None, trade_end_date=None, apikey=None, data_path=None
):
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    indicator_list = [
        "GDP",
        "realGDP",
        "nominalPotentialGDP",
        "realGDPPerCapita",
        "federalFunds",
        "CPI",
        "inflationRate",
        "inflation",
        "retailSales",
        "consumerSentiment",
        "durableGoods",
        "unemploymentRate",
        "totalNonfarmPayroll",
        "initialClaims",
        "industrialProductionTotalIndex",
        "newPrivatelyOwnedHousingUnitsStartedTotalUnits",
        "totalVehicleSales",
        "retailMoneyFunds",
        "smoothedUSRecessionProbabilities",
        "3MonthOr90DayRatesAndYieldsCertificatesOfDeposit",
        "commercialBankInterestRateOnCreditCardPlansAllAccounts",
        "30YearFixedRateMortgageAverage",
        "15YearFixedRateMortgageAverage",
    ]
    economic_indicators = {}
    out_file = data_path / f"{trade_end_date}_economic_indicators.json"
    if not out_file.exists():
        for indicator in tqdm(indicator_list):
            ret = fmp_economic_indicators(
                apikey=apikey,
                name=indicator,
                from_date=trade_start_date,
                to_date=trade_end_date,
            )
            for r in ret:
                if r["date"] not in economic_indicators:
                    economic_indicators[r["date"]] = {}
                    economic_indicators[r["date"]][r["name"]] = r["value"]
                else:
                    economic_indicators[r["date"]][r["name"]] = r["value"]
        economic_indicators = dict(
            sorted(economic_indicators.items(), key=lambda x: x[0], reverse=True)
        )
        indicator_fields = indicator_list
        dates_sorted = sorted(economic_indicators.keys())
        last_values = {}
        for date in dates_sorted:
            for field in indicator_fields:
                if field in economic_indicators[date]:
                    last_values[field] = economic_indicators[date][field]
                elif field in last_values:
                    economic_indicators[date][field] = last_values[field]
        with open(out_file, "w") as f:
            json.dump(economic_indicators, f, indent=2)


def download_indices(
    indices=None,
    trade_start_date=None,
    trade_end_date=None,
    apikey=None,
    data_path=None,
):
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    for symbol in tqdm(indices):
        out_file = (
            data_path
            / f"{symbol.replace('^', '')}_{trade_end_date}_historical_index_price_full.json"
        )
        if not out_file.exists():
            try:
                ret = fmp_historical_price_eod(
                    apikey=apikey,
                    symbol=symbol,
                    from_date=trade_start_date,
                    to_date=trade_end_date,
                )
            except Exception as e:
                # Error for {symbol}: {e}, retrying in 2s...
                print(f"Error for {symbol}: {e}, retrying in 2s...")
                time.sleep(2)
            with open(out_file, "w") as f:
                json.dump(ret, f, indent=2)


def download_profiles(
    trade_stocks=None, trade_end_date=None, apikey=None, data_path=None
):
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    for stock in tqdm(trade_stocks):
        out_file = data_path / f"{stock}_{trade_end_date}_profile.json"
        if not out_file.exists():
            ret = fmp_profile(apikey=apikey, symbol=stock)[0]
            keys = list(ret.keys())
            for key in keys:
                if key not in [
                    "currency",
                    "companyName",
                    "industry",
                    "sector",
                    "description",
                    "exchange",
                ]:
                    del ret[key]
            with open(out_file, "w") as f:
                json.dump(ret, f, indent=2)


def main(
    trade_stocks=None,
    trade_start_date=None,
    trade_end_date=None,
    indices=None,
    apikey=None,
    data_path=None,
):
    data_path = data_path or Path(get_project_root()) / "data" / "fmp_data"
    data_path.mkdir(parents=True, exist_ok=True)
    download_analyst_stock_recommendations(
        trade_stocks=trade_stocks,
        trade_end_date=trade_end_date,
        apikey=apikey,
        data_path=data_path,
    )
    download_historical_price(
        trade_stocks=trade_stocks,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
        apikey=apikey,
        data_path=data_path,
    )
    download_stock_news(
        trade_stocks=trade_stocks,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
        apikey=apikey,
        data_path=data_path,
    )
    download_key_metrics(
        trade_stocks=trade_stocks,
        trade_end_date=trade_end_date,
        apikey=apikey,
        data_path=data_path,
    )
    download_ratings(
        trade_stocks=trade_stocks,
        trade_end_date=trade_end_date,
        apikey=apikey,
        data_path=data_path,
    )
    download_economic_indicators(
        trade_start_date=config.TRADE_START_DATE,
        trade_end_date=trade_end_date,
        apikey=apikey,
        data_path=data_path,
    )
    download_indices(
        indices=indices,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
        apikey=apikey,
        data_path=data_path,
    )

    download_profiles(
        trade_stocks=trade_stocks,
        trade_end_date=trade_end_date,
        apikey=apikey,
        data_path=data_path,
    )
