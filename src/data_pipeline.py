import os

from dotenv import load_dotenv

from src.data_analyst_stock_recommendations import main as analyst_stock_reco
from src.data_build_intervals import main as build_intervals
from src.data_clean_all import main as clean_all
from src.data_download_fmp import main as download_fmp
from src.data_earnings import main as earnings
from src.data_economic_indicators import main as economic_indicators
from src.data_key_metrics import main as key_metrics
from src.data_ratings import main as ratings
from src.data_ratios_features import main as ratios_features
from src.data_score_news_with_LLM import main as score_news_with_LLM
from src.data_sentiments import main as sentiments_ts
from src.data_stock_news_news import main as stock_news
from src.data_supervised_classes_trend import main as supervised_classes_trend
from src.data_treasury_rates import main as treasury_rates
from src.utils.printlog import PrintLogNone

# Load environment variables from .env file
# This ensures sensitive information like API keys is securely loaded into the environment.
load_dotenv()

# Set the API key for Financial Modeling Prep
# The API key is retrieved from the environment variables.
FMP_APIKEY = os.getenv("FMP_APIKEY")


def run_pipeline(config=None, log=PrintLogNone()):
    if config is None:
        import src.config as config
    with log:
        print("download from FMP:")
    trade_start_date = config.TRADE_START_DATE
    download_fmp(
        trade_stocks=config.TRADE_STOCKS,
        trade_start_date=trade_start_date,
        trade_end_date=config.TRADE_END_DATE,
        indices=config.INDICES,
        commodities=config.COMMODITIES,
        apikey=FMP_APIKEY,  # or config.FMP_APIKEY if you want to force the key here
    )
    with log:
        print("done.")
        print("build supervised classes BULLISH RANGE BEARISH:")
    supervised_classes_trend(config)
    with log:
        print("done.")
        print("key metrics:")
    key_metrics(config)
    with log:
        print("done.")
        print("earnings:")
    earnings(config)
    with log:
        print("done.")
        print("economic indicators:")
    economic_indicators(config)
    with log:
        print("done.")
        print("analyst stock:")
    analyst_stock_reco(config)
    with log:
        print("done.")
        print("ratings:")
    ratings(config)
    with log:
        print("done.")
        print("stock news:")
    stock_news(config)
    with log:
        print("done.")
        print("score news with LLM:")
    score_news_with_LLM(config)
    with log:
        print("done.")
        print("clean up:")
    clean_all(config)
    with log:
        print("done.")
        print("treasury rates:")
    treasury_rates(config)
    with log:
        print("done.")
        print("sentiments as time series:")
    sentiments_ts(config)
    with log:
        print("done.")
        print("build intervals:")
    build_intervals(config)
    with log:
        print("done.")
        print("add features ratios:")
    ratios_features(config)
    with log:
        print("done.")


if __name__ == "__main__":
    run_pipeline()
