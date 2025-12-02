import cProfile
import io
import os
import pstats
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.data_download_fmp import main as download_fmp
from src.data_tranform_clean import main as tranform_clean
from src.data_transform_analyst_stock_recommendations_time_series import (
    main as transform_analyst_stock_recommendations_time_series,
)
from src.data_transform_economic_indicators_time_series import (
    main as transform_economic_indicators_time_series,
)
from src.data_transform_key_metrics_time_series import (
    main as transform_key_metrics_time_series,
)
from src.data_transform_price_trends_indicators_time_series import (
    main as transform_price_trends_indicators_time_series,
)
from src.data_transform_ratings_time_series import main as transform_ratings_time_series
from src.data_transform_sentiments_time_series import (
    main as transform_sentiments_time_series,
)
from src.data_transform_split_intervals import main as transform_split_intervals
from src.data_transform_stock_news import main as data_transform_stock_news
from src.data_transform_stock_news_to_sentiment_scores import (
    main as transform_stock_news_to_sentiment_scores,
)
from src.utils.path import get_project_root
from src.utils.printlog import PrintLogNone


def run_pipeline(config=None, log_local=PrintLogNone()):

    if config is None:
        import src.config as config

    with log_local:
        print("download_fmp:")

    trade_start_date = config.TRADE_START_DATE

    if config.BASE_END_DATE_FILE is not None:
        print(
            f"Based data end date is defined and will be used from {config.BASE_END_DATE_FILE}"
        )
        # Parse the base end date
        base_end_dt = datetime.strptime(config.BASE_END_DATE_FILE, "%Y-%m-%d")
        # Subtract 20 calendar days
        dt = base_end_dt - timedelta(days=20)
        trade_start_date = dt.strftime("%Y-%m-%d")

    # Set the API key for Financial Modeling Prep
    # The API key is retrieved from the environment variables.
    FMP_APIKEY = os.getenv("FMP_APIKEY")

    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    data_path = Path(get_project_root()) / "data" / "fmp_data"
    data_path.mkdir(parents=True, exist_ok=True)

    download_fmp(
        trade_stocks=config.TRADE_STOCKS,
        trade_start_date=trade_start_date,
        trade_end_date=config.TRADE_END_DATE,
        indices=config.INDICES,
        apikey=FMP_APIKEY,  # or config.FMP_APIKEY if you want to force the key here
        data_path=data_path,
    )

    if config.ENABLE_PROFILER:
        profiler = cProfile.Profile()
        profiler.enable()
    with log_local:
        print("transform_price_trends_indicators_time_series:")
    transform_price_trends_indicators_time_series(config)

    with log_local:
        print("transform_key_metrics_time_series:")
    transform_key_metrics_time_series(config)

    with log_local:
        print("transform_economic_indicators_time_series:")
    transform_economic_indicators_time_series(config)

    with log_local:
        print("transform_analyst_stock_recommendations_time_series:")
    transform_analyst_stock_recommendations_time_series(config)

    with log_local:
        print("transform_ratings_time_series:")
    transform_ratings_time_series(config)

    with log_local:
        print("transform_stock_news:")
    data_transform_stock_news(config)

    with log_local:
        print("transform_stock_news_to_sentiment_scores:")
    transform_stock_news_to_sentiment_scores(config)

    with log_local:
        print("clean_all:")
    tranform_clean(config)

    with log_local:
        print("sentiments_ts:")
    transform_sentiments_time_series(config)

    with log_local:
        print("test_train_intervals:")
    transform_split_intervals(config)

    if config.ENABLE_PROFILER:
        profiler.disable()
        with log_local:
            stream_buffer = io.StringIO()

            stats = pstats.Stats(profiler, stream=stream_buffer)

            # --- TOP 50 BY CUMULATIVE TIME ---
            print("\n--- TOP 50 FUNCTIONS BY CUMULATIVE TIME for hist_trend ---")
            stats.sort_stats("cumulative")
            # Print the top 50 lines into the buffer
            stats.print_stats(50)

            # Print the buffer content to the console
            sys.stdout.write(stream_buffer.getvalue())

            # --- TOP 50 BY SELF TIME (tottime) ---
            stream_buffer.seek(0)
            stream_buffer.truncate(0)

            stats = pstats.Stats(profiler, stream=stream_buffer)

            print("\n--- TOP 50 FUNCTIONS BY SELF TIME (tottime) for hist_trend ---")
            stats.sort_stats("tottime")
            stats.print_stats(50)

            # Print the buffer content to the console
            sys.stdout.write(stream_buffer.getvalue())
            sys.stdout.write(stream_buffer.getvalue())
