"""
benchmark.py

This module is responsible for running benchmark simulations and computing portfolio metrics. It includes functions for simulating trading strategies, comparing portfolio performance with NASDAQ, and generating performance metrics.

Functions:
- compute_bench(): Simulates the benchmark trading strategy and computes portfolio metrics.
- compute_nasdaq_data(): Computes NASDAQ index data for comparison with portfolio performance.
- compute_annual_roi(): Calculates the ROI for each 1-year period from the end, non-sliding.
- run_benchmark(): Runs the benchmark simulation and computes all portfolio metrics.

Main Execution:
- Loads environment variables and initializes configurations.
- Runs benchmarks for top-performing parameter sets.
- Saves benchmark results and generates performance plots.
"""

import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import src.config as config
from config import (
    CMA_DIR,
    CMA_PARALLEL_PROCESSES,
    CMA_STOCKS_DROP_OUT,
    CMA_STOCKS_DROP_OUT_ROUND,
    INITIAL_CAPITAL,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_DIR,
)
from src.path import get_project_root
from src.plot import plot_portfolio_metrics
from src.printlog import PrintLog
from src.trade import (
    build_trade_data,
    close_positions,
    compute_position_sizes,
    get_param,
    open_positions,
    select_positions_to_close,
    select_positions_to_open,
)


def compute_bench(
    bench_data,
    remove_stocks,
    bench_start_date,
    bench_end_date,
    capital,
    max_positions,
    long_open_prob_thres_a,
    long_close_prob_thres_a,
    short_open_prob_thres_a,
    short_close_prob_thres_a,
    long_open_prob_thres_b,
    long_close_prob_thres_b,
    short_open_prob_thres_b,
    short_close_prob_thres_b,
    increase_positions_count,
):
    """
    Simulates the benchmark trading strategy and computes portfolio metrics.

    Args:
        bench_data (dict): Trading data for the benchmark.
        remove_stocks (int): Number of stocks to remove from the benchmark.
        bench_start_date (datetime): Start date for the benchmark.
        bench_end_date (datetime): End date for the benchmark.
        capital (float): Initial capital for the benchmark.
        max_positions (int): Maximum number of positions allowed.
        long_open_prob_thres_a (float): Threshold A for opening long positions.
        long_close_prob_thres_a (float): Threshold A for closing long positions.
        short_open_prob_thres_a (float): Threshold A for opening short positions.
        short_close_prob_thres_a (float): Threshold A for closing short positions.
        long_open_prob_thres_b (float): Threshold B for opening long positions.
        long_close_prob_thres_b (float): Threshold B for closing long positions.
        short_open_prob_thres_b (float): Threshold B for opening short positions.
        short_close_prob_thres_b (float): Threshold B for closing short positions.
        increase_positions_count (float): Rate for adjusting position sizes.

    Returns:
        tuple: Portfolio values, capital, positions, position history, total capital, and portfolio counts.
    """
    # Simulate the benchmark trading strategy and compute portfolio metrics.
    position_size = capital / max_positions
    capital_and_position = capital

    current_date = bench_start_date
    values_portfolio = []
    capital_portfolio = []
    count_portfolio = []
    positions_long_to_open = []
    positions_long_to_close = []
    positions_short_to_open = []
    positions_short_to_close = []
    positions = []
    positions_history = []
    prev_date = None

    sorted_keys = list(sorted(bench_data.keys()))

    for index, current_date in enumerate(sorted_keys):

        (
            long_open_prob_thres,
            short_open_prob_thres,
            long_close_prob_thres,
            short_close_prob_thres,
        ) = get_param(
            current_date,
            long_open_prob_thres_a,
            long_close_prob_thres_a,
            short_open_prob_thres_a,
            short_close_prob_thres_a,
            long_open_prob_thres_b,
            long_close_prob_thres_b,
            short_open_prob_thres_b,
            short_close_prob_thres_b,
        )

        # Close positions scheduled for closing
        (
            capital,
            positions_history,
            positions_long_to_close,
            positions_short_to_close,
        ) = close_positions(
            positions_long_to_close,
            positions_short_to_close,
            bench_data,
            current_date,
            capital,
            positions_history,
        )

        # Compute the total value of open positions
        position_sizes = compute_position_sizes(positions, bench_data, current_date)
        capital_and_position = position_sizes + capital
        position_size = capital_and_position / max_positions

        # Open new long positions
        positions, capital, positions_long_to_open = open_positions(
            positions_long_to_open,
            positions,
            bench_data,
            current_date,
            capital,
            capital_and_position,
            position_size,
            max_positions,
            "long",
            increase_positions_count,
        )

        # Open new short positions
        positions, capital, positions_short_to_open = open_positions(
            positions_short_to_open,
            positions,
            bench_data,
            current_date,
            capital,
            capital_and_position,
            position_size,
            max_positions,
            "short",
            increase_positions_count,
        )

        # Recompute the total value of open positions
        position_sizes = compute_position_sizes(positions, bench_data, current_date)

        # Record the portfolio value for this date
        capital_and_position = position_sizes + capital
        position_size = capital_and_position / max_positions
        values_portfolio.append(capital_and_position)
        capital_portfolio.append(capital)

        # Compute sorted candidate items for this date
        item = bench_data[current_date]

        count_portfolio.append(len(item))

        long_item = item.copy()
        long_item = dict(
            sorted(long_item.items(), key=lambda x: x[1]["ybull"], reverse=True)
        )

        short_item = item.copy()
        short_item = dict(
            sorted(short_item.items(), key=lambda x: x[1]["ybear"], reverse=True)
        )

        # Handle previous date's data
        if prev_date is not None:
            prev_item = bench_data[prev_date]
            prev_item = prev_item.copy()
        else:
            prev_date = current_date
            continue

        stock_filter = config.TRADE_STOCKS.copy()

        if remove_stocks > 0:
            random.shuffle(stock_filter)
            stock_filter = stock_filter[:-remove_stocks]
            # Ensure stocks with current positions remain in the filter
            for pos in positions:
                if pos["ticker"] not in stock_filter:
                    stock_filter.append(pos["ticker"])

        # Compute long positions to open
        positions_long_to_open = select_positions_to_open(
            long_item,
            prev_item,
            positions,
            positions_long_to_open,
            stock_filter,
            class_val=2,
            open_prob_thres=long_open_prob_thres,
            close_prob_thres=long_close_prob_thres,
        )

        # Compute short positions to open
        positions_short_to_open = select_positions_to_open(
            short_item,
            prev_item,
            positions,
            positions_short_to_open,
            stock_filter,
            class_val=0,
            open_prob_thres=short_open_prob_thres,
            close_prob_thres=short_close_prob_thres,
        )

        # Compute positions to close
        positions_long_to_close, positions_short_to_close, remove_pos_indexes = (
            select_positions_to_close(
                positions, item, long_close_prob_thres, short_close_prob_thres
            )
        )

        # Remove closed positions from the active list
        positions = [
            pos for i, pos in enumerate(positions) if i not in remove_pos_indexes
        ]
        prev_date = current_date

    return (
        values_portfolio,
        capital_portfolio,
        positions,
        positions_history,
        capital_and_position,
        count_portfolio,
    )


def compute_nasdaq_data(
    BENCH_START_DATE=None, BENCH_END_DATE=None, MODEL_PATH=None, data_path=None
):
    """
    Computes NASDAQ index data for comparison with portfolio performance.

    Args:
        BENCH_START_DATE (str, optional): Start date for the benchmark. Defaults to None.
        BENCH_END_DATE (str, optional): End date for the benchmark. Defaults to None.
        MODEL_PATH (str, optional): Path to the model directory. Defaults to None.
        data_path (str, optional): Path to the data directory. Defaults to None.

    Returns:
        dict: NASDAQ performance metrics including return, max drawdown, and longest drawdown period.
    """
    # Compute NASDAQ index data for comparison with portfolio performance

    dates_portfolio = config.DATES_PORTFOLIO

    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    MODEL_PATH = Path(get_project_root()) / MODEL_PATH

    bench_start_date = pd.to_datetime(BENCH_START_DATE, format="%Y-%m-%d")
    bench_end_date = pd.to_datetime(BENCH_END_DATE, format="%Y-%m-%d")

    if config.TRADE_DATA_LOAD is None:
        trade_data = build_trade_data(
            model_path=MODEL_PATH,
            data_path=data_path,
            file_date_str=BENCH_END_DATE,
            start_date=bench_start_date,
            end_date=bench_end_date,
        )
        config.TRADE_DATA_LOAD = trade_data
        config.DATES_PORTFOLIO = []

        sorted_keys = list(sorted(trade_data.keys()))
        for index, current_date in enumerate(sorted_keys):
            config.DATES_PORTFOLIO.append(
                pd.to_datetime(current_date, format="%Y-%m-%d")
            )
        dates_portfolio = config.DATES_PORTFOLIO

    # Metrics and NASDAQ comparison
    start_date = dates_portfolio[0]
    end_date = dates_portfolio[-1]

    if data_path is None:
        data_path = Path(get_project_root()) / "data" / "fmp_data"

    out_file = (
        data_path / f"IXIC_{config.BENCHMARK_END_DATE}_historical_index_price_full.json"
    )
    with open(out_file, "r") as f:
        nasdaq_history = json.load(f)

    if config.BASE_END_DATE_FILE is not None:
        base_historical_price_file = (
            data_path
            / f"IXIC_{config.BASE_END_DATE_FILE}_historical_index_price_full.json"
        )
        with open(base_historical_price_file, "r") as f:
            base_nasdaq_history = json.load(f)
        base_dates = []
        for item in base_nasdaq_history:
            base_dates.append(item["date"])
        add_part = []
        for item in nasdaq_history:
            date = item["date"]
            if date not in base_dates:
                add_part.append(item)
        nasdaq_history = add_part + base_nasdaq_history

    nasdaq_dates = [
        datetime.strptime(r["date"], "%Y-%m-%d").date() for r in nasdaq_history
    ]
    nasdaq_values = [r["close"] for r in nasdaq_history]
    nasdaq_filtered = [
        (d, v)
        for d, v in zip(nasdaq_dates, nasdaq_values)
        if start_date.date() <= d <= end_date.date()
    ]
    if nasdaq_filtered:
        nasdaq_dates_filt, nasdaq_values_filt = zip(*nasdaq_filtered)
    else:
        nasdaq_dates_filt, nasdaq_values_filt = [], []

    nasdaq_dates_filt = list(reversed(nasdaq_dates_filt))
    nasdaq_values_filt = list(reversed(nasdaq_values_filt))

    # Compute NASDAQ metrics
    if nasdaq_values_filt:
        nasdaq_values_arr = np.array(nasdaq_values_filt)
        nasdaq_cummax = np.maximum.accumulate(nasdaq_values_arr)
        nasdaq_drawdowns = (nasdaq_values_arr - nasdaq_cummax) / nasdaq_cummax
        nasdaq_max_drawdown = nasdaq_drawdowns.min()
        nasdaq_ret = (
            100 * (nasdaq_values_arr[-1] - nasdaq_values_arr[0]) / nasdaq_values_arr[0]
        )
        # Longest period without new high (NASDAQ)
        longest_nasdaq_drawdown = 0
        current_dd_nasdaq = 0
        for v, m in zip(nasdaq_values_arr, nasdaq_cummax):
            if v < m:
                current_dd_nasdaq += 1
                if current_dd_nasdaq > longest_nasdaq_drawdown:
                    longest_nasdaq_drawdown = current_dd_nasdaq
            else:
                current_dd_nasdaq = 0
    else:
        nasdaq_max_drawdown = float("nan")
        nasdaq_ret = float("nan")
        longest_nasdaq_drawdown = 0

    return {
        "nasdaq": {
            "dates_portfolio": nasdaq_dates_filt,
            "values_portfolio": nasdaq_values_filt,
            "return": float(nasdaq_ret),
            "max_drawdown": float(100 * nasdaq_max_drawdown),
            "longest_drawdown_period": int(longest_nasdaq_drawdown),
        }
    }


def compute_annual_roi(dates_portfolio, values_portfolio):
    """
    Calculates the ROI for each 1-year period from the end, non-sliding.

    Args:
        dates_portfolio (list): List of portfolio dates.
        values_portfolio (list): List of portfolio values.

    Returns:
        tuple: Annual ROI metrics including mean, standard deviation, min, max, and last ROI.
    """
    """
    Calculates the ROI for each 1-year period from the end, non-sliding.
    Returns a dict {start_date: ROI_in_%}
    """
    df = pd.DataFrame(
        {
            "date": dates_portfolio,
            "value": values_portfolio,
        }
    )
    df = df.sort_values("date").reset_index(drop=True)
    annual_roi = {}
    i = len(df) - 1
    while i > 0:
        end_row = df.iloc[i]
        end_date = end_row["date"]
        end_value = end_row["value"]
        # Search for the first date <= end_date - 365 days
        target_date = end_date - pd.Timedelta(days=365)
        prev_year_idx = df[df["date"] <= target_date].index
        if len(prev_year_idx) == 0:
            break  # Not enough history for a full period
        start_idx = prev_year_idx[-1]
        start_row = df.loc[start_idx]
        start_value = start_row["value"]
        roi = 100 * (end_value - start_value) / start_value
        annual_roi[str(end_row["date"].date())] = float(roi)  # key = end_date
        i = start_idx  # Move to the previous 1-year period
    annual_roi_std = (
        float(np.std(list(annual_roi.values()))) if annual_roi else float("nan")
    )
    annual_roi_mean = (
        float(np.mean(list(annual_roi.values()))) if annual_roi else float("nan")
    )
    annual_roi_min = (
        float(np.min(list(annual_roi.values()))) if annual_roi else float("nan")
    )
    annual_roi_max = (
        float(np.max(list(annual_roi.values()))) if annual_roi else float("nan")
    )

    return (
        annual_roi,
        annual_roi_mean,
        annual_roi_std,
        annual_roi_min,
        annual_roi_max,
        list(annual_roi.values())[0],
    )


def run_benchmark(
    FILE_BENCH_END_DATE=None,
    BENCH_START_DATE=None,
    BENCH_END_DATE=None,
    MAX_POSITIONS=config.MAX_POSITIONS,
    INIT_CAPITAL=None,
    LONG_OPEN_PROB_THRES_A=0.60,
    LONG_CLOSE_PROB_THRES_A=0.37,
    SHORT_OPEN_PROB_THRES_A=0.60,
    SHORT_CLOSE_PROB_THRES_A=0.37,
    LONG_OPEN_PROB_THRES_B=0.60,
    LONG_CLOSE_PROB_THRES_B=0.37,
    SHORT_OPEN_PROB_THRES_B=0.60,
    SHORT_CLOSE_PROB_THRES_B=0.37,
    INCREASE_POSITIONS_COUNT=0.4,
    MODEL_PATH=None,
    data_path=None,
    remove_stocks=5,
    force_reload=False,
):
    """
    Runs the benchmark simulation and computes all portfolio metrics.

    Args:
        FILE_BENCH_END_DATE (str, optional): End date for the benchmark file. Defaults to None.
        BENCH_START_DATE (str, optional): Start date for the benchmark. Defaults to None.
        BENCH_END_DATE (str, optional): End date for the benchmark. Defaults to None.
        MAX_POSITIONS (int, optional): Maximum number of positions allowed. Defaults to config.MAX_POSITIONS.
        INIT_CAPITAL (float, optional): Initial capital for the benchmark. Defaults to None.
        LONG_OPEN_PROB_THRES_A (float, optional): Threshold A for opening long positions. Defaults to 0.60.
        LONG_CLOSE_PROB_THRES_A (float, optional): Threshold A for closing long positions. Defaults to 0.37.
        SHORT_OPEN_PROB_THRES_A (float, optional): Threshold A for opening short positions. Defaults to 0.60.
        SHORT_CLOSE_PROB_THRES_A (float, optional): Threshold A for closing short positions. Defaults to 0.37.
        LONG_OPEN_PROB_THRES_B (float, optional): Threshold B for opening long positions. Defaults to 0.60.
        LONG_CLOSE_PROB_THRES_B (float, optional): Threshold B for closing long positions. Defaults to 0.37.
        SHORT_OPEN_PROB_THRES_B (float, optional): Threshold B for opening short positions. Defaults to 0.60.
        SHORT_CLOSE_PROB_THRES_B (float, optional): Threshold B for closing short positions. Defaults to 0.37.
        INCREASE_POSITIONS_COUNT (float, optional): Rate to increase positions count.
        MODEL_PATH (str, optional): Path to the model directory. Defaults to None.
        data_path (str, optional): Path to the data directory. Defaults to None.
        remove_stocks (int, optional): Number of stocks to remove from the benchmark. Defaults to 5.
        force_reload (bool, optional): Whether to force reload of trade data. Defaults to False.

    Returns:
        tuple: Metrics, positions, and list of removed stocks.
    """
    # Run the benchmark simulation and compute all portfolio metrics
    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    MODEL_PATH = Path(get_project_root()) / MODEL_PATH

    if data_path is None:
        data_path = Path(get_project_root()) / "data" / "fmp_data"

    if FILE_BENCH_END_DATE is None:
        FILE_BENCH_END_DATE = BENCH_END_DATE

    bench_start_date = pd.to_datetime(BENCH_START_DATE, format="%Y-%m-%d")
    bench_end_date = pd.to_datetime(BENCH_END_DATE, format="%Y-%m-%d")

    trade_data = config.TRADE_DATA_LOAD
    dates_portfolio = config.DATES_PORTFOLIO
    remove_stocks_list = []

    if config.TRADE_DATA_LOAD is None or force_reload:
        trade_data = build_trade_data(
            model_path=MODEL_PATH,
            data_path=data_path,
            file_date_str=FILE_BENCH_END_DATE,
            start_date=bench_start_date,
            end_date=bench_end_date,
        )
        config.TRADE_DATA_LOAD = trade_data
        config.DATES_PORTFOLIO = []

        sorted_keys = list(sorted(trade_data.keys()))
        for index, current_date in enumerate(sorted_keys):
            config.DATES_PORTFOLIO.append(
                pd.to_datetime(current_date, format="%Y-%m-%d")
            )
        dates_portfolio = config.DATES_PORTFOLIO
    (
        values_portfolio,
        capital_portfolio,
        positions,
        positions_history,
        capital_and_position,
        count_portfolio,
    ) = compute_bench(
        trade_data,
        remove_stocks,
        bench_start_date,
        bench_end_date,
        INIT_CAPITAL,
        MAX_POSITIONS,
        LONG_OPEN_PROB_THRES_A,
        LONG_CLOSE_PROB_THRES_A,
        SHORT_OPEN_PROB_THRES_A,
        SHORT_CLOSE_PROB_THRES_A,
        LONG_OPEN_PROB_THRES_B,
        LONG_CLOSE_PROB_THRES_B,
        SHORT_OPEN_PROB_THRES_B,
        SHORT_CLOSE_PROB_THRES_B,
        INCREASE_POSITIONS_COUNT,
    )

    # Portfolio metrics
    portfolio_values_arr = np.array(values_portfolio)
    portfolio_cummax = np.maximum.accumulate(portfolio_values_arr)
    portfolio_drawdowns = (portfolio_values_arr - portfolio_cummax) / portfolio_cummax
    portfolio_max_drawdown = portfolio_drawdowns.min()
    portfolio_ret = 100 * (capital_and_position - INIT_CAPITAL) / INIT_CAPITAL

    # Longest period without new high (portfolio)
    longest_portfolio_drawdown = 0
    current_dd = 0
    for v, m in zip(portfolio_values_arr, portfolio_cummax):
        if v < m:
            current_dd += 1
            if current_dd > longest_portfolio_drawdown:
                longest_portfolio_drawdown = current_dd
        else:
            current_dd = 0

    # Ulcer Index calculation
    # The Ulcer Index is the square root of the mean of squared drawdowns (in %)
    if len(portfolio_drawdowns) > 0:
        ulcer_index = np.sqrt(np.mean((100 * np.minimum(portfolio_drawdowns, 0)) ** 2))
    else:
        ulcer_index = float("nan")

    (
        annual_roi,
        annual_roi_mean,
        annual_roi_std,
        annual_roi_min,
        annual_roi_max,
        annual_roi_last,
    ) = compute_annual_roi(dates_portfolio, values_portfolio)

    perf = 0
    positions_count = len(positions_history) + len(positions)

    long_A_positions = len(
        [
            pos
            for pos in positions_history
            if pos["type"] == "long"
            and ("A" in pos["open_interval"] or "C" in pos["open_interval"])
        ]
    )

    long_B_positions = len(
        [
            pos
            for pos in positions_history
            if pos["type"] == "long"
            and ("B" in pos["open_interval"] or "D" in pos["open_interval"])
        ]
    )

    short_A_positions = len(
        [
            pos
            for pos in positions_history
            if pos["type"] == "short"
            and ("A" in pos["open_interval"] or "C" in pos["open_interval"])
        ]
    )

    short_B_positions = len(
        [
            pos
            for pos in positions_history
            if pos["type"] == "short"
            and ("B" in pos["open_interval"] or "D" in pos["open_interval"])
        ]
    )

    long_rate = long_A_positions / (long_A_positions + long_B_positions + 1)
    short_rate = short_A_positions / (short_A_positions + short_B_positions + 1)
    AB_rate = (long_A_positions + short_A_positions) / (positions_count + 1)
    long_short_rate = (long_A_positions + long_B_positions) / (positions_count + 1)

    def gaussian_penalty_weight(x, center=0.5, sigma=0.2):
        # Gaussian penalty weight function for CMA-ES optimization
        # Returns a weight in [0, 1] that penalizes deviations from center
        # sigma controls the width of the Gaussian (smaller = narrower)
        weight = np.exp(-((x - center) ** 2) / (2 * sigma**2))
        return max(0.0, weight)

    if float(annual_roi_mean) > 5.0 and longest_portfolio_drawdown > 5:
        perf = (
            LONG_OPEN_PROB_THRES_A  # favor higher thresholds for better safety
            * LONG_CLOSE_PROB_THRES_A
            * SHORT_OPEN_PROB_THRES_A
            * SHORT_CLOSE_PROB_THRES_A
            * LONG_OPEN_PROB_THRES_B
            * LONG_CLOSE_PROB_THRES_B
            * SHORT_OPEN_PROB_THRES_B
            * SHORT_CLOSE_PROB_THRES_B
            * INCREASE_POSITIONS_COUNT
            * gaussian_penalty_weight(long_rate, center=0.5, sigma=0.2)
            * gaussian_penalty_weight(short_rate, center=0.5, sigma=0.2)
            * gaussian_penalty_weight(AB_rate, center=0.5, sigma=0.2)
            * gaussian_penalty_weight(long_short_rate, center=0.7, sigma=0.2)
            * (float(portfolio_ret) ** 3.0)
            / (
                (1 + (float(longest_portfolio_drawdown) / 100))
                * (0.2 + (float(annual_roi_std) / 10))
                * (1 + (abs(float(portfolio_max_drawdown)) / 10))
            )
        )

    metrics = {
        "portfolio": {
            "perf": perf,
            "dates_portfolio": dates_portfolio,
            "values_portfolio": values_portfolio,
            "count_portfolio": count_portfolio,
            "capital_portfolio": capital_portfolio,
            "return": float(portfolio_ret),
            "max_drawdown": float(100 * portfolio_max_drawdown),
            "ulcer_index": float(ulcer_index),
            "random_stocks_removed": int(remove_stocks),
            "longest_drawdown_period": int(longest_portfolio_drawdown),
            # ...other possible metrics...
            "annual_roi": annual_roi,
            "annual_roi_std": annual_roi_std,
            "annual_roi_mean": annual_roi_mean,
            "annual_roi_min": annual_roi_min,
            "annual_roi_max": annual_roi_max,
            "long_rate": long_rate,
            "short_rate": short_rate,
            "AB_rate": AB_rate,
            "long_short_rate": long_short_rate,
        },
        "positions_count": positions_count,
    }

    # Return metrics, image, and positions
    return metrics, positions_history + positions, remove_stocks_list


if __name__ == "__main__":
    """
    Main execution block for the benchmark pipeline. It performs the following steps:
    1. Loads environment variables and configurations.
    2. Runs benchmarks for top-performing parameter sets.
    3. Saves benchmark results and generates performance plots.
    """

    # Load environment variables from .env file
    # This ensures sensitive information like API keys is securely loaded into the environment.
    load_dotenv()

    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    local_log = PrintLog(extra_name="_benchmark", enable=False)

    for top in range(1, CMA_PARALLEL_PROCESSES):

        if os.path.exists(os.path.join(CMA_DIR, f"top{top}_params.json")):

            with open(os.path.join(CMA_DIR, f"top{top}_params.json"), "r") as f:
                XBEST = json.load(f)

            (
                # max_positions,
                long_open_prob_thres_a,
                long_close_prob_thres_a,
                short_open_prob_thres_a,
                short_close_prob_thres_a,
                long_open_prob_thres_b,
                long_close_prob_thres_b,
                short_open_prob_thres_b,
                short_close_prob_thres_b,
                increase_position_count,
            ) = list(XBEST)

            returns = []
            max_drawdowns = []
            ulcer_indexes = []
            performances = []
            metrics_list = []

            random.seed(42)
            positions = None
            for index in range(CMA_STOCKS_DROP_OUT_ROUND):
                remove_stocks = 0 if index == 0 else CMA_STOCKS_DROP_OUT
                metrics, pos, remove_stocks_list = run_benchmark(
                    FILE_BENCH_END_DATE=TEST_END_DATE,
                    BENCH_START_DATE=TEST_START_DATE,
                    BENCH_END_DATE=TEST_END_DATE,
                    INIT_CAPITAL=INITIAL_CAPITAL,
                    LONG_OPEN_PROB_THRES_A=long_open_prob_thres_a,
                    LONG_CLOSE_PROB_THRES_A=long_close_prob_thres_a,
                    SHORT_OPEN_PROB_THRES_A=short_open_prob_thres_a,
                    SHORT_CLOSE_PROB_THRES_A=short_close_prob_thres_a,
                    LONG_OPEN_PROB_THRES_B=long_open_prob_thres_b,
                    LONG_CLOSE_PROB_THRES_B=long_close_prob_thres_b,
                    SHORT_OPEN_PROB_THRES_B=short_open_prob_thres_b,
                    SHORT_CLOSE_PROB_THRES_B=short_close_prob_thres_b,
                    INCREASE_POSITIONS_COUNT=increase_position_count,
                    MODEL_PATH=TRAIN_DIR,
                    data_path=None,
                    remove_stocks=remove_stocks,
                )
                metrics_list.append(metrics)
                if remove_stocks == 0:
                    positions = pos

            nasdaq_metrics = compute_nasdaq_data(
                BENCH_START_DATE=TEST_START_DATE,
                BENCH_END_DATE=TEST_END_DATE,
                MODEL_PATH=TRAIN_DIR,
                data_path=None,
            )

            json_path = os.path.join(
                local_log.output_dir_time,
                f"top{top}_positions.json",
            )
            with open(json_path, "w") as f:
                json.dump(positions, f, indent=2)

            plot, metrics_text = plot_portfolio_metrics(metrics_list, nasdaq_metrics)
            png_path = os.path.join(
                local_log.output_dir_time,
                f"top{top}_bench.png",
            )
            plot.save(png_path)

            if 5 >= top >= 1:  # Copy best top 5 overall in same place
                shutil.copy(
                    png_path,
                    os.path.join("./outputs", f"top{top}_best.png"),
                )
                shutil.copy(
                    json_path,
                    os.path.join("./outputs", f"top{top}_positions.json"),
                )
                shutil.copy(
                    os.path.join(CMA_DIR, f"top{top}_params.json"),
                    os.path.join("./outputs", f"top{top}_params.json"),
                )

            with local_log:
                print(f"Benchmark results for top{top}_best:")
                print(metrics_text)
    local_log.copy_last()
    local_log.copy_last()
