import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import src.config as config
from src.utils.path import get_project_root
from src.utils.plot import plot_portfolio_metrics
from src.utils.trade import (
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
    long_open_prob_thresa,
    long_close_prob_thresa,
    short_open_prob_thresa,
    short_close_prob_thresa,
    long_open_prob_thresb,
    long_close_prob_thresb,
    short_open_prob_thresb,
    short_close_prob_thresb,
    long_prob_powera,
    short_prob_powera,
    long_prob_powerb,
    short_prob_powerb,
    leverage=1.0,
):
    position_size = capital / max_positions
    capital_and_position = capital

    current_date = bench_start_date
    dates_portfolio = []
    values_portfolio = []
    capital_portfolio = []
    positions_long_to_open = []
    positions_long_to_close = []
    positions_short_to_open = []
    positions_short_to_close = []
    positions = []
    positions_history = []
    prev_date = None

    sorted_keys = list(sorted(bench_data.keys()))

    for current_date in sorted_keys:

        (
            long_open_prob_thres,
            short_open_prob_thres,
            long_close_prob_thres,
            short_close_prob_thres,
            long_prob_power,
            short_prob_power,
        ) = get_param(
            current_date,
            long_open_prob_thresa,
            long_close_prob_thresa,
            short_open_prob_thresa,
            short_close_prob_thresa,
            long_open_prob_thresb,
            long_close_prob_thresb,
            short_open_prob_thresb,
            short_close_prob_thresb,
            long_prob_powera,
            short_prob_powera,
            long_prob_powerb,
            short_prob_powerb,
        )

        # closes positions
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

        # compute position sizes
        position_sizes = compute_position_sizes(
            positions, bench_data, current_date, leverage
        )
        capital_and_position = position_sizes + capital
        position_size = capital_and_position / max_positions

        # open new positions (long)
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
            long_prob_power,
        )

        # open new positions (short)
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
            short_prob_power,
        )

        # compute position sizes
        position_sizes = compute_position_sizes(
            positions, bench_data, current_date, leverage
        )

        # record portfolio value
        capital_and_position = position_sizes + capital
        position_size = capital_and_position / max_positions
        dates_portfolio.append(pd.to_datetime(current_date, format="%Y-%m-%d"))
        values_portfolio.append(capital_and_position)
        capital_portfolio.append(capital)

        # compute sorted candidates items
        item = bench_data[current_date]
        long_item = item.copy()
        long_item = dict(
            sorted(long_item.items(), key=lambda x: x[1]["index_prob"], reverse=True)
        )

        short_item = item.copy()
        short_item = dict(
            sorted(short_item.items(), key=lambda x: x[1]["index_prob"], reverse=True)
        )

        # handle prev item
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

        for pos in positions:
            stock_filter.append(pos["ticker"])

        stock_filter = list(set(stock_filter))

        # compute long position to open
        new_open_ybull = 0
        positions_long_to_open, new_open_ybull = select_positions_to_open(
            long_item,
            prev_item,
            positions,
            positions_long_to_open,
            stock_filter,
            class_val=2,
            open_prob_thres=long_open_prob_thres,
            new_open_yprob=new_open_ybull,
        )

        # compute short position to open
        new_open_ybear = 0
        positions_short_to_open, new_open_ybear = select_positions_to_open(
            short_item,
            prev_item,
            positions,
            positions_short_to_open,
            stock_filter,
            class_val=0,
            open_prob_thres=short_open_prob_thres,
            new_open_yprob=new_open_ybear,
        )

        # compute positions to close
        positions_long_to_close, positions_short_to_close, remove_pos_indexes = (
            select_positions_to_close(
                positions,
                item,
                long_close_prob_thres,
                short_close_prob_thres,
                new_open_ybull,
                new_open_ybear,
                capital,
            )
        )

        positions = [
            pos for i, pos in enumerate(positions) if i not in remove_pos_indexes
        ]
        prev_date = current_date
    return (
        dates_portfolio,
        values_portfolio,
        capital_portfolio,
        positions,
        positions_history,
        capital_and_position,
    )


def run_benchmark(
    BENCH_START_DATE=None,
    BENCH_END_DATE=None,
    MAX_POSITIONS=config.MAX_POSITIONS,
    INIT_CAPITAL=None,
    LONG_OPEN_PROB_THRESA=0.60,
    LONG_CLOSE_PROB_THRESA=0.37,
    SHORT_OPEN_PROB_THRESA=0.60,
    SHORT_CLOSE_PROB_THRESA=0.37,
    LONG_OPEN_PROB_THRESB=0.60,
    LONG_CLOSE_PROB_THRESB=0.37,
    SHORT_OPEN_PROB_THRESB=0.60,
    SHORT_CLOSE_PROB_THRESB=0.37,
    LONG_PROB_POWERA=1.0,
    SHORT_PROB_POWERA=1.0,
    LONG_PROB_POWERB=1.0,
    SHORT_PROB_POWERB=1.0,
    MODEL_PATH=None,
    remove_stocks=5,
    leverage=1.0,
):
    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    MODEL_PATH = Path(get_project_root()) / MODEL_PATH

    out_file = (
        config.DATA_DIR
        / f"IXIC_{config.TRADE_END_DATE}_historical_index_price_full.json"
    )
    with open(out_file, "r") as f:
        nasdaq_history = json.load(f)

    bench_start_date = pd.to_datetime(BENCH_START_DATE, format="%Y-%m-%d")
    bench_end_date = pd.to_datetime(BENCH_END_DATE, format="%Y-%m-%d")

    trade_data = config.TRADE_DATA_LOAD
    remove_stocks_list = []

    if config.TRADE_DATA_LOAD is None:
        trade_data = build_trade_data(
            model_path=MODEL_PATH,
            bench_start_date=bench_start_date,
            bench_end_date=bench_end_date,
        )
        config.TRADE_DATA_LOAD = trade_data

    (
        dates_portfolio,
        values_portfolio,
        capital_portfolio,
        positions,
        positions_history,
        capital_and_position,
    ) = compute_bench(
        trade_data,
        remove_stocks,
        bench_start_date,
        bench_end_date,
        INIT_CAPITAL,
        MAX_POSITIONS,
        LONG_OPEN_PROB_THRESA,
        LONG_CLOSE_PROB_THRESA,
        SHORT_OPEN_PROB_THRESA,
        SHORT_CLOSE_PROB_THRESA,
        LONG_OPEN_PROB_THRESB,
        LONG_CLOSE_PROB_THRESB,
        SHORT_OPEN_PROB_THRESB,
        SHORT_CLOSE_PROB_THRESB,
        LONG_PROB_POWERA,
        SHORT_PROB_POWERA,
        LONG_PROB_POWERB,
        SHORT_PROB_POWERB,
    )

    # Metrics and NASDAQ comparison
    start_date = dates_portfolio[0]
    end_date = dates_portfolio[-1]
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

    # Calcul Ulcer Index
    # L'Ulcer Index est la racine carrée de la moyenne des drawdowns au carré (en %)
    if len(portfolio_drawdowns) > 0:
        ulcer_index = np.sqrt(np.mean((100 * np.minimum(portfolio_drawdowns, 0)) ** 2))
    else:
        ulcer_index = float("nan")

    # Nasdaq metrics
    if nasdaq_values_filt:
        nasdaq_values_arr = np.array(nasdaq_values_filt)
        nasdaq_cummax = np.maximum.accumulate(nasdaq_values_arr)
        nasdaq_drawdowns = (nasdaq_values_arr - nasdaq_cummax) / nasdaq_cummax
        nasdaq_max_drawdown = nasdaq_drawdowns.min()
        nasdaq_ret = (
            100 * (nasdaq_values_arr[-1] - nasdaq_values_arr[0]) / nasdaq_values_arr[0]
        )
        # Longest period without new high (nasdaq)
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

    def compute_annual_roi(dates_portfolio, values_portfolio):
        """
        Calcule le ROI pour chaque tranche de 1 an à partir de la fin, non glissant.
        Retourne un dict {date_debut: ROI_en_%}
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
            # Cherche la première date <= end_date - 365 jours
            target_date = end_date - pd.Timedelta(days=365)
            prev_year_idx = df[df["date"] <= target_date].index
            if len(prev_year_idx) == 0:
                break  # Pas assez de recul pour une tranche complète
            start_idx = prev_year_idx[-1]
            start_row = df.loc[start_idx]
            start_value = start_row["value"]
            roi = 100 * (end_value - start_value) / start_value
            annual_roi[str(end_row["date"].date())] = float(roi)  # clé = end_date
            i = start_idx  # Passe à la tranche précédente d'un an
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
    longest_portfolio_drawdown_ratio = float(longest_portfolio_drawdown) / float(
        len(dates_portfolio)
    )
    positions_count_rate = float(positions_count) / float(len(dates_portfolio))

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

    def maximize(x, center=0.5, coef=4):
        weight = 1 - coef * (x - center) ** 2
        return max(0.0, weight)

    if (
        float(annual_roi_mean) > 5.0
        and float(100 * ulcer_index) > 50.0
        and longest_portfolio_drawdown > 5
    ):
        perf = (
            maximize(long_rate, center=0.5)
            * maximize(short_rate, center=0.5)
            * maximize(AB_rate, center=0.5)
            * maximize(long_short_rate, center=0.7)
            * float(portfolio_ret)
            / (
                (float(ulcer_index) / 10)
                * float(longest_portfolio_drawdown)
                * (0.25 + (float(annual_roi_std) / 10))
                * (abs(float(100 * portfolio_max_drawdown)) / 30)
            )
        )

    metrics = {
        "portfolio": {
            "perf": perf,
            "dates_portfolio": dates_portfolio,
            "values_portfolio": values_portfolio,
            "return": float(portfolio_ret),
            "max_drawdown": float(100 * portfolio_max_drawdown),
            "ulcer_index": float(100 * ulcer_index),
            "random_stocks_removed": int(remove_stocks),
            "longest_drawdown_period": int(longest_portfolio_drawdown),
            # ...autres métriques éventuelles...
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
        "nasdaq": {
            "dates_portfolio": nasdaq_dates_filt,
            "values_portfolio": nasdaq_values_filt,
            "return": float(nasdaq_ret),
            "max_drawdown": float(100 * nasdaq_max_drawdown),
            "longest_drawdown_period": int(longest_nasdaq_drawdown),
        },
        "positions_count": positions_count,
    }

    # Plotting to in-memory image
    image, _ = plot_portfolio_metrics(metrics)

    # Return metrics, image, and positions
    return metrics, image, positions_history + positions, remove_stocks_list
