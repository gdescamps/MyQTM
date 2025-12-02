"""
trade.py

This module contains functions for building trade data, managing positions, and computing trading parameters. It is used to simulate trading strategies and manage portfolio operations.

Functions:
- build_trade_data(): Constructs trade data from benchmark data and models.
- close_positions(): Closes long and short positions and updates capital and history.
- compute_position_sizes(): Computes total position sizes and updates position info.
- open_positions(): Opens new positions and updates capital and positions list.
- select_positions_to_open(): Selects tickers to open new positions based on criteria.
- select_positions_to_close(): Selects positions to close based on thresholds and criteria.
- get_param(): Retrieves trading parameters based on the current date and interval type.

"""

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import src.config as config
from src.interval import get_interval_type
from src.printlog import PrintLogNone


def build_trade_data(
    model_path,
    data_path,
    file_date_str,
    start_date,
    end_date,
    end_limit: bool = True,
):
    """
    Constructs trade data from benchmark data and models.

    Args:
        model_path (Path): Path to the directory containing the models.
        data_path (Path): Path to the directory containing the benchmark data.
        file_date_str (str): Date string for the benchmark file.
        start_date (datetime): Start date for the trade data.
        end_date (datetime): End date for the trade data.
        end_limit (bool, optional): Whether to enforce end limits. Defaults to True.

    Returns:
        dict: Trade data organized by date and stock.
    """
    current_date = start_date
    trade_data = {}

    selected_featuresA_path = model_path / "selected_featuresA.json"
    with open(selected_featuresA_path, "r") as f:
        selected_featuresA = json.load(f)
    selected_featuresB_path = model_path / "selected_featuresB.json"
    with open(selected_featuresB_path, "r") as f:
        selected_featuresB = json.load(f)

    df_bench = pd.read_csv(data_path / f"{file_date_str}_benchmark_XY.csv")
    df_bench["date"] = pd.to_datetime(df_bench["date"])

    df_bench_copy = df_bench.copy()
    df_benchA_X = df_bench_copy[selected_featuresA]
    df_bench_copy = df_bench.copy()
    df_benchB_X = df_bench_copy[selected_featuresB]
    df_bench_Y = df_bench_copy["trend"]
    dbenchA = xgb.DMatrix(df_benchA_X, label=df_bench_Y.values.ravel())
    dbenchB = xgb.DMatrix(df_benchB_X, label=df_bench_Y.values.ravel())

    modelA_path = model_path / "best_modelA.pkl"
    best_modelA = joblib.load(modelA_path)
    ntree_limit_path = model_path / "best_modelA_ntree_limit.json"
    with open(ntree_limit_path, "r") as f:
        ntree_limit_A = json.load(f)

    modelB_path = model_path / "best_modelB.pkl"
    best_modelB = joblib.load(modelB_path)

    ntree_limit_path = model_path / "best_modelB_ntree_limit.json"
    with open(ntree_limit_path, "r") as f:
        ntree_limit_B = json.load(f)

    y_probA = best_modelA.predict(
        dbenchA, iteration_range=(0, ntree_limit_A["ntree_limit"])
    )
    y_probB = best_modelB.predict(
        dbenchB, iteration_range=(0, ntree_limit_B["ntree_limit"])
    )

    stocks = []
    while current_date <= end_date:
        current_date_str = current_date.strftime("%Y-%m-%d")
        # current_date is already a Timestamp
        df_today = df_bench[df_bench["date"] == current_date]
        if len(df_today) == 0:
            current_date += pd.Timedelta(days=1)
            continue
        today = {}
        today_stocks = df_bench["stock_name"][df_today.index]
        today_open = df_bench["open"][df_today.index]
        today_y = df_bench["trend"][df_today.index]

        interval_type = get_interval_type(current_date_str, end_limit=end_limit)
        if "A" in interval_type:
            today_y_prob = y_probB[df_today.index]
        elif "B" in interval_type:
            today_y_prob = y_probA[df_today.index]
        elif "C" in interval_type:
            today_y_prob = y_probB[df_today.index]
        elif "D" in interval_type:
            today_y_prob = y_probA[df_today.index]

        for stock, open_price, yprob, y in zip(
            today_stocks, today_open, today_y_prob, today_y
        ):
            stocks.append(stock)
            today[stock] = {
                "class": int(np.argmax(yprob)),
                "Y": y,
                "ybear": float(yprob[0]),
                "ybull": float(yprob[2]),
                "open": float(open_price),
            }
        trade_data[current_date_str] = today
        current_date += pd.Timedelta(days=1)

    stocks = list(set(stocks))
    for stock in stocks:
        current_date = start_date
        pc = None
        index_long = -1
        index_long_close = -1
        index_short = -1
        index_short_close = -1
        index_prob = 0.0
        while current_date <= end_date:
            current_date_str = current_date.strftime("%Y-%m-%d")
            if current_date_str not in trade_data:
                current_date += pd.Timedelta(days=1)
                continue
            today = trade_data[current_date_str]
            if stock not in today:
                current_date += pd.Timedelta(days=1)
                continue
            today_stock = today[stock]
            c = today_stock["class"]
            class_long = 2
            class_short = 0
            index_prob = 0.0
            if pc is not None and c == class_long and pc != class_long:
                index_long = 0
                index_prob = today_stock["ybull"]
            elif pc is not None and c == class_short and pc != class_short:
                index_short = 0
                index_prob = today_stock["ybear"]
            elif pc is not None and c != class_long and pc == class_long:
                index_long_close = 0
                index_prob = 1.0 - today_stock["ybull"]
            elif pc is not None and c != class_short and pc == class_short:
                index_short_close = 0
                index_prob = 1.0 - today_stock["ybear"]

            today_stock["index_long"] = index_long
            today_stock["index_short"] = index_short
            today_stock["index_long_close"] = index_long_close
            today_stock["index_short_close"] = index_short_close
            today_stock["index_prob"] = index_prob

            pc = c
            if index_long >= 0:
                index_long += 1
            if index_short >= 0:
                index_short += 1
            if index_long_close >= 0:
                index_long_close += 1
            if index_short_close >= 0:
                index_short_close += 1
            current_date += pd.Timedelta(days=1)

    return trade_data


def close_positions(
    positions_long_to_close,
    positions_short_to_close,
    bench_data,
    current_date,
    capital,
    positions_history,
    callback=None,
    leverage=1.0,
    log=PrintLogNone(),
):
    """
    Closes long and short positions and updates capital and history.

    Args:
        positions_long_to_close (list): List of long positions to close.
        positions_short_to_close (list): List of short positions to close.
        bench_data (dict): Benchmark data for the current date.
        current_date (datetime): Current date for closing positions.
        capital (float): Current capital.
        positions_history (list): History of closed positions.
        callback (function, optional): Callback function for additional processing. Defaults to None.
        leverage (float, optional): Leverage factor for positions. Defaults to 1.0.
        log (PrintLog, optional): Logger for printing logs. Defaults to PrintLogNone().

    Returns:
        tuple: Updated capital, positions history, and cleared positions to close.
    """
    # Close long positions
    if len(positions_long_to_close):
        interval = get_interval_type(current_date)
        for item in positions_long_to_close:
            ticker = item["ticker"]
            open_price = item["open_price"]
            leverage_size = item["size"]
            size = leverage_size / leverage
            close_price = bench_data[current_date][ticker]["open"]
            item["close_price"] = close_price
            item["close_interval"] = interval
            if callback is not None:
                callback(item, log=log)
            close_price = item["close_price"]
            gain = (close_price - open_price) / open_price
            item["gain"] = 100 * gain
            capital += (gain * leverage_size) + size
            positions_history.append(item)
        positions_long_to_close = []

    # Close short positions
    if len(positions_short_to_close):
        interval = get_interval_type(current_date)
        for item in positions_short_to_close:
            ticker = item["ticker"]
            open_price = item["open_price"]
            leverage_size = item["size"]
            size = leverage_size / leverage
            close_price = bench_data[current_date][ticker]["open"]
            item["close_price"] = close_price
            item["close_interval"] = interval
            if callback is not None:
                callback(item, log=log)
            close_price = item["close_price"]
            gain = (open_price - close_price) / open_price
            item["gain"] = 100 * gain
            capital += (gain * leverage_size) + size
            positions_history.append(item)
        positions_short_to_close = []
    return capital, positions_history, positions_long_to_close, positions_short_to_close


def compute_position_sizes(positions, bench_data, current_date, leverage=1.0):
    """
    Computes total position sizes and updates position info (gain, end, days).

    Args:
        positions (list): List of current positions.
        bench_data (dict): Benchmark data for the current date.
        current_date (datetime): Current date for computing position sizes.
        leverage (float, optional): Leverage factor for positions. Defaults to 1.0.

    Returns:
        float: Total position sizes.
    """
    position_sizes = 0
    for item in positions:
        ticker = item["ticker"]
        open_price = item["open_price"]
        leverage_size = item["size"]
        size = leverage_size / leverage
        close_price = bench_data[current_date][ticker]["open"]
        if item["type"] == "long":
            gain = (close_price - open_price) / open_price
        else:
            gain = (open_price - close_price) / open_price
        item["gain"] = 100 * gain
        item["end"] = current_date
        position_sizes += (gain * leverage_size) + size
    return position_sizes


def open_positions(
    positions_to_open,
    positions,
    bench_data,
    current_date,
    capital,
    capital_and_position,
    position_size,
    max_positions,
    open_prob_thres,
    pos_type,
    prob_power,
    prob_size_rate,
    callback=None,
    leverage=1.0,
    log=PrintLogNone(),
):
    """
    Opens new positions (long or short) and updates capital and positions list.

    Args:
        positions_to_open (list): List of positions to open.
        positions (list): Current list of positions.
        bench_data (dict): Benchmark data for the current date.
        current_date (datetime): Current date for opening positions.
        capital (float): Current capital.
        capital_and_position (float): Total capital and position value.
        position_size (float): Size of each position.
        max_positions (int): Maximum number of positions allowed.
        open_prob_thres (float): Probability threshold for opening positions.
        pos_type (str): Type of position ('long' or 'short').
        prob_power (float): Power factor for probability adjustment.
        prob_size_rate (float): Rate for adjusting position sizes.
        callback (function, optional): Callback function for additional processing. Defaults to None.
        leverage (float, optional): Leverage factor for positions. Defaults to 1.0.
        log (PrintLog, optional): Logger for printing logs. Defaults to PrintLogNone().

    Returns:
        tuple: Updated positions, capital, and cleared positions to open.
    """
    interval = get_interval_type(current_date)
    if len(positions_to_open):
        for item in positions_to_open:
            if capital > 100.0:
                signal_prob = item["yprob"]
                prob_power = float(abs(prob_power))
                signal_prob = signal_prob ** (50.0 * prob_power)
                size_factor_val = 1 + (
                    prob_size_rate * signal_prob * (max_positions - 1)
                )
                size_factor_val = min(size_factor_val, max_positions)
                size_factor_val = max(size_factor_val, 1)
                size = min(capital, size_factor_val * position_size)
                ticker = item["ticker"]
                if ticker not in bench_data[current_date]:
                    continue
                open_position = {
                    "type": pos_type,
                    "ticker": ticker,
                    "size": leverage * size,
                    "capital_and_position": capital_and_position,
                    "start": current_date,
                    "open_price": bench_data[current_date][ticker]["open"],
                    "yprob": item["yprob"],
                    "open_interval": interval,
                }
                if callback is not None:
                    callback(open_position, log=log)
                positions.append(open_position)
                capital -= size
        positions_to_open = []
    return positions, capital, positions_to_open


def select_positions_to_open(
    item_dict,
    prev_item,
    positions,
    positions_to_open,
    stock_filter,
    class_val,
    open_prob_thres,
    new_open_yprob,
):
    """
    Selects tickers to open new positions (long or short).

    Args:
        item_dict (dict): Dictionary of current items.
        prev_item (dict): Dictionary of previous items.
        positions (list): Current list of positions.
        positions_to_open (list): List of positions to open.
        stock_filter (list): List of stocks to filter.
        class_val (int): Class value for position type (2 for long, 0 for short).
        open_prob_thres (float): Probability threshold for opening positions.
        new_open_yprob (float): New open probability.

    Returns:
        tuple: Updated positions to open and new open probability.
    """
    already_open = []
    for pos in positions:
        already_open.append(pos["ticker"])

    for ticker in item_dict.keys():
        if ticker not in stock_filter:
            continue
        if ticker in already_open:
            continue
        if ticker in prev_item and ticker in item_dict:
            if class_val == 2:
                index = item_dict[ticker]["index_long"]
            elif class_val == 0:
                index = item_dict[ticker]["index_short"]
            if index >= 0 and index <= config.OPEN_DELAY:
                if item_dict[ticker]["index_prob"] >= open_prob_thres:
                    positions_to_open.append(
                        {
                            "ticker": ticker,
                            "yprob": item_dict[ticker]["index_prob"],
                        }
                    )
                    if new_open_yprob < item_dict[ticker]["index_prob"]:
                        new_open_yprob = item_dict[ticker]["index_prob"]
                else:
                    break

    return positions_to_open, new_open_yprob


def select_positions_to_close(
    positions,
    item,
    long_close_prob_thres,
    short_close_prob_thres,
    new_open_ybull,
    new_open_ybear,
    capital,
):
    """
    Selects positions to close (long and short) based on thresholds and criteria.

    Args:
        positions (list): Current list of positions.
        item (dict): Current item data.
        long_close_prob_thres (float): Probability threshold for closing long positions.
        short_close_prob_thres (float): Probability threshold for closing short positions.
        new_open_ybull (float): New open probability for long positions.
        new_open_ybear (float): New open probability for short positions.
        capital (float): Current capital.

    Returns:
        tuple: Positions to close (long and short) and indexes of removed positions.
    """
    positions_long_to_close = []
    positions_short_to_close = []
    remove_pos_indexes = []
    for pos_index, pos in enumerate(positions):
        open_price = pos["open_price"]
        close_price = item[pos["ticker"]]["open"]
        if pos["type"] == "long":
            gain = (close_price - open_price) / open_price
        else:
            gain = (open_price - close_price) / open_price
        gain *= 100

        c = item[pos["ticker"]]["class"]

        if pos["type"] == "long":
            if (
                c == 0
                or item[pos["ticker"]]["ybull"] < long_close_prob_thres
                or (
                    item[pos["ticker"]]["ybull"] < new_open_ybull
                    and capital < 100.0
                    and gain > 20.0
                )
            ):
                if c == 0 or item[pos["ticker"]]["ybull"] < long_close_prob_thres:
                    pos["close_reason"] = "prob"
                    pos["close_yprob"] = item[pos["ticker"]]["ybull"]
                elif (
                    item[pos["ticker"]]["ybull"] < new_open_ybull
                    and capital < 100.0
                    and gain > 20.0
                ):
                    pos["close_reason"] = "new_open"
                    pos["close_yprob"] = item[pos["ticker"]]["ybull"]
                positions_long_to_close.append(pos)
                remove_pos_indexes.append(pos_index)
        elif pos["type"] == "short":
            if (
                c == 2
                or item[pos["ticker"]]["ybear"] < short_close_prob_thres
                or (
                    item[pos["ticker"]]["ybear"] < new_open_ybear
                    and capital < 100.0
                    and gain > 20.0
                )
            ):
                if c == 2 or item[pos["ticker"]]["ybear"] < short_close_prob_thres:
                    pos["close_reason"] = "prob"
                    pos["close_yprob"] = item[pos["ticker"]]["ybear"]
                elif (
                    item[pos["ticker"]]["ybear"] < new_open_ybear
                    and capital < 100.0
                    and gain > 20.0
                ):
                    pos["close_reason"] = "new_open"
                    pos["close_yprob"] = item[pos["ticker"]]["ybear"]
                positions_short_to_close.append(pos)
                remove_pos_indexes.append(pos_index)
    return positions_long_to_close, positions_short_to_close, remove_pos_indexes


def get_param(
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
    end_limit: bool = True,
):
    """
    Retrieves trading parameters based on the current date and interval type.

    Args:
        current_date (datetime): Current date for retrieving parameters.
        long_open_prob_thresa (float): Threshold A for opening long positions.
        long_close_prob_thresa (float): Threshold A for closing long positions.
        short_open_prob_thresa (float): Threshold A for opening short positions.
        short_close_prob_thresa (float): Threshold A for closing short positions.
        long_open_prob_thresb (float): Threshold B for opening long positions.
        long_close_prob_thresb (float): Threshold B for closing long positions.
        short_open_prob_thresb (float): Threshold B for opening short positions.
        short_close_prob_thresb (float): Threshold B for closing short positions.
        long_prob_powera (float): Power factor A for long positions.
        short_prob_powera (float): Power factor A for short positions.
        long_prob_powerb (float): Power factor B for long positions.
        short_prob_powerb (float): Power factor B for short positions.
        end_limit (bool, optional): Whether to enforce end limits. Defaults to True.

    Returns:
        tuple: Trading parameters for the current interval type.
    """
    interval_type = get_interval_type(current_date, end_limit=end_limit)
    if "A" in interval_type:
        long_open_prob_thres = long_open_prob_thresb
        short_open_prob_thres = short_open_prob_thresb
        long_close_prob_thres = long_close_prob_thresb
        short_close_prob_thres = short_close_prob_thresb
        long_prob_power = long_prob_powerb
        short_prob_power = short_prob_powerb
    elif "B" in interval_type:
        long_open_prob_thres = long_open_prob_thresa
        short_open_prob_thres = short_open_prob_thresa
        long_close_prob_thres = long_close_prob_thresa
        short_close_prob_thres = short_close_prob_thresa
        long_prob_power = long_prob_powera
        short_prob_power = short_prob_powera
    elif "C" in interval_type:
        long_open_prob_thres = long_open_prob_thresb
        short_open_prob_thres = short_open_prob_thresb
        long_close_prob_thres = long_close_prob_thresb
        short_close_prob_thres = short_close_prob_thresb
        long_prob_power = long_prob_powerb
        short_prob_power = short_prob_powerb
    elif "D" in interval_type:
        long_open_prob_thres = long_open_prob_thresa
        short_open_prob_thres = short_open_prob_thresa
        long_close_prob_thres = long_close_prob_thresa
        short_close_prob_thres = short_close_prob_thresa
        long_prob_power = long_prob_powera
        short_prob_power = short_prob_powera
    return (
        long_open_prob_thres,
        short_open_prob_thres,
        long_close_prob_thres,
        short_close_prob_thres,
        long_prob_power,
        short_prob_power,
    )
