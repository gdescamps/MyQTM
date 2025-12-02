import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval
from src.utils.indicators import compute_rsi
from src.utils.path import get_project_root
from src.utils.resample import round_floats
from src.utils.trends import detect_trends


def sum_chunks(lst, n):
    """Calculate average of chunks of size n from a list."""
    return [int(sum(lst[i : i + n]) / n) for i in range(0, len(lst), n)]


def process_all_stocks(config):
    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    df_indexes = {}
    # Precompute index part
    for indice in ["VIX"]:  # Only VIX is used in TOP_FEATURES

        historical_index_price_file = (
            data_path
            / f"{indice}_{config.TRADE_END_DATE}_historical_index_price_full.json"
        )
        if not os.path.exists(historical_index_price_file):
            historical_index_price_file = (
                data_path
                / f"{indice}_{config.TRADE_END_DATE}_historical_price_full.json"
            )

        with open(historical_index_price_file, "r") as f:
            historical_price = json.load(f)

        # If BASE_END_DATE is set, merge base and trade period data
        if config.BASE_END_DATE_FILE is not None:
            base_historical_price_file = (
                data_path
                / f"{indice}_{config.BASE_END_DATE_FILE}_historical_index_price_full.json"
            )
            if not os.path.exists(base_historical_price_file):
                base_historical_price_file = (
                    data_path
                    / f"{indice}_{config.BASE_END_DATE_FILE}_historical_price_full.json"
                )

            with open(base_historical_price_file, "r") as f:
                base_historical_price = json.load(f)
            base_dates = []
            for item in base_historical_price:
                base_dates.append(item["date"])
            after_base_data = []
            for item in historical_price:
                date = item["date"]
                if date not in base_dates:
                    after_base_data.append(item)
            historical_price = after_base_data + base_historical_price

        df_index = pd.DataFrame(historical_price)
        df_index["date"] = pd.to_datetime(df_index["date"])
        df_index = df_index.drop(
            columns=[
                "change",
                "label",
                "changeOverTime",
                "adjClose",
                "unadjustedVolume",
                "vwap",
                "changePercent",
            ],
            errors="ignore",
        )

        # Ensure columns are present and in the desired order
        df_index = df_index[["date", "close"]]

        # Sort dataframe in descending date order (most recent at top)
        df_index = df_index.sort_values("date", ascending=False).reset_index(drop=True)

        for col in ["close"]:

            col_values = df_index[col].values
            n = len(col_values)

            # Extract max and min for normalization from base data using numpy
            max_values = [float(col_values[i:].max()) for i in range(n)]
            min_values = [float(col_values[i:].min()) for i in range(n)]

            ema_50 = [col_values[i : min(n, i + 50)].tolist() for i in range(n)]

            # Calculate weekly time series
            weekly = [
                col_values[i : min(n, i + 7 * config.TS_SIZE)].tolist()[::7]
                for i in range(n)
            ]

            # Normalize values
            for i in range(len(weekly)):
                max_value = max_values[i]
                min_value = min_values[i]

                weekly[i] = [
                    round_floats(
                        100.0 * (v - min_value) / (max_value - min_value + 1e-9)
                    )
                    for v in weekly[i]
                ]

                ema_50[i] = round_floats(
                    100.0
                    * (float(np.mean(ema_50[i])) - min_value)
                    / (max_value - min_value + 1e-9)
                )

            df_index[f"{indice.lower()}_ema_50"] = ema_50
            df_index[f"ts_{indice.lower()}_w_{col}"] = weekly

        df_index = df_index.drop(
            columns=["close"],
            errors="ignore",
        )

        df_index = df_index.sort_values("date", ascending=True).reset_index(drop=True)
        df_index = df_index.set_index("date")

        df_indexes[indice] = df_index

    # Process each stock
    for stock in tqdm(config.TRADE_STOCKS):

        historical_price_file = (
            data_path / f"{stock}_{config.TRADE_END_DATE}_historical_price_full.json"
        )

        with open(historical_price_file, "r") as f:
            historical_price = json.load(f)

        # If BASE_END_DATE is set, merge base and trade period data
        if config.BASE_END_DATE_FILE is not None:

            base_historical_price_file = (
                data_path / f"{stock}_{config.BASE_END_DATE}_historical_price_full.json"
            )

            with open(base_historical_price_file, "r") as f:
                base_historical_price = json.load(f)

            base_dates = []
            for item in base_historical_price:
                base_dates.append(item["date"])

            base_end_date = pd.to_datetime(config.BASE_END_DATE)
            after_base_data = []
            for item in historical_price:
                date = pd.to_datetime(item["date"])
                if date > base_end_date:
                    after_base_data.append(item)

            historical_price = after_base_data + base_historical_price

        # Detect trends
        df_historical_price = pd.DataFrame(historical_price).iloc[::-1]
        df_historical_price["date"] = pd.to_datetime(
            df_historical_price["date"], format="%Y-%m-%d"
        )
        df_historical_price.set_index("date", inplace=True)
        trend_segments = detect_trends(
            df_historical_price,
            ma_short=1,
            ma_mid=3,
            ma_long=10,
            min_days=14,
            avg_daily_percent_thr=0.15,
        )

        # Save trend segments
        trends_file = data_path / f"{stock}_{config.TRADE_END_DATE}_trends.json"

        with open(trends_file, "w") as f:
            json.dump(trend_segments, f, indent=4)

        df = pd.DataFrame(historical_price)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop(
            columns=[
                "change",
                "label",
                "changeOverTime",
                "adjClose",
                "unadjustedVolume",
                "vwap",
                "changePercent",
            ],
            errors="ignore",
        )
        # Ensure columns are present and in the desired order
        df = df[["date", "open", "high", "low", "close", "volume"]]
        # Sort dataframe in ascending date order
        df = df.sort_values("date", ascending=True).reset_index(drop=True)

        # Add trend and dayx columns (day counter since trend start)
        trend_map = {}
        trend_index = ["bearish", "range", "bullish"]
        for seg in trend_segments:
            start = pd.to_datetime(seg["start"])
            end = pd.to_datetime(seg["end"])
            for d in pd.date_range(start, end):
                trend_class_index = trend_index.index(seg["trend"])
                trend_map[d.date()] = (trend_class_index, start)

        trends = []
        current_range_start = None

        # Map trends to dataframe dates
        for idx, d in enumerate(df["date"]):
            d_date = d.date()
            if d_date in trend_map:
                trend, start = trend_map[d_date]
                current_range_start = None  # Reset range start
            else:
                trend = 1
                if current_range_start is None:
                    current_range_start = d
            trends.append(trend)

        df["trend"] = trends
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

        df["trend"] = df["trend"].astype("Int64")
        df = df.dropna()

        # Process close price features
        for col in ["close"]:
            col_values = df[col].values
            n = len(col_values)

            # Extract max and min for normalization from base data using numpy
            max_values = [float(col_values[i:].max()) for i in range(n)]
            min_values = [float(col_values[i:].min()) for i in range(n)]

            # Calculate technical indicators
            rsi_14 = [col_values[i : min(n, i + 15)].tolist() for i in range(n)]
            ema_4 = [col_values[i : min(n, i + 4)].tolist() for i in range(n)]
            ema_8 = [col_values[i : min(n, i + 8)].tolist() for i in range(n)]
            ema_30 = [col_values[i : min(n, i + 30)].tolist() for i in range(n)]
            ema_50 = [col_values[i : min(n, i + 50)].tolist() for i in range(n)]
            ema_200 = [col_values[i : min(n, i + 200)].tolist() for i in range(n)]

            # Calculate daily and weekly time series
            daily = [
                col_values[i : min(n, i + config.TS_SIZE)].tolist() for i in range(n)
            ]

            weekly = [
                col_values[i : min(n, i + 7 * config.TS_SIZE)].tolist()[::7]
                for i in range(n)
            ]

            # Normalize all values
            for i in range(len(daily)):
                max_value = max_values[i]
                min_value = min_values[i]

                daily[i] = [
                    round_floats(
                        100.0 * (v - min_value) / (max_value - min_value + 1e-9)
                    )
                    for v in daily[i]
                ]
                weekly[i] = [
                    round_floats(
                        100.0 * (v - min_value) / (max_value - min_value + 1e-9)
                    )
                    for v in weekly[i]
                ]
                ema_4[i] = round_floats(
                    100.0
                    * (float(np.mean(ema_4[i])) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                ema_8[i] = round_floats(
                    100.0
                    * (float(np.mean(ema_8[i])) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                ema_30[i] = round_floats(
                    100.0
                    * (float(np.mean(ema_30[i])) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                ema_50[i] = round_floats(
                    100.0
                    * (float(np.mean(ema_50[i])) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                ema_200[i] = round_floats(
                    100.0
                    * (float(np.mean(ema_200[i])) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                rsi_14[i] = round_floats(compute_rsi(rsi_14[i]))

            # Add all calculated features to dataframe
            df["rsi_14"] = rsi_14
            df["ema_4"] = ema_4
            df["ema_8"] = ema_8
            df["ema_30"] = ema_30
            df["ema_50"] = ema_50
            df["ema_200"] = ema_200
            df[f"ts_d_{col}"] = daily
            df[f"ts_w_{col}"] = weekly

        # Process volume features
        for col in ["volume"]:
            col_values = df[col].values
            n = len(col_values)

            # Extract max and min for normalization from base data using numpy
            max_volumes = [float(col_values[i:].max()) for i in range(n)]
            min_volumes = [float(col_values[i:].min()) for i in range(n)]

            # Calculate daily and weekly volume time series
            daily = [
                col_values[i : min(n, i + config.TS_SIZE)].tolist() for i in range(n)
            ]
            weekly = [
                sum_chunks(col_values[i : min(n, i + 7 * config.TS_SIZE)].tolist(), 7)
                for i in range(n)
            ]

            # Normalize volume values
            for i in range(len(daily)):
                max_volume = max_volumes[i]
                min_volume = min_volumes[i]

                daily[i] = [
                    round_floats(
                        100.0 * (v - min_volume) / (max_volume - min_volume + 1e-9)
                    )
                    for v in daily[i]
                ]
                weekly[i] = [
                    round_floats(
                        100.0 * (v - min_volume) / (max_volume - min_volume + 1e-9)
                    )
                    for v in weekly[i]
                ]

            df[f"ts_d_{col}"] = daily
            df[f"ts_w_{col}"] = weekly

        # Create time series for each indicator
        for col in ["rsi_14", "ema_4", "ema_8", "ema_30", "ema_50", "ema_200"]:
            col_values = df[col].values
            ts_size = config.TS_SIZE + 1
            n = len(col_values)

            # Preallocate list for time series
            ts_values = []
            for i in range(n):
                end_idx = min(n, i + ts_size)
                ts_values.append(col_values[i:end_idx].tolist())

            df[f"ts_{col}"] = ts_values

        # Drop raw OHLCV columns
        df = df.drop(
            columns=["high", "low", "close", "volume"],
            errors="ignore",
        )

        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        df = df.set_index("date")

        # Merge with index data (VIX)
        for indice in ["VIX"]:
            df_index = df_indexes[indice]
            df = merge_by_interval(df, df_index, "vix_days")

        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        df["month"] = df["date"].dt.month

        # Remove the last 200 rows to exclude incomplete time series data
        df = df.iloc[:-200]

        # Save processed data to CSV
        output_file = (
            data_path / f"{stock}_{config.TRADE_END_DATE}_historical_price_full.csv"
        )
        df.to_csv(output_file, index=False)


def main(config=None):
    """Main entry point for processing all stocks."""
    if config is None:
        import src.config as config
    process_all_stocks(config)
