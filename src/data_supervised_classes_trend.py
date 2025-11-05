import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval
from src.utils.indicators import compute_rsi
from src.utils.round import round_floats
from src.utils.trends import detect_trends


def sum_chunks(lst, n):
    return [int(sum(lst[i : i + n]) / n) for i in range(0, len(lst), n)]


def process_all_stocks(config):

    for stock in tqdm(config.TRADE_STOCKS):

        historical_price_file = (
            config.DATA_DIR
            / f"{stock}_{config.TRADE_END_DATE}_historical_price_full.json"
        )

        with open(historical_price_file, "r") as f:
            historical_price = json.load(f)

        df_historical_price = pd.DataFrame(historical_price).iloc[::-1]
        df_historical_price["date"] = pd.to_datetime(
            df_historical_price["date"], format="%Y-%m-%d"
        )
        df_historical_price.set_index("date", inplace=True)
        # extract trend segments to build supervised classes BULLISH RANGE BEARISH
        trend_segments = detect_trends(
            df_historical_price,
            ma_short=1,
            ma_mid=3,
            ma_long=10,
            min_days=14,
            avg_daily_percent_thr=0.15,
        )
        # Save trend segments to a JSON file
        trends_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_trends.json"
        with open(trends_file, "w") as f:
            json.dump(trend_segments, f, indent=4)

        df = pd.DataFrame(historical_price)
        df["date"] = pd.to_datetime(df["date"])
        # remove unneeded columns
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
        # Put df in ascending order by date (most recent at the top)
        df = df.sort_values("date", ascending=True).reset_index(drop=True)

        # Ajout des colonnes trend et dayx (compteur de jours depuis le début de la tendance)
        trend_map = {}
        trend_index = ["bearish", "range", "bullish"]
        for seg in trend_segments:
            start = pd.to_datetime(seg["start"])
            end = pd.to_datetime(seg["end"])
            for d in pd.date_range(start, end):
                trend_class_index = trend_index.index(
                    seg["trend"]
                )  # Ensure trend is in index
                trend_map[d.date()] = (trend_class_index, start)

        trends = []
        current_range_start = None

        for idx, d in enumerate(df["date"]):
            d_date = d.date()
            if d_date in trend_map:
                trend, start = trend_map[d_date]
                current_range_start = None  # reset range start
            else:
                trend = 1
                if current_range_start is None:
                    current_range_start = d
            trends.append(trend)

        df["trend"] = trends
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

        df["trend"] = df["trend"].astype("Int64")
        df = df.dropna()

        for col in ["close"]:

            max_values = [max(df[col].iloc[i:].tolist()) for i in range(len(df))]
            min_values = [min(df[col].iloc[i:].tolist()) for i in range(len(df))]

            rsi_14 = [
                df[col].iloc[i : min(len(df), i + 15)].tolist() for i in range(len(df))
            ]
            ema_4 = [
                df[col].iloc[i : min(len(df), i + 4)].tolist() for i in range(len(df))
            ]
            ema_8 = [
                df[col].iloc[i : min(len(df), i + 8)].tolist() for i in range(len(df))
            ]
            ema_30 = [
                df[col].iloc[i : min(len(df), i + 30)].tolist() for i in range(len(df))
            ]
            ema_50 = [
                df[col].iloc[i : min(len(df), i + 50)].tolist() for i in range(len(df))
            ]
            ema_200 = [
                df[col].iloc[i : min(len(df), i + 200)].tolist() for i in range(len(df))
            ]

            daily = [
                df[col].iloc[i : min(len(df), i + config.TS_SIZE)].tolist()
                for i in range(len(df))
            ]
            weekly = [
                df[col].iloc[i : min(len(df), i + 7 * config.TS_SIZE)].tolist()[::7]
                for i in range(len(df))
            ]

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
                    * (np.mean(ema_4[i]) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                ema_8[i] = round_floats(
                    100.0
                    * (np.mean(ema_8[i]) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                ema_30[i] = round_floats(
                    100.0
                    * (np.mean(ema_30[i]) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                ema_50[i] = round_floats(
                    100.0
                    * (np.mean(ema_50[i]) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                ema_200[i] = round_floats(
                    100.0
                    * (np.mean(ema_200[i]) - min_value)
                    / (max_value - min_value + 1e-9)
                )
                rsi_14[i] = round_floats(compute_rsi(rsi_14[i]))

            df["rsi_14"] = rsi_14
            df["ema_4"] = ema_4
            df["ema_8"] = ema_8
            df["ema_30"] = ema_30
            df["ema_50"] = ema_50
            df["ema_200"] = ema_200
            df[f"ts_d_{col}"] = daily
            df[f"ts_w_{col}"] = weekly

        for col in ["volume"]:

            max_volumes = [max(df[col].iloc[i:].tolist()) for i in range(len(df))]
            min_volumes = [min(df[col].iloc[i:].tolist()) for i in range(len(df))]

            daily = [
                df[col].iloc[i : min(len(df), i + config.TS_SIZE)].tolist()
                for i in range(len(df))
            ]
            weekly = [
                sum_chunks(
                    df[col].iloc[i : min(len(df), i + 7 * config.TS_SIZE)].tolist(), 7
                )
                for i in range(len(df))
            ]
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

        for col in ["rsi_14", "ema_4", "ema_8", "ema_30", "ema_50", "ema_200"]:
            df[f"ts_{col}"] = [
                df[col].iloc[i : min(len(df), i + config.TS_SIZE + 1)].tolist()
                for i in range(len(df))
            ]

        df = df.drop(
            columns=["high", "low", "close", "volume"],
            errors="ignore",
        )

        for indice in ["VIX", "GCUSD", "IXIC", "CLUSD"]:

            historical_index_price_file = (
                config.DATA_DIR
                / f"{indice}_{config.TRADE_END_DATE}_historical_index_price_full.json"
            )
            if not os.path.exists(historical_index_price_file):
                historical_index_price_file = (
                    config.DATA_DIR
                    / f"{indice}_{config.TRADE_END_DATE}_historical_price_full.json"
                )

            with open(historical_index_price_file, "r") as f:
                historical_price = json.load(f)

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

            # Put df in ascending order by date (most recent at the top)
            df_index = df_index.sort_values("date", ascending=False).reset_index(
                drop=True
            )

            for col in ["close"]:

                max_values = [
                    max(df_index[col].iloc[i:].tolist()) for i in range(len(df_index))
                ]
                min_values = [
                    min(df_index[col].iloc[i:].tolist()) for i in range(len(df_index))
                ]

                ema_50 = [
                    df_index[col].iloc[i : min(len(df_index), i + 50)].tolist()
                    for i in range(len(df_index))
                ]

                ema_200 = [
                    df_index[col].iloc[i : min(len(df_index), i + 200)].tolist()
                    for i in range(len(df_index))
                ]

                daily = [
                    df_index[col]
                    .iloc[i : min(len(df_index), i + config.TS_SIZE)]
                    .tolist()
                    for i in range(len(df_index))
                ]

                weekly = [
                    df_index[col]
                    .iloc[i : min(len(df_index), i + 7 * config.TS_SIZE)]
                    .tolist()[::7]
                    for i in range(len(df_index))
                ]

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

                    ema_50[i] = round_floats(
                        100.0
                        * (np.mean(ema_50[i]) - min_value)
                        / (max_value - min_value + 1e-9)
                    )
                    ema_200[i] = round_floats(
                        100.0
                        * (np.mean(ema_200[i]) - min_value)
                        / (max_value - min_value + 1e-9)
                    )

                df_index[f"{indice.lower()}_ema_50"] = ema_50
                df_index[f"{indice.lower()}_ema_200"] = ema_200
                df_index[f"ts_{indice.lower()}_d_{col}"] = daily
                df_index[f"ts_{indice.lower()}_w_{col}"] = weekly

            df_index = df_index.drop(
                columns=["close"],
                errors="ignore",
            )

            df = df.sort_values("date", ascending=True).reset_index(drop=True)
            df = df.set_index("date")

            df_index = df_index.sort_values("date", ascending=True).reset_index(
                drop=True
            )
            df_index = df_index.set_index("date")

            df = merge_by_interval(df, df_index, "vix_days")

        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        df["month"] = df["date"].dt.month

        # Add commodities and indices data
        df = df.iloc[
            :-200
        ]  # Remove the last 200 rows to remove incomplete time series data

        output_file = (
            config.DATA_DIR
            / f"{stock}_{config.TRADE_END_DATE}_historical_price_full.csv"
        )
        df.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
