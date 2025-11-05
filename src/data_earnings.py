import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval
from src.utils.round import round_floats


def process_all_stocks(config):

    for stock in tqdm(config.TRADE_STOCKS):

        earnings_file = (
            config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_earnings.json"
        )

        with open(earnings_file, "r") as f:
            earnings = json.load(f)

        earnings = round_floats(earnings, precision=3)

        df_earnings = pd.DataFrame(earnings)
        df_earnings["date"] = pd.to_datetime(df_earnings["date"])
        df_earnings = df_earnings.drop(
            columns=["symbol", "estimatedEarning"],
            errors="ignore",
        )

        df_earnings = df_earnings.bfill()

        TS_SIZE = 3
        for col in df_earnings.columns:

            if col in ["date"]:
                continue

            values = [
                [
                    float(x) if pd.notnull(x) else np.nan
                    for x in df_earnings[col].iloc[
                        i : min(len(df_earnings), i + TS_SIZE)
                    ]
                ]
                for i in range(len(df_earnings))
            ]

            df_earnings[f"ts_earnings_{col}"] = values
            df_earnings = df_earnings.drop(
                columns=[col],
                errors="ignore",
            )

        df_earnings = df_earnings.sort_values("date", ascending=True).reset_index(
            drop=True
        )
        df_earnings = df_earnings.set_index("date")

        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_price = pd.read_csv(output_file)
        df_price["date"] = pd.to_datetime(df_price["date"])
        df_price = df_price.sort_values("date", ascending=True).reset_index(drop=True)
        df_price = df_price.set_index("date")

        merged = merge_by_interval(df_price, df_earnings, "earnings_days")
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
