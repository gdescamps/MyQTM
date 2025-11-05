import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval


def process_all_stocks(config):
    for stock in tqdm(config.TRADE_STOCKS):

        ratings_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_ratings.json"

        with open(ratings_file, "r") as f:
            ratings = json.load(f)

        ratings = pd.DataFrame(ratings)
        ratings["date"] = pd.to_datetime(ratings["date"])
        ratings = ratings.drop(
            columns=[
                "symbol",
            ],
            errors="ignore",
        )

        TS_SIZE = 2
        for col in ratings.columns:
            if col in ["date"]:
                continue
            values = [
                [
                    x if pd.notnull(x) else np.nan
                    for x in ratings[col].iloc[i : min(len(ratings), i + TS_SIZE)]
                ]
                for i in range(len(ratings))
            ]
            ratings[f"ts_r_{col}"] = values
            ratings = ratings.drop(
                columns=[col],
                errors="ignore",
            )

        ratings = ratings.sort_values("date", ascending=True).reset_index(drop=True)
        ratings = ratings.set_index("date")

        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_all = pd.read_csv(output_file)
        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all = df_all.sort_values("date", ascending=True).reset_index(drop=True)
        df_all = df_all.set_index("date")

        merged = merge_by_interval(df_all, ratings, "r_days")
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
