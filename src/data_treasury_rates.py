# %%
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval


def process_all_stocks(config):

    treasury_rates_file = (
        config.DATA_DIR / f"{config.TRADE_END_DATE}_treasury_rates.json"
    )

    with open(treasury_rates_file, "r") as f:
        treasury_rates = json.load(f)

    df_treasury_rates = pd.DataFrame(treasury_rates)

    df_treasury_rates["date"] = pd.to_datetime(df_treasury_rates["date"])
    df_treasury_rates = df_treasury_rates.drop(
        columns=[
            "month2",
            "month3",
            "month6",
            "year2",
            "year2",
            "year3",
            "year7",
            "year10",
            "year20",
            "year30",
        ],
        errors="ignore",
    )

    TS_SIZE = 2
    for col in df_treasury_rates.columns:
        if col in ["date"]:
            continue
        values = [
            [
                float(x) if pd.notnull(x) else np.nan
                for x in df_treasury_rates[col].iloc[
                    i : min(len(df_treasury_rates), i + TS_SIZE)
                ]
            ]
            for i in range(len(df_treasury_rates))
        ]
        df_treasury_rates[f"ts_tr_{col}"] = values
        df_treasury_rates = df_treasury_rates.drop(
            columns=[col],
            errors="ignore",
        )

    df_treasury_rates = df_treasury_rates.sort_values(
        "date", ascending=True
    ).reset_index(drop=True)
    df_treasury_rates = df_treasury_rates.set_index("date")

    for stock in tqdm(config.TRADE_STOCKS):

        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_price = pd.read_csv(output_file)
        df_price["date"] = pd.to_datetime(df_price["date"])
        df_price = df_price.sort_values("date", ascending=True).reset_index(drop=True)
        df_price = df_price.set_index("date")

        merged = merge_by_interval(df_price, df_treasury_rates, None)
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
