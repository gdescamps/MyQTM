import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval


def process_all_stocks(config):
    for stock in tqdm(config.TRADE_STOCKS):

        analyst_stock_recommendations_file = (
            config.DATA_DIR
            / f"{stock}_{config.TRADE_END_DATE}_analyst_stock_recommendations.json"
        )

        with open(analyst_stock_recommendations_file, "r") as f:
            analyst_stock_recommendations = json.load(f)

        df_analyst_stock_recommendations = pd.DataFrame(analyst_stock_recommendations)
        df_analyst_stock_recommendations["date"] = pd.to_datetime(
            df_analyst_stock_recommendations["date"]
        )
        df_analyst_stock_recommendations = df_analyst_stock_recommendations.drop(
            columns=[
                "symbol",
            ],
            errors="ignore",
        )

        df_analyst_stock_recommendations

        TS_SIZE = 4
        for col in df_analyst_stock_recommendations.columns:
            if col in ["date"]:
                continue
            values = [
                [
                    float(x) if pd.notnull(x) else np.nan
                    for x in df_analyst_stock_recommendations[col].iloc[
                        i : min(len(df_analyst_stock_recommendations), i + TS_SIZE)
                    ]
                ]
                for i in range(len(df_analyst_stock_recommendations))
            ]
            df_analyst_stock_recommendations[f"ts_asr_{col}"] = values
            df_analyst_stock_recommendations = df_analyst_stock_recommendations.drop(
                columns=[col],
                errors="ignore",
            )

        df_analyst_stock_recommendations

        df_analyst_stock_recommendations = df_analyst_stock_recommendations.sort_values(
            "date", ascending=True
        ).reset_index(drop=True)
        df_analyst_stock_recommendations = df_analyst_stock_recommendations.set_index(
            "date"
        )

        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_all = pd.read_csv(output_file)
        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all = df_all.sort_values("date", ascending=True).reset_index(drop=True)
        df_all = df_all.set_index("date")

        merged = merge_by_interval(df_all, df_analyst_stock_recommendations, "asr_days")
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
