import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.df import merge_by_interval
from src.path import get_project_root


def process_all_stocks(config):
    # Set the path to the data directory and create it if it doesn't already exist
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    for stock in tqdm(config.TRADE_STOCKS):

        # Load analyst stock recommendations for the current stock
        analyst_stock_recommendations_file = (
            data_path
            / f"{stock}_{config.BENCHMARK_END_DATE}_analyst_stock_recommendations.json"
        )

        with open(analyst_stock_recommendations_file, "r") as f:
            analyst_stock_recommendations = json.load(f)

        # Merge with base historical data if available
        if config.BASE_END_DATE_FILE is not None:

            base_analyst_stock_recommendations_file = (
                data_path
                / f"{stock}_{config.BASE_END_DATE_FILE}_analyst_stock_recommendations.json"
            )

            with open(base_analyst_stock_recommendations_file, "r") as f:
                base_analyst_stock_recommendations = json.load(f)
            base_dates = []
            for item in base_analyst_stock_recommendations:
                base_dates.append(item["date"])

            base_end_date = pd.to_datetime(config.BASE_END_DATE)
            after_base_data = []
            for item in analyst_stock_recommendations:
                date = pd.to_datetime(item["date"])
                if date > base_end_date:
                    after_base_data.append(item)

            analyst_stock_recommendations = (
                after_base_data + base_analyst_stock_recommendations
            )

        # Convert to DataFrame and process
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

        # Create time series features with a window size of 4
        TS_SIZE = 4
        for col in df_analyst_stock_recommendations.columns:
            if col in ["date"]:
                continue
            # Convert each column value to a list of floats representing a time series
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
            # Remove the original column after converting to time series
            df_analyst_stock_recommendations = df_analyst_stock_recommendations.drop(
                columns=[col],
                errors="ignore",
            )

        df_analyst_stock_recommendations

        # Sort by date and set as index
        df_analyst_stock_recommendations = df_analyst_stock_recommendations.sort_values(
            "date", ascending=True
        ).reset_index(drop=True)
        df_analyst_stock_recommendations = df_analyst_stock_recommendations.set_index(
            "date"
        )

        # Load existing data and merge with analyst recommendations
        output_file = data_path / f"{stock}_{config.BENCHMARK_END_DATE}_all.csv"
        df_all = pd.read_csv(output_file)
        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all = df_all.sort_values("date", ascending=True).reset_index(drop=True)
        df_all = df_all.set_index("date")

        # Merge datasets by time interval
        merged = merge_by_interval(df_all, df_analyst_stock_recommendations, "asr_days")
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        # Save merged data to file
        output_file = data_path / f"{stock}_{config.BENCHMARK_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    # Load default config if not provided
    if config is None:
        import src.config as config
    process_all_stocks(config)
