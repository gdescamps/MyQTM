import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval
from src.utils.path import get_project_root


def process_all_stocks(config):
    # Set the path to the data directory and create it if it doesn't already exist
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    for stock in tqdm(config.TRADE_STOCKS):

        # Load ratings data for the current stock
        ratings_file = data_path / f"{stock}_{config.TRADE_END_DATE}_ratings.json"

        with open(ratings_file, "r") as f:
            ratings = json.load(f)

        # Merge with base historical data if available
        if config.BASE_END_DATE_FILE is not None:

            base_ratings_file = (
                data_path / f"{stock}_{config.BASE_END_DATE_FILE}_ratings.json"
            )

            with open(base_ratings_file, "r") as f:
                base_ratings = json.load(f)
            base_dates = []
            for item in base_ratings:
                base_dates.append(item["date"])

            base_end_date = pd.to_datetime(config.BASE_END_DATE)
            after_base_data = []
            for item in ratings:
                date = pd.to_datetime(item["date"])
                if date > base_end_date:
                    after_base_data.append(item)

            ratings = after_base_data + base_ratings

        # Convert to DataFrame and parse dates
        ratings = pd.DataFrame(ratings)
        ratings["date"] = pd.to_datetime(ratings["date"])

        # Keep only columns used in TOP_FEATURES
        columns_to_keep = [
            "date",
            "ratingScore",
            "ratingDetailsDCFScore",
            "ratingDetailsROEScore",
            "ratingDetailsROAScore",
            "ratingDetailsDEScore",
            "ratingDetailsPEScore",
            "ratingDetailsPBScore",
        ]

        if "overallScore" in ratings.columns and "ratingScore" in ratings.columns:
            ratings["ratingScore"] = ratings["overallScore"].combine_first(
                ratings["ratingScore"]
            )
        elif "overallScore" in ratings.columns:
            ratings["ratingScore"] = ratings["overallScore"]

        if (
            "discountedCashFlowScore" in ratings.columns
            and "ratingDetailsDCFScore" in ratings.columns
        ):
            ratings["ratingDetailsDCFScore"] = ratings[
                "discountedCashFlowScore"
            ].combine_first(ratings["ratingDetailsDCFScore"])
        elif "discountedCashFlowScore" in ratings.columns:
            ratings["ratingDetailsDCFScore"] = ratings["discountedCashFlowScore"]

        if (
            "returnOnEquityScore" in ratings.columns
            and "ratingDetailsROEScore" in ratings.columns
        ):
            ratings["ratingDetailsROEScore"] = ratings[
                "returnOnEquityScore"
            ].combine_first(ratings["ratingDetailsROEScore"])
        elif "returnOnEquityScore" in ratings.columns:
            ratings["ratingDetailsROEScore"] = ratings["returnOnEquityScore"]

        if (
            "returnOnAssetsScore" in ratings.columns
            and "ratingDetailsROAScore" in ratings.columns
        ):
            ratings["ratingDetailsROAScore"] = ratings[
                "returnOnAssetsScore"
            ].combine_first(ratings["ratingDetailsROAScore"])
        elif "returnOnAssetsScore" in ratings.columns:
            ratings["ratingDetailsROAScore"] = ratings["returnOnAssetsScore"]

        if (
            "debtToEquityScore" in ratings.columns
            and "ratingDetailsDEScore" in ratings.columns
        ):
            ratings["ratingDetailsDEScore"] = ratings[
                "debtToEquityScore"
            ].combine_first(ratings["ratingDetailsDEScore"])
        elif "debtToEquityScore" in ratings.columns:
            ratings["ratingDetailsDEScore"] = ratings["debtToEquityScore"]

        if (
            "priceToEarningsScore" in ratings.columns
            and "ratingDetailsPEScore" in ratings.columns
        ):
            ratings["ratingDetailsPEScore"] = ratings[
                "priceToEarningsScore"
            ].combine_first(ratings["ratingDetailsPEScore"])
        elif "priceToEarningsScore" in ratings.columns:
            ratings["ratingDetailsPEScore"] = ratings["priceToEarningsScore"]

        if (
            "priceToBookScore" in ratings.columns
            and "ratingDetailsPBScore" in ratings.columns
        ):
            ratings["ratingDetailsPBScore"] = ratings["priceToBookScore"].combine_first(
                ratings["ratingDetailsPBScore"]
            )
        elif "priceToBookScore" in ratings.columns:
            ratings["ratingDetailsPBScore"] = ratings["priceToBookScore"]

        # Drop all other columns
        ratings = ratings[columns_to_keep]

        # Create time series features with a window size of 2
        TS_SIZE = 2
        for col in ratings.columns:
            if col in ["date"]:
                continue
            # Convert each column value to a list representing a time series
            values = [
                [
                    x if pd.notnull(x) else np.nan
                    for x in ratings[col].iloc[i : min(len(ratings), i + TS_SIZE)]
                ]
                for i in range(len(ratings))
            ]
            ratings[f"ts_r_{col}"] = values
            # Remove the original column after converting to time series
            ratings = ratings.drop(columns=[col], errors="ignore")

        # Sort by date and set as index
        ratings = ratings.sort_values("date", ascending=True).reset_index(drop=True)
        ratings = ratings.set_index("date")

        # Load existing data and merge with ratings
        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_all = pd.read_csv(output_file)
        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all = df_all.sort_values("date", ascending=True).reset_index(drop=True)
        df_all = df_all.set_index("date")

        # Merge datasets by time interval
        merged = merge_by_interval(df_all, ratings, "r_days")
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        # Save merged data to file
        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    # Load default config if not provided
    if config is None:
        import src.config as config
    process_all_stocks(config)
