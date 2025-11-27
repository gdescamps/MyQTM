import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval
from src.utils.path import get_project_root


def preprocess_economic_indicators(config=None):
    # Set the path to the data directory and create it if it doesn't already exist
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    if config is None:
        import src.config as config

    # Load economic indicators data
    economic_indicators_file = (
        data_path / f"{config.TRADE_END_DATE}_economic_indicators.json"
    )

    with open(economic_indicators_file, "r") as f:
        economic_indicators = json.load(f)

    # Merge with base historical data if available
    if config.BASE_END_DATE is not None:

        base_economic_indicators_file = (
            data_path / f"{config.BASE_END_DATE}_economic_indicators.json"
        )

        with open(base_economic_indicators_file, "r") as f:
            base_economic_indicators = json.load(f)

        base_end_date = pd.to_datetime(config.BASE_END_DATE)
        after_base_data = {}
        for key in economic_indicators.keys():
            key_date = pd.to_datetime(key)
            if key_date > base_end_date:
                after_base_data[key] = economic_indicators[key]

        economic_indicators = {**after_base_data, **base_economic_indicators}

    # Convert dictionary to list format for DataFrame creation
    economic_indicators_list = []

    for k in economic_indicators.keys():
        d = economic_indicators[k]
        d["date"] = k
        economic_indicators_list.append(d)

    df_economic_indicators = pd.DataFrame(economic_indicators_list)

    df_economic_indicators["date"] = pd.to_datetime(df_economic_indicators["date"])
    # Drop unnecessary columns, keeping only durableGoods, retailSales, and federalFunds
    df_economic_indicators = df_economic_indicators.drop(
        columns=[
            "GDP",
            "realGDP",
            "nominalPotentialGDP",
            "realGDPPerCapita",
            "retailMoneyFunds",
            "commercialBankInterestRateOnCreditCardPlansAllAccounts",
            "15YearFixedRateMortgageAverage",
            "consumerSentiment",
            # Additional drops - only keep durableGoods, retailSales, federalFunds
            "inflationRate",
            "CPI",
            "industrialProductionTotalIndex",
            "unemploymentRate",
            "smoothedUSRecessionProbabilities",
            "newPrivatelyOwnedHousingUnitsStartedTotalUnits",
            "30YearFixedRateMortgageAverage",  # if exists
        ],
        errors="ignore",
    )

    # Create time series features with a window size of 4
    TS_SIZE = 4
    for col in df_economic_indicators.columns:
        if col in ["date"]:
            continue
        # Convert each column value to a list of floats representing a time series
        values = [
            [
                float(x) if pd.notnull(x) else np.nan
                for x in df_economic_indicators[col].iloc[
                    i : min(len(df_economic_indicators), i + TS_SIZE)
                ]
            ]
            for i in range(len(df_economic_indicators))
        ]
        df_economic_indicators[f"ts_ei_{col}"] = values
        # Remove the original column after converting to time series
        df_economic_indicators = df_economic_indicators.drop(
            columns=[col],
            errors="ignore",
        )

    # Sort by date and set as index
    df_economic_indicators = df_economic_indicators.sort_values(
        "date", ascending=True
    ).reset_index(drop=True)
    df_economic_indicators = df_economic_indicators.set_index("date")
    return df_economic_indicators


def process_all_stocks(config):
    # Set the path to the data directory and create it if it doesn't already exist
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Preprocess economic indicators once
    df_economic_indicators = preprocess_economic_indicators()
    for stock in tqdm(config.TRADE_STOCKS):

        # Load price data for current stock
        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_price = pd.read_csv(output_file)
        df_price["date"] = pd.to_datetime(df_price["date"])
        df_price = df_price.sort_values("date", ascending=True).reset_index(drop=True)
        df_price = df_price.set_index("date")

        # Merge price data with economic indicators
        merged = merge_by_interval(df_price, df_economic_indicators, None)
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        # Save merged data to file
        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    # Load default config if not provided
    if config is None:
        import src.config as config
    process_all_stocks(config)
