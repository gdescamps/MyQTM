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

        # Load key metrics data for the current stock
        key_metrics_file = (
            data_path / f"{stock}_{config.TRADE_END_DATE}_key_metrics.json"
        )

        with open(key_metrics_file, "r") as f:
            key_metrics = json.load(f)

        # Merge with base historical data if available
        if config.BASE_END_DATE is not None:

            base_key_metrics_file = (
                data_path / f"{stock}_{config.BASE_END_DATE}_key_metrics.json"
            )

            with open(base_key_metrics_file, "r") as f:
                base_key_metrics = json.load(f)

            base_end_date = pd.to_datetime(config.BASE_END_DATE)
            after_base_data = []
            for item in key_metrics:
                date = pd.to_datetime(item["date"])
                if date > base_end_date:
                    after_base_data.append(item)

            key_metrics = after_base_data + base_key_metrics

        # Convert to DataFrame and process
        df_key_metrics = pd.DataFrame(key_metrics)
        df_key_metrics["date"] = pd.to_datetime(df_key_metrics["date"])
        # Drop unnecessary columns to retain only key financial metrics
        df_key_metrics = df_key_metrics.drop(
            columns=[
                "symbol",
                "calendarYear",
                "period",
                "revenuePerShare",
                "netIncomePerShare",
                "operatingCashFlowPerShare",
                "freeCashFlowPerShare",
                "cashPerShare",
                "bookValuePerShare",
                "tangibleBookValuePerShare",
                "shareholdersEquityPerShare",
                "interestDebtPerShare",
                "marketCap",
                "enterpriseValue",
                "peRatio",
                "priceToSalesRatio",
                "pocfratio",
                "pfcfRatio",
                "pbRatio",
                "ptbRatio",
                "evToSales",
                "enterpriseValueOverEBITDA",
                "evToOperatingCashFlow",
                "evToFreeCashFlow",
                "earningsYield",
                "freeCashFlowYield",
                "dividendYield",
                "payoutRatio",
                "salesGeneralAndAdministrativeToRevenue",
                "intangiblesToTotalAssets",
                "grahamNumber",
                "grahamNetNet",
                "workingCapital",
                "tangibleAssetValue",
                "netCurrentAssetValue",
                "investedCapital",
                "averageReceivables",
                "averagePayables",
                "averageInventory",
                "capexPerShare",
                "interestCoverage",
                "calendarYear",
                "period",
                "revenuePerShare",
                "netIncomePerShare",
                "operatingCashFlowPerShare",
                "freeCashFlowPerShare",
                "cashPerShare",
                "bookValuePerShare",
                "tangibleBookValuePerShare",
                "shareholdersEquityPerShare",
                "interestDebtPerShare",
                "marketCap",
                "enterpriseValue",
                "peRatio",
                "priceToSalesRatio",
                "pocfratio",
                "pfcfRatio",
                "pbRatio",
                "ptbRatio",
                "evToSales",
                "enterpriseValueOverEBITDA",
                "evToOperatingCashFlow",
                "evToFreeCashFlow",
                "earningsYield",
                "freeCashFlowYield",
                # "debtToEquity",
                # "debtToAssets",
                "netDebtToEBITDA",
                "currentRatio",
                "interestCoverage",
                "incomeQuality",
                "dividendYield",
                "payoutRatio",
                "salesGeneralAndAdministrativeToRevenue",
                "researchAndDdevelopementToRevenue",
                "intangiblesToTotalAssets",
                "capexToOperatingCashFlow",
                "capexToRevenue",
                "capexToDepreciation",
                "stockBasedCompensationToRevenue",
                "grahamNumber",
                "roic",
                "returnOnTangibleAssets",
                "grahamNetNet",
                "workingCapital",
                "tangibleAssetValue",
                "netCurrentAssetValue",
                "investedCapital",
                "averageReceivables",
                "averagePayables",
                "averageInventory",
                "daysSalesOutstanding",
                "daysPayablesOutstanding",
                "daysOfInventoryOnHand",
                "receivablesTurnover",
                "payablesTurnover",
                "inventoryTurnover",
                "roe",
                "capexPerShare",
            ],
            errors="ignore",
        )

        # Forward fill missing values
        df_key_metrics = df_key_metrics.bfill()

        # Create time series features with a window size of 3
        TS_SIZE = 3
        for col in df_key_metrics.columns:

            if col in ["date"]:
                continue

            # Convert each column value to a list of floats representing a time series
            values = [
                [
                    float(x) if pd.notnull(x) else np.nan
                    for x in df_key_metrics[col].iloc[
                        i : min(len(df_key_metrics), i + TS_SIZE)
                    ]
                ]
                for i in range(len(df_key_metrics))
            ]

            df_key_metrics[f"ts_km_{col}"] = values
            # Remove the original column after converting to time series
            df_key_metrics = df_key_metrics.drop(
                columns=[col],
                errors="ignore",
            )

        # Sort by date and set as index
        df_key_metrics = df_key_metrics.sort_values("date", ascending=True).reset_index(
            drop=True
        )
        df_key_metrics = df_key_metrics.set_index("date")

        # Load historical price data
        output_file = (
            data_path / f"{stock}_{config.TRADE_END_DATE}_historical_price_full.csv"
        )
        df_price = pd.read_csv(output_file)
        df_price["date"] = pd.to_datetime(df_price["date"])
        df_price = df_price.sort_values("date", ascending=True).reset_index(drop=True)
        df_price = df_price.set_index("date")

        # Merge price data with key metrics
        merged = merge_by_interval(df_price, df_key_metrics, "km_days")
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        # Save merged data to file
        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    # Load default config if not provided
    if config is None:
        import src.config as config
    process_all_stocks(config)
