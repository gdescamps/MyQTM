import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval
from src.utils.round import round_floats


def process_all_stocks(config):

    for stock in tqdm(config.TRADE_STOCKS):

        key_metrics_file = (
            config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_key_metrics.json"
        )

        with open(key_metrics_file, "r") as f:
            key_metrics = json.load(f)

        key_metrics = round_floats(key_metrics, precision=3)

        df_key_metrics = pd.DataFrame(key_metrics)
        df_key_metrics["date"] = pd.to_datetime(df_key_metrics["date"])
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
                # "netDebtToEBITDA",
                # "currentRatio",
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
                # "roic",
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
                # "roe",
                "capexPerShare",
            ],
            errors="ignore",
        )

        df_key_metrics = df_key_metrics.bfill()

        enable_min_max_normalisation = False
        no_min_max_normalisation = [
            "ts_km_roic",
            "ts_km_returnOnTangibleAssets",
            "ts_km_roe",
            "ts_km_stockBasedCompensationToRevenue",
        ]

        TS_SIZE = 3
        for col in df_key_metrics.columns:

            if col in ["date"]:
                continue

            if (enable_min_max_normalisation is False) or (
                col in no_min_max_normalisation
            ):
                values = [
                    [
                        float(x) if pd.notnull(x) else np.nan
                        for x in df_key_metrics[col].iloc[
                            i : min(len(df_key_metrics), i + TS_SIZE)
                        ]
                    ]
                    for i in range(len(df_key_metrics))
                ]
            else:
                max_values = [
                    max(df_key_metrics[col].iloc[i:].tolist())
                    for i in range(len(df_key_metrics))
                ]
                min_values = [
                    min(df_key_metrics[col].iloc[i:].tolist())
                    for i in range(len(df_key_metrics))
                ]

                values = [
                    [
                        (
                            (float(x) - min_values[i])
                            / (max_values[i] - min_values[i] + 1e-9)
                            if pd.notnull(x)
                            else np.nan
                        )
                        for x in df_key_metrics[col].iloc[
                            i : min(len(df_key_metrics), i + TS_SIZE)
                        ]
                    ]
                    for i in range(len(df_key_metrics))
                ]
            df_key_metrics[f"ts_km_{col}"] = values
            df_key_metrics = df_key_metrics.drop(
                columns=[col],
                errors="ignore",
            )

        df_key_metrics = df_key_metrics.sort_values("date", ascending=True).reset_index(
            drop=True
        )
        df_key_metrics = df_key_metrics.set_index("date")

        output_file = (
            config.DATA_DIR
            / f"{stock}_{config.TRADE_END_DATE}_historical_price_full.csv"
        )
        df_price = pd.read_csv(output_file)
        df_price["date"] = pd.to_datetime(df_price["date"])
        df_price = df_price.sort_values("date", ascending=True).reset_index(drop=True)
        df_price = df_price.set_index("date")

        merged = merge_by_interval(df_price, df_key_metrics, "km_days")
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
