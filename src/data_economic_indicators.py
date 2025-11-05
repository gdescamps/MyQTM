import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.df import merge_by_interval


def preprocess_economic_indicators(config=None):
    if config is None:
        import src.config as config

    economic_indicators_file = (
        config.DATA_DIR / f"{config.TRADE_END_DATE}_economic_indicators.json"
    )

    with open(economic_indicators_file, "r") as f:
        economic_indicators = json.load(f)

    economic_indicators_list = []

    for k in economic_indicators.keys():
        d = economic_indicators[k]
        d["date"] = k
        economic_indicators_list.append(d)

    df_economic_indicators = pd.DataFrame(economic_indicators_list)

    df_economic_indicators["date"] = pd.to_datetime(df_economic_indicators["date"])
    df_economic_indicators = df_economic_indicators.drop(
        columns=[
            "GDP",  # retardé et corrélé à realGDP
            "realGDP",  # plus utile en macro long terme
            "nominalPotentialGDP",  # indicateur structurel, très lent
            "realGDPPerCapita",  # encore plus structurel
            "retailMoneyFunds",  # difficile à exploiter sans contexte
            "commercialBankInterestRateOnCreditCardPlansAllAccounts",  # spécifique et lent
            "15YearFixedRateMortgageAverage",  # redondant avec le 30Y
        ],
        errors="ignore",
    )

    TS_SIZE = 4
    for col in df_economic_indicators.columns:
        if col in ["date"]:
            continue
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
        df_economic_indicators = df_economic_indicators.drop(
            columns=[col],
            errors="ignore",
        )

    df_economic_indicators = df_economic_indicators.sort_values(
        "date", ascending=True
    ).reset_index(drop=True)
    df_economic_indicators = df_economic_indicators.set_index("date")
    return df_economic_indicators


def process_all_stocks(config):
    df_economic_indicators = preprocess_economic_indicators()
    for stock in tqdm(config.TRADE_STOCKS):

        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_price = pd.read_csv(output_file)
        df_price["date"] = pd.to_datetime(df_price["date"])
        df_price = df_price.sort_values("date", ascending=True).reset_index(drop=True)
        df_price = df_price.set_index("date")

        merged = merge_by_interval(df_price, df_economic_indicators, None)
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        output_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
