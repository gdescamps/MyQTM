import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.df import merge_by_interval
from src.path import get_project_root


def process_all_stocks(config):
    # Define the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Iterate through each stock and its corresponding sentiments files
    for stock in tqdm(config.TRADE_STOCKS):

        news = []

        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_all = pd.read_csv(output_file)
        df_all.set_index("date", inplace=True)

        file = "news_sentiments.json"

        # Construct the file path for the sentiments file
        sentiments_file = data_path / f"{stock}_{config.TRADE_END_DATE}_{file}"
        sentiments_file = str(sentiments_file)

        if os.path.exists(sentiments_file) is False:
            continue

        # Load the sentiments from the JSON file
        with open(sentiments_file, "r") as f:
            sentiments = json.load(f)

        # Append each sentiment to the list of all sentiments
        for key in sentiments.keys():
            s = sentiments[key]
            news.append(
                {
                    "date": key,
                    "sentiment_company": int(s["company"]),
                    "sentiment_competitors": int(s["competitors"]),
                    "sentiment_global": int(s["global"]),
                }
            )

        if len(news) == 0:
            print(f"No news data for {stock}. Skipping.")
            continue

        df_news = pd.DataFrame(news)
        df_news["date"] = pd.to_datetime(df_news["date"])
        df_news[["sentiment_company", "sentiment_competitors", "sentiment_global"]] = (
            df_news[
                ["sentiment_company", "sentiment_competitors", "sentiment_global"]
            ].astype(int)
        )

        TS_SIZE = 10
        for col in df_news.columns:
            if col in ["date"]:
                continue
            values = [
                [
                    float(x) if pd.notnull(x) else np.nan
                    for x in df_news[col].iloc[i : min(len(df_news), i + TS_SIZE)]
                ]
                for i in range(len(df_news))
            ]
            df_news[f"ts_news_{col}"] = values
            df_news = df_news.drop(
                columns=[col],
                errors="ignore",
            )

        df_news = df_news.sort_values("date", ascending=True).reset_index(drop=True)
        df_news = df_news.set_index("date")

        df_all.reset_index(inplace=True)
        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all = df_all.sort_values("date", ascending=True).reset_index(drop=True)
        df_all = df_all.set_index("date")

        merged = merge_by_interval(df_all, df_news, "news_days")
        merged = merged.dropna()
        merged = merged.iloc[20:]
        merged[["news_days"]] = merged[["news_days"]].astype(int)

        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        merged.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
