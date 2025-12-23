import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import src.config as config
from config import BENCHMARK_START_DATE, TRAIN_DIR
from src.path import get_project_root
from src.printlog import PrintLog
from src.trade import build_trade_data

if __name__ == "__main__":

    # Load environment variables from .env file
    # This ensures sensitive information like API keys is securely loaded into the environment.
    load_dotenv()

    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    data_path = Path(get_project_root()) / "data" / "fmp_data"
    model_path = Path(get_project_root()) / TRAIN_DIR

    local_log = PrintLog(extra_name="_analyse", enable=False)

    BENCHMARK_END_DATE = [
        "2025-12-10",
        "2025-12-11",
        "2025-12-12",
        "2025-12-15",
        "2025-12-16",
        "2025-12-17",
        "2025-12-18",
        "2025-12-19",
        "2025-12-22",
    ]
    TICKER = ["NKE", "ROST", "NFLX", "VRTX"]

    benchmark_end_dates = (
        BENCHMARK_END_DATE
        if isinstance(BENCHMARK_END_DATE, list)
        else [BENCHMARK_END_DATE]
    )
    # current_date = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")
    # if os.path.exists(os.path.join(data_path, f"{current_date}_benchmark_XY.csv")):
    #     benchmark_end_date = current_date

    with local_log:
        print(f"Analyse until {benchmark_end_dates} :")

    bench_start_date = pd.to_datetime(BENCHMARK_START_DATE, format="%Y-%m-%d")

    dates_portfolio = config.DATES_PORTFOLIO
    remove_stocks_list = []

    trade_data_by_date = []
    for benchmark_end_date in benchmark_end_dates:
        bench_end_date = pd.to_datetime(benchmark_end_date, format="%Y-%m-%d")
        trade_data = build_trade_data(
            model_path=model_path,
            data_path=data_path,
            file_date_str=benchmark_end_date,
            start_date=bench_start_date,
            end_date=bench_end_date,
        )
        trade_data_by_date.append(
            {
                "benchmark_end_date": benchmark_end_date,
                "trade_data": trade_data,
            }
        )

    color_map = {2: "green", 1: "orange", 0: "red"}
    dates_label = ", ".join(benchmark_end_dates)

    for ticker in TICKER:
        trade_datasets = []
        for entry in trade_data_by_date:
            benchmark_end_date = entry["benchmark_end_date"]
            trade_data = entry["trade_data"]
            dates_sorted = sorted(trade_data.keys())
            ybull_values = []
            y_values = []
            date_values = []

            for date_str in dates_sorted:
                ticker_data = trade_data.get(date_str, {}).get(ticker)
                if ticker_data is None:
                    continue
                date_values.append(pd.to_datetime(date_str, format="%Y-%m-%d"))
                ybull_values.append(ticker_data.get("ybull"))
                y_values.append(ticker_data.get("Y"))

            if len(date_values) == 0:
                print(f"No data to plot for ticker {ticker} on {benchmark_end_date}")
                continue

            trade_datasets.append(
                {
                    "benchmark_end_date": benchmark_end_date,
                    "date_values": np.array(date_values),
                    "ybull_values": np.array(ybull_values, dtype=float),
                    "y_values": np.array(y_values, dtype=int),
                }
            )

        if len(trade_datasets) == 0:
            print(f"No data to plot for ticker {ticker}")
            continue

        end_date = max(dataset["date_values"].max() for dataset in trade_datasets)
        ranges = [
            ("full", "Full", None),
            ("last_year", "Last year", end_date - pd.Timedelta(days=365)),
            ("last_month", "Last month", end_date - pd.Timedelta(days=30)),
        ]

        def plot_range(title, start_date, png_suffix):
            markersize = 12 if png_suffix == "last_month" else 6
            fig, ax = plt.subplots(figsize=(12, 5))
            for dataset_idx, dataset in enumerate(trade_datasets):
                date_values = dataset["date_values"]
                ybull_values = dataset["ybull_values"]
                y_values = dataset["y_values"]
                if start_date is None:
                    range_mask = np.ones_like(date_values, dtype=bool)
                else:
                    range_mask = date_values >= start_date
                for y_class, color in color_map.items():
                    mask = (y_values == y_class) & range_mask
                    label = f"Y={y_class}" if dataset_idx == 0 else None
                    ax.plot(
                        date_values[mask],
                        ybull_values[mask],
                        color=color,
                        marker=".",
                        linestyle="None",
                        markersize=markersize,
                        alpha=0.5,
                        label=label,
                    )
            ax.set_title(f"{title} - {ticker} ({dates_label})")
            ax.set_xlabel("Date")
            ax.set_ylabel("ybull")
            ax.legend()
            fig.autofmt_xdate()

            png_path = os.path.join(
                local_log.output_dir_time,
                f"{ticker}_{png_suffix}.png",
            )
            fig.savefig(png_path)
            plt.close(fig)

        for png_suffix, title, start_date in ranges:
            plot_range(title, start_date, png_suffix)
