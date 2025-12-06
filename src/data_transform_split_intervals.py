from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.interval import get_interval_type
from src.path import get_project_root
from src.xgb import prepare_pipeline_for_xgboost_with_pipe


def safe_div(a, b):
    # Safe division that returns NaN when dividing by zero or NaN
    return np.where((b != 0) & (~np.isnan(b)), a / b, np.nan)


def generate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate financial and technical ratios from existing features.
    """
    df_ratios = df.copy()

    def add_ratio(name, num, denom):
        # Add a new ratio column if both numerator and denominator exist
        if {num, denom}.issubset(df.columns):
            df_ratios[name] = safe_div(df[num], df[denom])
            # Check for NaN values in the new ratio
            nan_count = df_ratios[name].isna().sum()
            if nan_count > 0:
                print(
                    f"⚠️  {name}: {nan_count} NaN values ({100*nan_count/len(df_ratios):.1f}%)"
                )

    # === 1️⃣ Multi-horizon momentum and volatility ===
    add_ratio("momentum_1w", "ts_w_close_0", "ts_w_close_1")

    # === 2️⃣ Technical indicators ===
    add_ratio("rsi_over_price", "rsi_14", "ts_d_close_0")

    # === 6️⃣ Earnings and analyst ratings ===
    add_ratio(
        "hold_vs_buy", "ts_asr_analystRatingsHold_0", "ts_asr_analystRatingsbuy_0"
    )

    # === 7️⃣ Cross-market sentiment and macro ===
    add_ratio("sentiment_vs_rate", "ts_news_sentiment_global_0", "ts_ei_federalFunds_0")

    # === 8️⃣ Automatic temporal crosses ===
    # Keep only the specific ratios that are used
    add_ratio("ratio_ts_rsi_14_0_over_ts_rsi_14_1", "ts_rsi_14_0", "ts_rsi_14_1")
    add_ratio(
        "ratio_ts_d_volume_1_over_ts_d_volume_2", "ts_d_volume_1", "ts_d_volume_2"
    )
    add_ratio("ratio_ts_d_close_0_over_ts_d_close_2", "ts_d_close_0", "ts_d_close_2")

    # ✅ {len(df_ratios.columns) - len(df.columns)} new ratios generated.
    print(f"✅ {len(df_ratios.columns) - len(df.columns)} new ratios generated.")
    return df_ratios


def main(config=None):

    if config is None:
        import src.config as config

    # Set the path to the data directory and create it if it doesn't already exist
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    stock_list = list(config.TRADE_STOCKS)
    stock_list = sorted(stock_list)

    df_list = []

    # Load and clean data for all stocks
    for idx, stock in enumerate(stock_list):
        all_file = data_path / f"{stock}_{config.BENCHMARK_END_DATE}_all.csv"
        df = pd.read_csv(all_file)

        rows_before = len(df)
        # Identify and remove dates with missing data
        if "date" in df.columns:
            dates_with_na = df[df.isna().any(axis=1)]["date"].unique()
            if len(dates_with_na) > 0:
                print(
                    f"⚠️ {stock}: Removing {len(dates_with_na)} dates with missing data:"
                )
                print(f"   Dates: {sorted(dates_with_na)}")

        df.dropna(inplace=True)

        # Skip if the DataFrame is empty after dropna
        if df.empty:
            print(f"⚠️ {stock}: Skipping - DataFrame is empty after dropna")
            continue

        rows_after = len(df)

        if rows_before > rows_after:
            print(f"   Removed {rows_before - rows_after} rows total")

        # Skip if no 'news_days' column exists
        if "news_days" not in df.columns:
            print(f"⚠️ {stock}: Skipping - no 'news_days' column")
            continue

        df["stock"] = idx
        df_list.append(df)

    # Concatenate all stock dataframes
    df_all_concat = pd.concat(df_list, ignore_index=True)

    # Prepare pipeline and preprocess data for XGBoost
    pipe, df_all = prepare_pipeline_for_xgboost_with_pipe(df_all_concat)
    joblib.dump(pipe, data_path / f"{config.BENCHMARK_END_DATE}_prep_pipeline.joblib")

    # Add stock names column in a single operation to avoid fragmentation
    stock_names = [stock_list[int(idx)] for idx in df_all["stock"]]
    df_all = df_all.assign(stock_name=stock_names).copy()

    # Initialize date lists for each interval part
    part1A_dates = []
    part1B_dates = []
    part2A_dates = []
    part2B_dates = []
    part3A_dates = []
    part3B_dates = []

    # Classify dates into their respective interval types
    for date in df_all["date"]:
        type = get_interval_type(date)
        if type is None:
            continue
        if "part1A" in type:
            part1A_dates.append(date)
        elif "part1B" in type:
            part1B_dates.append(date)
        elif "part2A" in type:
            part2A_dates.append(date)
        elif "part2B" in type:
            part2B_dates.append(date)
        elif "part3A" in type:
            part3A_dates.append(date)
        elif "part3B" in type:
            part3B_dates.append(date)

    df_all = df_all.copy()

    # Save complete benchmark dataset
    df_all.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_benchmark_XY.csv", index=False
    )

    # Split data into parts and save X (features) and Y (target) for each part
    df_part1A = df_all[df_all["date"].isin(part1A_dates)].copy()
    df_part1A.dropna(inplace=True)
    df_part1A.reset_index(inplace=True)
    df_part1A.drop(columns=["date"], inplace=True)
    df_part1A_Y = df_part1A["trend"]
    df_part1A_X = df_part1A.drop(columns=["trend"])
    df_part1A_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part1A_X.csv", index=False
    )
    df_part1A_Y.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part1A_Y.csv", index=False
    )

    df_part1B = df_all[df_all["date"].isin(part1B_dates)].copy()
    df_part1B.dropna(inplace=True)
    df_part1B.reset_index(inplace=True)
    df_part1B.drop(columns=["date"], inplace=True)
    df_part1B_Y = df_part1B["trend"]
    df_part1B_X = df_part1B.drop(columns=["trend"])
    df_part1B_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part1B_X.csv", index=False
    )
    df_part1B_Y.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part1B_Y.csv", index=False
    )

    df_part2A = df_all[df_all["date"].isin(part2A_dates)].copy()
    df_part2A.dropna(inplace=True)
    df_part2A.reset_index(inplace=True)
    df_part2A.drop(columns=["date"], inplace=True)
    df_part2A_Y = df_part2A["trend"]
    df_part2A_X = df_part2A.drop(columns=["trend"])
    df_part2A_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part2A_X.csv", index=False
    )
    df_part2A_Y.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part2A_Y.csv", index=False
    )

    df_part2B = df_all[df_all["date"].isin(part2B_dates)].copy()
    df_part2B.dropna(inplace=True)
    df_part2B.reset_index(inplace=True)
    df_part2B.drop(columns=["date"], inplace=True)
    df_part2B_Y = df_part2B["trend"]
    df_part2B_X = df_part2B.drop(columns=["trend"])
    df_part2B_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part2B_X.csv", index=False
    )
    df_part2B_Y.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part2B_Y.csv", index=False
    )

    df_part3A = df_all[df_all["date"].isin(part3A_dates)].copy()
    df_part3A.dropna(inplace=True)
    df_part3A.reset_index(inplace=True)
    df_part3A.drop(columns=["date"], inplace=True)
    df_part3A_Y = df_part3A["trend"]
    df_part3A_X = df_part3A.drop(columns=["trend"])
    df_part3A_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part3A_X.csv", index=False
    )
    df_part3A_Y.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part3A_Y.csv", index=False
    )

    df_part3B = df_all[df_all["date"].isin(part3B_dates)].copy()
    df_part3B.dropna(inplace=True)
    df_part3B.reset_index(inplace=True)
    df_part3B.drop(columns=["date"], inplace=True)
    df_part3B_Y = df_part3B["trend"]
    df_part3B_X = df_part3B.drop(columns=["trend"])
    df_part3B_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part3B_X.csv", index=False
    )
    df_part3B_Y.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part3B_Y.csv", index=False
    )

    # Reload feature datasets and generate ratios for each part
    df_part1A_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part1A_X.csv")
    df_part1B_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part1B_X.csv")
    df_part2A_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part2A_X.csv")
    df_part2B_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part2B_X.csv")
    df_part3A_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part3A_X.csv")
    df_part3B_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part3B_X.csv")

    # Generate financial and technical ratios for each part
    df_part1A_X = generate_ratios(df_part1A_X)
    df_part1B_X = generate_ratios(df_part1B_X)
    df_part2A_X = generate_ratios(df_part2A_X)
    df_part2B_X = generate_ratios(df_part2B_X)
    df_part3A_X = generate_ratios(df_part3A_X)
    df_part3B_X = generate_ratios(df_part3B_X)

    # Save enriched feature datasets
    df_part1A_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part1A_X.csv", index=False
    )
    df_part1B_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part1B_X.csv", index=False
    )
    df_part2A_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part2A_X.csv", index=False
    )
    df_part2B_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part2B_X.csv", index=False
    )
    df_part3A_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part3A_X.csv", index=False
    )
    df_part3B_X.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_part3B_X.csv", index=False
    )

    # Generate ratios for the complete benchmark dataset and save
    df_bench = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_benchmark_XY.csv")
    df_bench = generate_ratios(df_bench)
    df_bench.to_csv(
        data_path / f"{config.BENCHMARK_END_DATE}_benchmark_XY.csv", index=False
    )
