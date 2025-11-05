import joblib
import pandas as pd

from src.utils.interval import get_interval_type
from src.utils.xgb import prepare_pipeline_for_xgboost_with_pipe


def main(config=None):

    if config is None:
        import src.config as config

    df_all_concat = pd.DataFrame()

    stock_list = list(config.TRADE_STOCKS)
    stock_list = sorted(stock_list)

    for idx, stock in enumerate(stock_list):
        all_file = config.DATA_DIR / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df = pd.read_csv(all_file)
        df.dropna(inplace=True)
        if "news_days" not in df.columns:
            continue
        df["stock"] = idx
        df_all_concat = pd.concat([df_all_concat, df], ignore_index=True)

    pipe, df_all = prepare_pipeline_for_xgboost_with_pipe(df_all_concat)
    joblib.dump(pipe, config.DATA_DIR / f"{config.TRADE_END_DATE}_prep_pipeline.joblib")

    # Ajout de la colonne stock_name en une seule opération pour éviter la fragmentation
    stock_names = [stock_list[int(idx)] for idx in df_all["stock"]]
    df_all = df_all.assign(stock_name=stock_names).copy()

    part1A_dates = []
    part1B_dates = []
    part2A_dates = []
    part2B_dates = []
    part3A_dates = []
    part3B_dates = []

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

    df_all.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_benchmark_XY.csv", index=False
    )

    df_part1A = df_all[df_all["date"].isin(part1A_dates)].copy()
    df_part1A.dropna(inplace=True)
    df_part1A.reset_index(inplace=True)
    df_part1A.drop(columns=["date"], inplace=True)
    df_part1A_Y = df_part1A["trend"]
    df_part1A_X = df_part1A.drop(columns=["trend"])
    df_part1A_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part1A_X.csv", index=False
    )
    df_part1A_Y.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part1A_Y.csv", index=False
    )

    df_part1B = df_all[df_all["date"].isin(part1B_dates)].copy()
    df_part1B.dropna(inplace=True)
    df_part1B.reset_index(inplace=True)
    df_part1B.drop(columns=["date"], inplace=True)
    df_part1B_Y = df_part1B["trend"]
    df_part1B_X = df_part1B.drop(columns=["trend"])
    df_part1B_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part1B_X.csv", index=False
    )
    df_part1B_Y.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part1B_Y.csv", index=False
    )

    df_part2A = df_all[df_all["date"].isin(part2A_dates)].copy()
    df_part2A.dropna(inplace=True)
    df_part2A.reset_index(inplace=True)
    df_part2A.drop(columns=["date"], inplace=True)
    df_part2A_Y = df_part2A["trend"]
    df_part2A_X = df_part2A.drop(columns=["trend"])
    df_part2A_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part2A_X.csv", index=False
    )
    df_part2A_Y.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part2A_Y.csv", index=False
    )

    df_part2B = df_all[df_all["date"].isin(part2B_dates)].copy()
    df_part2B.dropna(inplace=True)
    df_part2B.reset_index(inplace=True)
    df_part2B.drop(columns=["date"], inplace=True)
    df_part2B_Y = df_part2B["trend"]
    df_part2B_X = df_part2B.drop(columns=["trend"])
    df_part2B_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part2B_X.csv", index=False
    )
    df_part2B_Y.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part2B_Y.csv", index=False
    )

    df_part3A = df_all[df_all["date"].isin(part3A_dates)].copy()
    df_part3A.dropna(inplace=True)
    df_part3A.reset_index(inplace=True)
    df_part3A.drop(columns=["date"], inplace=True)
    df_part3A_Y = df_part3A["trend"]
    df_part3A_X = df_part3A.drop(columns=["trend"])
    df_part3A_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part3A_X.csv", index=False
    )
    df_part3A_Y.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part3A_Y.csv", index=False
    )

    df_part3B = df_all[df_all["date"].isin(part3B_dates)].copy()
    df_part3B.dropna(inplace=True)
    df_part3B.reset_index(inplace=True)
    df_part3B.drop(columns=["date"], inplace=True)
    df_part3B_Y = df_part3B["trend"]
    df_part3B_X = df_part3B.drop(columns=["trend"])
    df_part3B_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part3B_X.csv", index=False
    )
    df_part3B_Y.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part3B_Y.csv", index=False
    )
