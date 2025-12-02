from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.path import get_project_root


def process_all_stocks(config):
    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    for stock in tqdm(config.TRADE_STOCKS):

        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_all = pd.read_csv(output_file)

        for col in df_all.columns:
            if "days" in col:
                if df_all[col].dtype == "float64":
                    df_all[col] = df_all[col].fillna(-1).astype(int)

        output_file = data_path / f"{stock}_{config.TRADE_END_DATE}_all.csv"
        df_all.to_csv(output_file, index=False)


def main(config=None):
    if config is None:
        import src.config as config
    process_all_stocks(config)
