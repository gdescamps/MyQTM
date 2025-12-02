# %%
from pathlib import Path

import pandas as pd

import src.config as config
from src.path import get_project_root

# %%
# Set the path to the data directory
# Create a directory to store downloaded data if it doesn't already exist.
data_path = Path(get_project_root()) / "data" / "fmp_data"

# %%
df_bench1 = pd.read_csv(data_path / "2025-09-05_benchmark_XY.csv")

# %%
df_bench2 = pd.read_csv(data_path / "2025-11-28_benchmark_XY.csv")

# %%
df_bench1.tail()

# %%
df_bench2.tail()

# %%
for i, stock in enumerate(config.TRADE_STOCKS):
    if stock != "AAPL":
        continue
    df_stock1 = df_bench1[df_bench1["stock_name"] == stock]
    df_stock2 = df_bench2[df_bench2["stock_name"] == stock]
    for col in df_stock1.columns:
        if "date" in col:
            continue
        if "trend" in col:
            continue
        if "km" in col:
            continue
        if "ei" in col:
            continue
        if "debt" in col:
            continue
        if "roe" in col:
            continue
        if "roic" in col:
            continue
        if "buy_vs_sell" in col:
            continue
        if "strongBuy_vs_strongSell" in col:
            continue
        if "sentiment_vs_recession" in col:
            continue

        assert col in df_stock2.columns, f"Column {col} missing in stock {stock}"
        vals1 = df_stock1[col].values
        vals2 = df_stock2[col].values
        vals2 = vals2[: len(vals1)]

        if len(vals2) != len(vals1):
            continue

        # assert len(vals1) == len(
        #     vals2
        # ), f"Length mismatch in stock {stock}, column {col}"

        # Trouver les indices où il y a des différences
        diff_mask = vals1 != vals2
        diff_indices = [idx for idx, is_diff in enumerate(diff_mask) if is_diff]

        # Check if all differences are NaN on both sides
        all_diffs_are_nan = (
            all(pd.isna(vals1[idx]) and pd.isna(vals2[idx]) for idx in diff_indices)
            if len(diff_indices) > 0
            else False
        )

        if len(diff_indices) > 10 and not all_diffs_are_nan:
            # Récupérer la date de la ligne

            date = (
                df_stock1.iloc[diff_indices[0]]["date"]
                if "date" in df_stock1.columns
                else "N/A"
            )

            print(
                f"Number of different rows for {date} {stock} in {col}: {len(diff_indices)}"
            )
            for idx in diff_indices[:10]:
                val1 = vals1[idx]
                val2 = vals2[idx]

                # Récupérer la date de la ligne
                date1 = (
                    df_stock1.iloc[idx]["date"]
                    if "date" in df_stock1.columns
                    else "N/A"
                )

                # Calculer la différence relative si les valeurs ne sont pas nulles
                if pd.notna(val1) and pd.notna(val2):
                    if val1 != 0:
                        rel_diff = abs((val2 - val1) / val1) * 100
                        print(
                            f"  Row {idx} [{date1}]: {val1:.6f} -> {val2:.6f} (diff: {val2-val1:.6f}, {rel_diff:.2f}%)"
                        )
                    else:
                        print(
                            f"  Row {idx} [{date1}]: {val1:.6f} -> {val2:.6f} (diff: {val2-val1:.6f})"
                        )
                # else:
                #     print(f"  Row {idx} [{date1}]: {val1} -> {val2} (NaN detected)")

# %%
