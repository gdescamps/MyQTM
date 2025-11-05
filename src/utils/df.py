import pandas as pd


def merge_by_interval(
    df1: pd.DataFrame, df2: pd.DataFrame, col_name_diff_day: str
) -> pd.DataFrame:
    """
    Merges df2 into df1 by associating to each date in df1 the columns of df2
    corresponding to the last date in df2 that is less than or equal.

    Indexes must be of datetime type. Columns of df1 and df2 must be different.

    Parameters:
        df1 (pd.DataFrame): Reference DataFrame with a datetime index.
        df2 (pd.DataFrame): DataFrame containing columns to add, with a datetime index.

    Returns:
        pd.DataFrame: A merged DataFrame with columns from df2 aligned to df1,
                      and a new column 'days_diff' indicating the date difference.
    """
    # Check index types
    if not pd.api.types.is_datetime64_any_dtype(df1.index):
        raise ValueError("The index of df1 must be of datetime type")
    if not pd.api.types.is_datetime64_any_dtype(df2.index):
        raise ValueError("The index of df2 must be of datetime type")

    # Reset and rename index to keep original date as a column
    df1_reset = df1.reset_index()
    df1_reset.rename(columns={df1_reset.columns[0]: "date"}, inplace=True)

    df2_reset = df2.reset_index()
    df2_reset.rename(columns={df2_reset.columns[0]: "matched_df2_date"}, inplace=True)

    # Sort by date
    df1_sorted = df1_reset.sort_values("date")
    df2_sorted = df2_reset.sort_values("matched_df2_date")

    # Merge using merge_asof
    merged = pd.merge_asof(
        df1_sorted,
        df2_sorted,
        left_on="date",
        right_on="matched_df2_date",
        direction="backward",
    )

    # Compute the difference in days
    if col_name_diff_day is not None:
        merged[col_name_diff_day] = (
            merged["date"] - merged["matched_df2_date"]
        ).dt.days

    # Drop the matched_df2_date column
    merged.drop(columns=["matched_df2_date"], inplace=True)

    return merged
