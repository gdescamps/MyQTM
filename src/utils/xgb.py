import ast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class CustomExploder(BaseEstimator, TransformerMixin):
    """
    Transformer that applies the 'exploded' logic:
    - Numeric lists -> separate columns
    - Lists of strings -> LabelEncoder
    - Simple strings -> LabelEncoder
    - Numeric values -> unchanged
    """

    def __init__(self):
        self.encoders_ = {}  # store LabelEncoders per column
        self.columns_ = []  # keep track of created columns

    def fit(self, X, y=None):
        X = X.copy()
        self.columns_ = []
        self.encoders_ = {}

        for col in tqdm(X.columns):
            col_data = X[col]

            # helper for parsing lists
            def try_parse(val):
                if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                    try:
                        return ast.literal_eval(val)
                    except Exception:
                        return val
                return val

            parsed = col_data.apply(try_parse)

            # Check that all elements have the same type in the column (after parsing)
            types_in_col = parsed.map(type).unique()
            if len(types_in_col) > 1:
                raise AssertionError(
                    f"Column '{col}' contains multiple types after parsing: {types_in_col}"
                )

            if (
                parsed.dtype == object
                and parsed.apply(lambda x: isinstance(x, list)).all()
            ):
                sample = parsed.iloc[0]
                if all(isinstance(e, (int, float)) for e in sample):
                    max_len = parsed.map(len).max()
                    self.columns_.extend([f"{col}_{i}" for i in range(max_len)])
                elif all(isinstance(e, str) for e in sample):
                    flat_set = list({item for sublist in parsed for item in sublist})
                    le = LabelEncoder().fit(flat_set)
                    self.encoders_[col] = le
                    max_len = parsed.map(len).max()
                    self.columns_.extend([f"{col}_{i}" for i in range(max_len)])
                else:
                    raise ValueError(f"Column {col} contains a mixed list.")
            elif parsed.dtype == object and col != "date":
                le = LabelEncoder().fit(parsed.astype(str))
                self.encoders_[col] = le
                self.columns_.append(col)
            else:
                self.columns_.append(col)

        return self

    def transform(self, X):
        X = X.copy()
        # final_df = pd.DataFrame(index=X.index)
        col_blocks = []  # List of DataFrames or Series to concatenate

        for col in tqdm(X.columns):
            col_data = X[col]

            def try_parse(val):
                if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                    try:
                        return ast.literal_eval(val)
                    except Exception:
                        return val
                return val

            parsed = col_data.apply(try_parse)

            # Check that all elements have the same type in the column (after parsing)
            types_in_col = parsed.map(type).unique()
            if len(types_in_col) > 1:
                raise AssertionError(
                    f"Column '{col}' contains multiple types after parsing: {types_in_col}"
                )

            if (
                parsed.dtype == object
                and parsed.apply(lambda x: isinstance(x, list)).all()
            ):
                sample = parsed.iloc[0]
                if all(isinstance(e, (int, float)) for e in sample):
                    max_len = parsed.map(len).max()
                    cols = {
                        f"{col}_{i}": parsed.apply(
                            lambda x: x[i] if i < len(x) else np.nan
                        )
                        for i in range(max_len)
                    }
                    col_blocks.append(pd.DataFrame(cols, index=X.index))
                elif all(isinstance(e, str) for e in sample):
                    le = self.encoders_[col]
                    max_len = parsed.map(len).max()
                    cols = {
                        f"{col}_{i}": parsed.apply(
                            lambda x: le.transform([x[i]])[0] if i < len(x) else np.nan
                        )
                        for i in range(max_len)
                    }
                    col_blocks.append(pd.DataFrame(cols, index=X.index))
                else:
                    raise ValueError(f"Column {col} contains a mixed list.")
            elif parsed.dtype == object and col != "date":
                le = self.encoders_[col]
                col_blocks.append(
                    pd.Series(le.transform(parsed.astype(str)), name=col, index=X.index)
                )
            else:
                col_blocks.append(pd.Series(col_data, name=col, index=X.index))

        # Concatenate all columns at once to avoid fragmentation
        final_df = pd.concat(col_blocks, axis=1)

        # reindex to expected columns (guarantee stable schema)
        # final_df = final_df.reindex(columns=self.columns_, fill_value=np.nan)

        return final_df


def prepare_pipeline_for_xgboost_with_pipe(df: pd.DataFrame):
    """
    Build and fit a sklearn pipeline that transforms df.
    Returns the pipeline and the transformed DataFrame.
    """
    pipe = Pipeline([("exploder", CustomExploder())])
    transformed = pipe.fit_transform(df)
    return pipe, transformed
