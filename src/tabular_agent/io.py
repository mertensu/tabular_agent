"""Table IO: CSV and Parquet by file extension."""

import os

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"unsupported extension: {path}")


def save_parquet(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False)
    return path
