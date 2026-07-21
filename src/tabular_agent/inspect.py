"""Dataset profiling for the inspect_data tool."""

import pandas as pd


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict:
    prof: dict = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": {}}
    for col in df.columns:
        s = df[col]
        prof["columns"][col] = {
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
    if target is not None and target in df.columns:
        s = df[target]
        prof["target"] = {
            "name": target,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": int(s.isna().sum()),
        }
    return prof
