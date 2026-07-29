"""Dataset profiling for the inspect_data tool."""

import pandas as pd

#: Low-cardinality categoricals get their actual values listed. Without this an
#: agent knows a column has 3 levels but not what they are — which makes an
#: ordinal encoding impossible to request, since the level *order* is a required
#: argument. Capped so a high-cardinality column cannot flood the context.
MAX_REPORTED_CATEGORIES = 20


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict:
    prof: dict = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": {}}
    for col in df.columns:
        s = df[col]
        info = {
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if not pd.api.types.is_numeric_dtype(s) and info["n_unique"] <= MAX_REPORTED_CATEGORIES:
            info["categories"] = sorted(str(v) for v in s.dropna().unique())
        prof["columns"][col] = info
    if target is not None and target in df.columns:
        s = df[target]
        prof["target"] = {
            "name": target,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": int(s.isna().sum()),
        }
    return prof
