"""Dataset profiling for the inspect_data tool."""

import pandas as pd

#: Low-cardinality categoricals get their actual values listed. Without this an
#: agent knows a column has 3 levels but not what they are — which makes an
#: ordinal encoding impossible to request, since the level *order* is a required
#: argument. Capped so a high-cardinality column cannot flood the context.
MAX_REPORTED_CATEGORIES = 20


def profile_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
    column_notes: dict[str, str] | None = None,
) -> dict:
    """Shape, per-column statistics, and whatever the data's owner said about it.

    ``column_notes`` is the data dictionary: ``{column: what it means}``, supplied
    by whoever handed over the dataset. It rides on the column it describes rather
    than arriving as a separate block, because that is where the reader already
    is — deciding what a column is *for* happens next to its dtype and
    cardinality, not three screens away.

    The statistics cannot supply this. "1115 unique integers" does not say whether
    a column is a store identifier worth encoding or a row number to throw away,
    and no profile can tell that a column will simply not exist at predict time.
    """
    column_notes = column_notes or {}
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
        if col in column_notes:
            info["description"] = column_notes[col]
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
