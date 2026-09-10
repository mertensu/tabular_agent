"""Dataset profiling for the inspect_data tool and public proposal facade."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .encoders import to_periods
from .io import load_table

#: Low-cardinality categoricals get their actual values listed. Without this an
#: agent knows a column has 3 levels but not what they are — which makes an
#: ordinal encoding impossible to request, since the level *order* is a required
#: argument. Capped so a high-cardinality column cannot flood the context.
MAX_REPORTED_CATEGORIES = 20


SUPPORTED_TABLE_SUFFIXES = {".csv", ".parquet"}


def propose_identifier_columns(
    frame: pd.DataFrame,
    serving_columns: list[str] | None = None,
) -> list[str]:
    """Conservatively propose row identifiers; never apply the proposal.

    A unique column called ``PassengerId`` or ``transaction_id`` is usually a
    carrier for joining predictions back to rows, not a quantity a model should
    learn. Uniqueness alone is not enough (a continuous measurement can also be
    unique), and an ``id``-shaped name alone is not enough (``store_id`` may be a
    repeated, useful entity). Requiring both keeps this a proposal narrow enough
    to put in front of the owner without silently deciding the role.
    """
    available = (
        set(serving_columns) if serving_columns is not None else set(frame.columns)
    )
    proposed = []
    for column in frame.columns:
        raw_name = str(column)
        name = raw_name.lower()
        named = (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or raw_name.endswith("Id")
            or raw_name.endswith("ID")
        )
        series = frame[column]
        if column in available and named and series.notna().all() and series.is_unique:
            proposed.append(column)
    return proposed


def table_paths(source: str | os.PathLike) -> list[Path]:
    """Return every supported table in a file/folder, without descending."""
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_TABLE_SUFFIXES:
            raise ValueError(f"unsupported table format: {path.suffix!r}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"data source does not exist: {path}")
    paths = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_TABLE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no CSV or parquet tables in {path}")
    return paths


def profile_collection(source: str | os.PathLike) -> dict:
    """Profile all tables and propose, but never apply, modeling decisions."""
    paths = table_paths(source)
    frames = {p.name: load_table(str(p)) for p in paths}
    names = list(frames)

    def name_rank(name: str, tokens: tuple[str, ...]) -> int:
        return 0 if Path(name).stem.lower() in tokens else 1

    training = min(
        names,
        key=lambda n: (name_rank(n, ("train", "training")), -len(frames[n]), n),
    )
    serving_matches = [
        n for n in names
        if name_rank(n, ("test", "serving", "predict", "prediction", "inference")) == 0
    ]
    serving = min(serving_matches) if serving_matches else None
    train = frames[training]
    serving_columns = list(frames[serving].columns) if serving else None
    identifier_columns = propose_identifier_columns(train, serving_columns)

    target_tokens = {"target", "label", "response", "outcome", "sales", "y"}
    train_only = [
        c for c in train.columns
        if serving_columns is None or c not in serving_columns
    ]
    target_candidates = sorted(
        train_only,
        key=lambda c: (str(c).lower() not in target_tokens, str(c).lower()),
    )
    ordering_candidates = []
    for column in train.columns:
        series = train[column]
        named = any(token in str(column).lower() for token in ("date", "time", "timestamp"))
        if pd.api.types.is_datetime64_any_dtype(series) or named:
            ordering_candidates.append(column)

    joins = []
    for right_name, right in frames.items():
        if right_name in {training, serving}:
            continue
        for key in [c for c in train.columns if c in right.columns]:
            if serving_columns is not None and key not in serving_columns:
                continue
            if right[key].notna().all() and right[key].is_unique:
                added = [c for c in right.columns if c != key and c not in train.columns]
                if added:
                    joins.append({
                        "right_table": right_name,
                        "left_on": key,
                        "right_on": key,
                        "columns": added,
                    })

    return {
        "source": str(Path(source).expanduser().resolve()),
        "tables": {
            name: {"path": str(paths[names.index(name)]), **profile_dataframe(frame)}
            for name, frame in frames.items()
        },
        "training_table": training,
        "serving_table": serving,
        "target_candidates": target_candidates,
        "training_only_columns": train_only,
        "ordering_candidates": ordering_candidates,
        "serving_columns": serving_columns,
        "identifier_columns": identifier_columns,
        "joins": joins,
    }


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


def profile_stream(
    stream: pd.DataFrame,
    event_time: str,
    observed_at: str,
    table_columns: list[str] | None = None,
    notes: str | None = None,
) -> dict:
    """What an agent needs to decide whether — and at what lead — to use a stream.

    Three things, and the third is the one no ordinary profile would think to
    compute.

    * **Which columns could join.** A stream is useless without a key, and the key
      is whatever the two files share. Listing the overlap turns "here is a second
      file" into "here is how it attaches".
    * **What the file spans**, so a lead can be read against something.
    * **The notice distribution** — how far ahead of the event each record was
      observed. This is the number that decides everything. A stream whose records
      all arrive months early costs nothing to restrict; one where a third of them
      arrive the same day loses that third at every horizon, and the loss falls
      entirely on the rows being forecast, because the training rows keep them.

    ``notice_zero_share`` is that failure stated as one number. It is computed
    here rather than left to the agent because the agent cannot: the two
    timestamps live in a file no tool returns rows from.
    """
    n = int(len(stream))
    out: dict = {
        "n_records": n,
        "event_time": event_time,
        "observed_at": observed_at,
        "columns": [str(c) for c in stream.columns],
    }
    if notes:
        out["notes"] = notes
    if table_columns is not None:
        shared = [c for c in stream.columns if c in set(table_columns)]
        out["shared_with_table"] = shared
        if not shared:
            out["warning"] = (
                "no column of this stream appears in the table, so there is nothing "
                "to join it on. Check that the key carries the same name on both "
                "sides."
            )
    if not n:
        return out

    event = to_periods(stream[event_time])
    observed = to_periods(stream[observed_at])
    finite = np.isfinite(event) & np.isfinite(observed)
    if not finite.any():
        return out
    notice = (event - observed)[finite]
    out["event_time_range"] = [
        round(float(np.nanmin(event)), 3),
        round(float(np.nanmax(event)), 3),
    ]
    # In the event time column's own units, which `to_periods` makes days for a
    # date column. Quartiles rather than a mean: a handful of records booked a
    # year out drags a mean somewhere no record actually is, and the decision this
    # feeds — which lead to ask for — is a decision about where the mass sits.
    out["notice"] = {
        "median": round(float(np.median(notice)), 3),
        "q25": round(float(np.quantile(notice, 0.25)), 3),
        "q75": round(float(np.quantile(notice, 0.75)), 3),
    }
    out["notice_zero_share"] = round(float((notice <= 0).mean()), 4)
    return out
