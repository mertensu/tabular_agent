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
    """Write a table so that a reader sees the old file or the new one, never half.

    Atomic for the same reason `journal.save` is (2026-08-06), and it should have
    been done at the same time — the journal is the *record*, and these are the
    data. Rossmann v3 found the gap: two tool calls claimed one table id, both
    wrote this path directly, the writes interleaved, and the next read died on
    `OSError: Corrupt snappy compressed data` — taking a run with eight
    experiments and a good model down with it.

    `os.replace` is atomic within a filesystem, so a concurrent overwrite now
    resolves to last-writer-wins with a valid file at every instant, rather than
    to a corrupt one. That is defence in depth: the id claim below should stop two
    writers ever choosing the same path, and this makes it survivable when
    something else does.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_parquet(tmp, index=False)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path
