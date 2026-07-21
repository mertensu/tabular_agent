import os

import pandas as pd

from tabular_agent.io import load_table, save_parquet
from tabular_agent.runctx import RunContext


def test_save_and_load_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = save_parquet(df, str(tmp_path / "sub" / "data.parquet"))
    assert os.path.exists(path)
    loaded = load_table(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_load_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    df = load_table(str(p))
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_runcontext_path_creates_dirs(tmp_path):
    ctx = RunContext(str(tmp_path / "run1"))
    p = ctx.path("model", "final.pkl")
    assert p.endswith(os.path.join("run1", "model", "final.pkl"))
    assert os.path.isdir(os.path.dirname(p))
