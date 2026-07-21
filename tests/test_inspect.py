import numpy as np
import pandas as pd

from tabular_agent.inspect import profile_dataframe


def test_profile_basic_shape_and_columns():
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", "x", "y"]})
    prof = profile_dataframe(df)
    assert prof["n_rows"] == 3
    assert prof["n_cols"] == 2
    assert prof["columns"]["a"]["n_missing"] == 1
    assert prof["columns"]["b"]["n_unique"] == 2


def test_profile_includes_target_block():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    prof = profile_dataframe(df, target="y")
    assert prof["target"]["name"] == "y"
    assert prof["target"]["n_unique"] == 2
