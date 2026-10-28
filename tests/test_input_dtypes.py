"""A serving file that stores training values under another type replays the same."""

import numpy as np
import pandas as pd

from tabular_agent.pipeline import (
    Pipeline, conform_input_dtypes, feature_table_bundle, input_dtypes,
)

KEY_CODE = '''
def transform_features(df):
    df = df.copy()
    df["key"] = df["shop"].astype(str) + "_" + df["open"].astype(str)
    return df
'''


def _bundle(train):
    p = Pipeline("regression")
    p.add_transform(KEY_CODE, new_columns=["key"])
    return {"pipeline": p, "feature_names": ["open", "key"],
            "input_dtypes": input_dtypes(train, exclude=["y"])}


def test_whole_floats_go_back_to_integers():
    df = pd.DataFrame({"a": [1.0, 0.0], "b": [1.0, np.nan], "c": [1.5, 2.0]})
    out, notes = conform_input_dtypes(df, {"a": "int64", "b": "int64", "c": "int64"})
    assert str(out["a"].dtype) == "int64"
    assert str(out["b"].dtype) == "Int64" and out["b"].isna().sum() == 1
    # 1.5 is not a whole number: converting it would change a value.
    assert str(out["c"].dtype) == "float64"
    assert len(notes) == 2


def test_integers_widen_to_float_and_01_to_bool():
    df = pd.DataFrame({"f": [1, 2], "b": [0, 1], "s": ["x", "y"]})
    out, _ = conform_input_dtypes(df, {"f": "float64", "b": "bool", "s": "int64"})
    assert str(out["f"].dtype) == "float64"
    assert str(out["b"].dtype) == "bool"
    # A text column is not a lossless numeric conversion; it is left alone.
    assert str(out["s"].dtype) == "object" or str(out["s"].dtype).startswith("str")


def test_a_text_key_built_from_an_integer_column_matches_at_serving():
    train = pd.DataFrame({"shop": [1, 2], "open": [1, 0], "y": [3.0, 4.0]})
    serving = pd.DataFrame({"shop": [1, 2, 3], "open": [1.0, 0.0, np.nan]})
    X = feature_table_bundle(_bundle(train), serving)
    assert list(X["key"][:2]) == ["1_1", "2_0"]
    # The missing value reaches the models as a float NaN, not a pandas NA.
    assert str(X["open"].dtype) == "float64" and np.isnan(X["open"].iloc[2])


def test_a_bundle_without_recorded_types_replays_rows_as_given():
    train = pd.DataFrame({"shop": [1], "open": [1], "y": [3.0]})
    bundle = {**_bundle(train), "input_dtypes": {}}
    X = feature_table_bundle(bundle, pd.DataFrame({"shop": [1], "open": [1.0]}))
    assert X["key"].iloc[0] == "1_1.0"
