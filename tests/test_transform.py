import numpy as np
import pandas as pd
import pytest

from tabular_agent.transform import transform_features, SandboxError
from tabular_agent.pipeline import Pipeline


def test_adds_columns_and_preserves_rows():
    df = pd.DataFrame({"x": [3.0, 6.0], "y": [4.0, 8.0]})
    code = (
        "import numpy as np\n"
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['dist'] = np.sqrt(df['x']**2 + df['y']**2)\n"
        "    return df\n"
    )
    out, summary = transform_features(df, code)
    assert list(out["dist"]) == [5.0, 10.0]
    assert summary["n_rows"] == 2
    assert summary["new_columns"] == ["dist"]
    # original rows untouched
    assert list(out["x"]) == [3.0, 6.0]


def test_full_length_output_leaves_the_sandbox():
    # Unlike explore (JSON-capped), transform returns the whole frame.
    df = pd.DataFrame({"a": range(500)})
    code = (
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['b'] = df['a'] * 2\n"
        "    return df\n"
    )
    out, summary = transform_features(df, code)
    assert len(out) == 500
    assert summary["n_rows"] == 500


def test_reordering_rows_is_rejected():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    code = (
        "def transform_features(df):\n"
        "    return df.sort_values('x', ascending=False)\n"
    )
    with pytest.raises(SandboxError, match="preserve the row index"):
        transform_features(df, code)


def test_dropping_rows_is_rejected():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    code = (
        "def transform_features(df):\n"
        "    return df[df['x'] > 1]\n"
    )
    with pytest.raises(SandboxError, match="preserve the row index"):
        transform_features(df, code)


def test_non_dataframe_return_is_rejected():
    df = pd.DataFrame({"x": [1.0]})
    code = "def transform_features(df):\n    return df['x'].mean()\n"
    with pytest.raises(SandboxError, match="must return a pandas DataFrame"):
        transform_features(df, code)


def test_cannot_train_a_model_here():
    # The structural guard: no model/sklearn libs -> cannot learn state.
    df = pd.DataFrame({"x": [1.0]})
    code = (
        "import lightgbm\n"
        "def transform_features(df):\n    return df\n"
    )
    with pytest.raises(SandboxError, match="train_tabular_model"):
        transform_features(df, code)


def test_pipeline_persists_and_replays_transform():
    df = pd.DataFrame({"x": [3.0, 6.0], "y": [4.0, 8.0]})
    code = (
        "import numpy as np\n"
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['dist'] = np.sqrt(df['x']**2 + df['y']**2)\n"
        "    return df\n"
    )
    p = Pipeline("regression")
    p.add_transform(code)
    # survives serialization (only the code is stored)
    restored = Pipeline.from_dict(p.to_dict())
    a = p.transform(df)
    b = restored.transform(df)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["dist"]) == [5.0, 10.0]
