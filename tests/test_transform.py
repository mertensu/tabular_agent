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
    with pytest.raises(SandboxError, match="finalize_run"):
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


ROW_LOCAL = """
import numpy as np
def transform_features(df):
    df = df.copy()
    df["ratio"] = df["a"] / (df["b"] + 1)
    df["flag"] = (df["a"] > 2).astype(int)
    df["binned"] = np.digitize(df["a"], [0, 5, 10])
    return df
"""

BATCH_DEPENDENT = """
import pandas as pd
def transform_features(df):
    df = df.copy()
    df["a_bin"] = pd.qcut(df["a"], 4, labels=False, duplicates="drop")
    return df
"""

NORMALISED = """
def transform_features(df):
    df = df.copy()
    df["a_z"] = (df["a"] - df["a"].mean()) / df["a"].std()
    return df
"""


def _frame(n=40):
    return pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) * 2})


def test_row_local_code_is_stable_and_accepted():
    out, summary = transform_features(_frame(), ROW_LOCAL)
    assert set(summary["new_columns"]) == {"ratio", "flag", "binned"}


def test_qcut_is_rejected_because_it_reads_the_whole_column():
    """The real bug: quantiles over 800 training rows, over 5 at serving time."""
    with pytest.raises(SandboxError, match="depend on which other rows"):
        transform_features(_frame(), BATCH_DEPENDENT)


def test_batch_normalisation_is_rejected_too():
    with pytest.raises(SandboxError, match="a_z"):
        transform_features(_frame(), NORMALISED)


def test_the_message_points_at_the_tool_that_can_do_it_properly():
    with pytest.raises(SandboxError, match="fit_encoding"):
        transform_features(_frame(), NORMALISED)


def test_the_check_can_be_switched_off():
    """Predict-time replay must not pay for a fit-time check."""
    out, _ = transform_features(_frame(), BATCH_DEPENDENT, check_stability=False)
    assert "a_bin" in out.columns
