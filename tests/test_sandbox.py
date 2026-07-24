import numpy as np
import pandas as pd
import pytest

from tabular_agent.sandbox import SandboxError, run_readonly


def _df():
    return pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "x", "y", "y"]})


def test_runs_pandas_numpy_code():
    code = "def explore(df):\n    import numpy as np\n    return float(np.mean(df['a']))"
    assert run_readonly(_df(), code) == 2.5


def test_blocks_training_lib_with_teaching_hint():
    code = "def explore(df):\n    import lightgbm\n    return 1"
    with pytest.raises(SandboxError) as exc:
        run_readonly(_df(), code)
    assert "train_tabular_model" in str(exc.value)


def test_blocks_sklearn_with_teaching_hint():
    code = "def explore(df):\n    from sklearn.ensemble import RandomForestClassifier\n    return 1"
    with pytest.raises(SandboxError) as exc:
        run_readonly(_df(), code)
    msg = str(exc.value)
    assert "fit_encoding" in msg or "train_tabular_model" in msg


def test_blocks_non_whitelisted_import():
    # A third-party name outside the whitelist is refused before it can even
    # be resolved (stdlib and pandas/numpy/scipy are the only things allowed).
    code = "def explore(df):\n    import definitely_not_a_real_pkg\n    return 1"
    with pytest.raises(SandboxError) as exc:
        run_readonly(_df(), code)
    assert "whitelist" in str(exc.value)


def test_requires_explore_function():
    code = "x = 1"  # no explore(df)
    with pytest.raises(SandboxError) as exc:
        run_readonly(_df(), code)
    assert "explore" in str(exc.value)


def test_user_exception_is_isolated_not_fatal():
    code = "def explore(df):\n    raise ValueError('boom')"
    with pytest.raises(SandboxError) as exc:
        run_readonly(_df(), code)
    assert "boom" in str(exc.value)
    # the test process itself is unharmed — a following call still works.
    assert run_readonly(_df(), "def explore(df):\n    return len(df)") == 4


def test_timeout_kills_runaway_code():
    code = "def explore(df):\n    while True:\n        pass"
    with pytest.raises(SandboxError) as exc:
        run_readonly(_df(), code, timeout=1.0)
    assert "timed out" in str(exc.value)


def test_dataframe_result_is_capped():
    big = pd.DataFrame({"v": np.arange(1000)})
    code = "def explore(df):\n    return df"
    out = run_readonly(big, code, max_rows=50)
    assert out["__kind__"] == "dataframe"
    assert out["n_rows"] == 1000
    assert out["truncated"] is True
    assert len(out["rows"]) == 50
