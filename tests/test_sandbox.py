import numpy as np
import pandas as pd
import pytest

from tabular_agent.sandbox import SandboxError, run_readonly


def _df():
    return pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "x", "y", "y"]})


def test_runs_pandas_numpy_code():
    code = "def explore(df):\n    import numpy as np\n    return float(np.mean(df['a']))"
    assert run_readonly({"df": _df()}, code)["result"] == 2.5


def test_blocks_training_lib_with_teaching_hint():
    code = "def explore(df):\n    import lightgbm\n    return 1"
    with pytest.raises(SandboxError) as exc:
        run_readonly({"df": _df()}, code)
    assert "train_tabular_model" in str(exc.value)


def test_blocks_sklearn_with_teaching_hint():
    code = "def explore(df):\n    from sklearn.ensemble import RandomForestClassifier\n    return 1"
    with pytest.raises(SandboxError) as exc:
        run_readonly({"df": _df()}, code)
    msg = str(exc.value)
    assert "fit_encoding" in msg or "train_tabular_model" in msg


def test_blocks_non_whitelisted_import():
    # A third-party name outside the whitelist is refused before it can even
    # be resolved (stdlib and pandas/numpy/scipy are the only things allowed).
    code = "def explore(df):\n    import definitely_not_a_real_pkg\n    return 1"
    with pytest.raises(SandboxError) as exc:
        run_readonly({"df": _df()}, code)
    assert "whitelist" in str(exc.value)


def test_requires_explore_function():
    code = "x = 1"  # no explore(df)
    with pytest.raises(SandboxError) as exc:
        run_readonly({"df": _df()}, code)
    assert "explore" in str(exc.value)


def test_user_exception_is_isolated_not_fatal():
    code = "def explore(df):\n    raise ValueError('boom')"
    with pytest.raises(SandboxError) as exc:
        run_readonly({"df": _df()}, code)
    assert "boom" in str(exc.value)
    # the test process itself is unharmed — a following call still works.
    assert run_readonly({"df": _df()}, "def explore(df):\n    return len(df)")["result"] == 4


def test_timeout_kills_runaway_code():
    code = "def explore(df):\n    while True:\n        pass"
    with pytest.raises(SandboxError) as exc:
        run_readonly({"df": _df()}, code, timeout=1.0)
    assert "timed out" in str(exc.value)


def test_dataframe_result_is_capped():
    big = pd.DataFrame({"v": np.arange(1000)})
    code = "def explore(df):\n    return df"
    out = run_readonly({"df": big}, code, max_rows=50)["result"]
    assert out["__kind__"] == "dataframe"
    assert out["n_rows"] == 1000
    assert out["truncated"] is True
    assert len(out["rows"]) == 50


# --------------------------------------------------------------- the namespace


def test_only_the_declared_frames_are_passed():
    # Three frames go in; `explore` names one, and gets one. Declaring the
    # others (in the right order) is not a rule the agent should have to keep.
    frames = {"lb": pd.DataFrame({"score": [1.0, 2.0]}), "df": _df(),
              "notes": pd.DataFrame({"text": ["x"]})}
    code = "def explore(lb):\n    return float(lb['score'].sum())"
    assert run_readonly(frames, code)["result"] == 3.0


def test_unknown_frame_name_lists_what_exists():
    code = "def explore(leaderboard):\n    return 1"
    with pytest.raises(SandboxError) as exc:
        run_readonly({"lb": _df(), "notes": _df()}, code)
    msg = str(exc.value)
    assert "leaderboard" in msg and "lb" in msg and "notes" in msg


def test_frames_are_not_shared_between_calls():
    # No kernel: a name bound in one call is gone in the next. The state is
    # `run_dir`, and anything the sandbox holds dies with the subprocess.
    run_readonly({"df": _df()}, "def explore(df):\n    global z\n    z = 99\n    return 1")
    with pytest.raises(SandboxError) as exc:
        run_readonly({"df": _df()}, "def explore(df):\n    return z")
    assert "z" in str(exc.value)


# ------------------------------------------------------------------- stdout


def test_stdout_is_captured_and_returned():
    code = "def explore(df):\n    print('rows:', len(df))\n    return 'done'"
    out = run_readonly({"df": _df()}, code)
    assert out["result"] == "done"
    assert "rows: 4" in out["stdout"]


def test_prints_before_an_error_survive_in_the_message():
    # The point of capturing: code that raises halfway still reports what it
    # computed on the way, which is what the fix is written from.
    code = (
        "def explore(df):\n"
        "    print('groups:', sorted(df['g'].unique()))\n"
        "    return df['missing_col'].sum()"
    )
    with pytest.raises(SandboxError) as exc:
        run_readonly({"df": _df()}, code)
    msg = str(exc.value)
    assert "missing_col" in msg
    assert "groups:" in msg and "'x', 'y'" in msg


def test_stdout_is_capped():
    from tabular_agent.sandbox import MAX_STDOUT_CHARS

    code = "def explore(df):\n    print('z' * 100000)\n    return 1"
    out = run_readonly({"df": _df()}, code)
    assert len(out["stdout"]) < MAX_STDOUT_CHARS + 100
    assert "truncated" in out["stdout"]


def test_child_cannot_read_caller_secrets(monkeypatch):
    # The provider key a run uses lives in the caller's environment; agent code
    # must not be able to read it, whatever the variable is called.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("SOME_CUSTOM_TOKEN", "also-secret")
    code = (
        "def explore(df):\n"
        "    import os\n"
        "    return {k: os.environ.get(k) for k in ('ANTHROPIC_API_KEY', 'SOME_CUSTOM_TOKEN')}"
    )
    result = run_readonly({"df": _df()}, code)["result"]
    assert result == {"ANTHROPIC_API_KEY": None, "SOME_CUSTOM_TOKEN": None}
