"""`fit_future_covariate_window` as a tool: the ordering rule against row filters.

A training-only filter removes rows from training but does not run at prediction
time. A window over a known-in-advance column counts neighbouring rows, so it has
to see the same rows in both places: built before the filter, never after.
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import filter_rows, fit_future_covariate_window, inspect_data
from tabular_agent.taskspec import TaskSpec

KEEP_OPEN = """
def keep_rows(df):
    return df["open"] == 1
"""
WINDOWS = [{"family": "trailing", "window": 7, "stat": "sum"}]


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=200)
    df = pd.DataFrame({"shop": np.repeat(["a", "b"], len(days)),
                       "date": np.tile(days, 2)})
    df["open"] = (df["date"].dt.dayofweek != 6).astype(int)
    df["y"] = df["open"] * (100 + rng.normal(0, 5, len(df)))
    path = tmp_path / "shops.csv"
    df.to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), str(path),
                   TaskSpec(target="y", n_splits=3, forecast_horizon=14.0),
                   split_on="date", future_known=["open"], workflow="forecast")
    inspect_data(s)
    return s


def test_a_window_after_a_training_only_filter_is_refused(session):
    filtered = filter_rows(session, KEEP_OPEN)
    with pytest.raises(ValueError, match="training-only row filter"):
        fit_future_covariate_window(session, "open", WINDOWS, by=["shop"],
                                    table=filtered["table"])


def test_windows_first_then_the_filter_is_the_supported_order(session):
    windowed = fit_future_covariate_window(session, "open", WINDOWS, by=["shop"])
    filtered = filter_rows(session, KEEP_OPEN, table=windowed["table"])
    table = pd.read_parquet(f"{session.run_dir}/{filtered['table']}")
    # Computed before the filter, the window still saw the closed days.
    assert (table["shop_open_roll7_sum"] < 7).any()
