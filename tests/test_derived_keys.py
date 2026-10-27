"""A key column derived by a stored step exists at prediction time; one derived from
a column that will not exist does not.

Group statistics, history, stream aggregates and future-covariate windows refuse a
key that cannot be looked up at prediction time. That has to be decided by
replaying the table's lineage on serving-shaped rows, not by comparing names
against the serving columns: almost every useful key is derived.
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    fit_group_stats,
    fit_history,
    inspect_data,
    transform_features,
)
from tabular_agent.taskspec import TaskSpec

WEEKDAY = """
import pandas as pd

def transform_features(df):
    df = df.copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    return df
"""
FROM_VISITS = """
def transform_features(df):
    df = df.copy()
    df["busy"] = (df["visits"] > 50).astype(int)
    return df
"""


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=200)
    df = pd.DataFrame({"shop": np.repeat(["a", "b"], len(days)), "date": np.tile(days, 2)})
    df["visits"] = rng.integers(20, 80, len(df))          # recorded, not known when predicting
    df["y"] = 100 + 5 * df["date"].dt.dayofweek + rng.normal(0, 3, len(df))
    path = tmp_path / "shops.csv"
    df.to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), str(path),
                   TaskSpec(target="y", n_splits=3, forecast_horizon=14.0),
                   split_on="date", workflow="forecast", serving_columns=["shop", "date"])
    inspect_data(s)
    return s


def test_a_key_derived_from_serving_columns_is_accepted(session):
    derived = transform_features(session, WEEKDAY)
    out = fit_history(session, ["shop", "dow"], "y", windows=[[56, "mean"]],
                      table=derived["table"])
    assert "shop_dow_y_56_mean" in out["new_columns"]
    assert fit_group_stats(session, ["dow"], "date", "count",
                           table=derived["table"])["new_columns"]


def test_a_key_derived_from_a_missing_column_is_still_refused(session):
    derived = transform_features(session, FROM_VISITS)
    with pytest.raises(ValueError, match="busy"):
        fit_history(session, ["shop", "busy"], "y", windows=[[56, "mean"]],
                    table=derived["table"])
