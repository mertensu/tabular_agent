"""The fitted ruler: what chooses the shipped table at finalize."""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import finalize_run, inspect_data, transform_features
from tabular_agent.probe import FITTED_RULER_MAX_ROUNDS, fit_fitted_ruler, probe_predict
from tabular_agent.taskspec import TaskSpec

#: Turns a string column holding the signal into a number: a real step.
SIGNAL_CODE = """
def transform_features(df):
    df = df.copy()
    df["s_num"] = df["s"].str[1:].astype(float)
    return df
"""

#: A row-local column with nothing in it about the target: a step inside the noise.
NOISE_CODE = """
def transform_features(df):
    df = df.copy()
    df["junk"] = (df["x1"] * 1e6) % 1.0
    return df
"""


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    x1, s = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "s": "v" + pd.Series(s.round(4)).astype(str),
                       "y": 0.3 * x1 + 2.0 * s + rng.normal(scale=0.5, size=n)})
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)
    return RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3))


def test_it_stops_where_held_back_rows_stop_improving():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=1000), "b[1]": rng.normal(size=1000)})
    y = X["a"] + rng.normal(scale=1.0, size=1000)

    fit = fit_fitted_ruler(X, y, "regression", "rmse")

    assert 1 <= fit.spec["rounds"] < FITTED_RULER_MAX_ROUNDS
    preds = probe_predict(fit, X, "regression")
    assert np.corrcoef(preds, X["a"])[0, 1] > 0.9


@pytest.mark.fitted_ruler
def test_finalize_ships_the_eligible_table_by_the_fitted_ruler(session):
    """The signal step clears 0094's bar under the fitted ruler; the noise step does not."""
    inspect_data(session)
    t1 = transform_features(session, SIGNAL_CODE)
    t2 = transform_features(session, NOISE_CODE, table=t1["table"])

    finalize_run(session)

    chosen = journal.load(session.run_dir)["terminal_selection"]
    assert chosen["table"] == t1["table"]
    assert chosen["chosen"] == "t1"
    assert chosen["eligible"] == ["t1"]
    assert set(chosen["scores"]) == {"work", "t1", "t2"}
    assert chosen["scores"]["t1"] < chosen["scores"]["work"]
    assert "probe_choice" in chosen
    assert t2["table"] != chosen["table"]
