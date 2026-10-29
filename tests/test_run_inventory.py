"""The first `finalize_run` the agent asks for returns what the run built, once."""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.stop import StopPolicy
from tabular_agent.harness.tools import (
    finalize_run,
    fit_encoding,
    run_inventory,
    transform_features,
)
from tabular_agent.taskspec import TaskSpec


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    store = rng.choice(["s1", "s2", "s3", "s4"], size=n)
    df = pd.DataFrame({
        "store": store,
        "day": np.arange(n) // 4,
        "x": rng.normal(size=n),
        "sales": pd.Series(store).map({"s1": 10.0, "s2": 20.0, "s3": 30.0, "s4": 40.0})
        + rng.normal(size=n),
    })
    path = str(tmp_path / "panel.csv")
    df.to_csv(path, index=False)
    return RunSession(
        str(tmp_path / "run"), path,
        TaskSpec(target="sales", task_type="regression", metric="rmse", n_splits=3,
                 forecast_horizon=1),
        split_on="day",
    )


def test_the_inventory_is_shown_once_and_finishes_nothing(session):
    out = fit_encoding(session, columns=["store"], method="target")
    transform_features(session, "def transform_features(df):\n    return df\n",
                       table=out["table"])

    inventory = run_inventory(session)

    assert inventory["finalized"] is False
    assert inventory["would_ship"]["table"] == "t1"
    assert inventory["would_ship"]["steps"] == ["t1: encoding"]
    assert inventory["tables_by_kind"]["encoding"] == {"built": 1, "passed": 1}
    assert inventory["tables_by_kind"]["transform"]["passed"] == 0
    assert "fit_history" in inventory["tools_never_called"]
    assert "fit_encoding" not in inventory["tools_never_called"]
    assert not journal.stop_state(session.run_dir).get("finalized")
    assert journal.stop_state(session.run_dir)["inventory_shown"] == {"tables": 2}
    # Shown once: the next request to finish goes through.
    assert run_inventory(session) is None


def test_a_passed_table_nothing_was_built_on_is_listed(session):
    fit_encoding(session, columns=["store"], method="target")

    inventory = run_inventory(session)

    # The table that would ship is not a loose end; it is the end.
    assert inventory["passed_but_not_built_on"] == []


def test_no_inventory_once_the_stop_rule_has_ended_the_search(session):
    session.stop_policy = StopPolicy(max_tables=1)
    fit_encoding(session, columns=["store"], method="target")

    assert run_inventory(session) is None


def test_no_inventory_after_finalizing(session):
    fit_encoding(session, columns=["store"], method="target")
    finalize_run(session)

    assert run_inventory(session) is None


def test_the_journal_says_whether_anything_was_built_after_the_inventory(session):
    fit_encoding(session, columns=["store"], method="target")
    run_inventory(session)
    fit_encoding(session, columns=["store"], method="frequency")
    finalize_run(session)

    state = journal.stop_state(session.run_dir)
    assert state["inventory_shown"] == {"tables": 1}
    assert state["tables_at_finalize"] == 2
