"""Agent-written code is run on the rows the bundle will predict, and what ships is named.

Code written against the training rows can fail on rows the bundle will be asked
to predict, because those can hold values the training rows never did (a later
date, a year whose holidays were never computed). The run replays each code step
on them as it is built, while the agent can still fix it.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tabular_agent as ta
from tabular_agent.api import _warn_if_serving_rows_fail
from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import _shipped_table, finalize_run, transform_features
from tabular_agent.taskspec import TaskSpec

# Training days are 0..499 and serving days 600..649. The lookup covers what the
# code's author saw, the way a table of holidays built from the training years does.
FAILS_LATER = (
    "def transform_features(df):\n"
    "    df = df.copy()\n"
    "    seen = {d: 1 for d in range(500)}\n"
    "    df['seen_day'] = df['day'].map(seen).astype(int)\n"
    "    return df\n"
)
MISSING_LATER = FAILS_LATER.replace(".astype(int)", "")
WORKS = (
    "def transform_features(df):\n"
    "    df = df.copy()\n"
    "    df['day_mod_7'] = df['day'] % 7\n"
    "    return df\n"
)


def _session(tmp_path, *, with_serving=True, profile="full"):
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
    serving_path = None
    if with_serving:
        prepared = tmp_path / "run" / "prepared"
        prepared.mkdir(parents=True)
        serving = pd.DataFrame({
            "store": np.tile(["s1", "s2", "s3", "s4"], 50),
            "day": 600 + np.arange(200) // 4,
            "x": rng.normal(size=200),
        })
        serving_path = str(prepared / "serving.parquet")
        serving.to_parquet(serving_path, index=False)
    return RunSession(
        str(tmp_path / "run"), path,
        TaskSpec(target="sales", task_type="regression", metric="rmse", n_splits=3,
                 forecast_horizon=1),
        split_on="day", serving_path=serving_path, profile=profile,
    )


def test_code_that_raises_only_on_the_serving_rows_is_refused(tmp_path):
    session = _session(tmp_path)

    with pytest.raises(ValueError, match="raised on the rows this model will be asked to predict"):
        transform_features(session, FAILS_LATER)

    assert journal.load(session.run_dir).get("tables") in (None, [])
    assert not (tmp_path / "run" / "tables" / "t1.parquet").exists()


def test_the_refusal_names_the_serving_rows_range(tmp_path):
    session = _session(tmp_path)

    with pytest.raises(ValueError, match="200 rows, day 600 to 649"):
        transform_features(session, FAILS_LATER)


def test_code_that_goes_missing_on_the_serving_rows_is_reported_not_refused(tmp_path):
    session = _session(tmp_path)

    out = transform_features(session, MISSING_LATER)

    assert out["table"] == "tables/t1.parquet"
    assert out["serving_rows"]["columns_missing_more_often"] == {
        "seen_day": {"missing_on_training_rows": 0.0, "missing_on_serving_rows": 1.0},
    }


def test_code_that_works_on_the_serving_rows_reports_nothing(tmp_path):
    session = _session(tmp_path)

    out = transform_features(session, WORKS)

    assert "serving_rows" not in out


def test_without_serving_rows_nothing_is_checked(tmp_path):
    session = _session(tmp_path, with_serving=False)

    out = transform_features(session, FAILS_LATER)

    assert out["table"] == "tables/t1.parquet"


def test_a_later_step_is_checked_on_serving_rows_replayed_through_its_parent(tmp_path):
    session = _session(tmp_path)
    parent = transform_features(session, WORKS)
    uses_parent = FAILS_LATER.replace("df['day'].map", "(df['day'] + df['day_mod_7'] * 0).map")

    with pytest.raises(ValueError, match="asked to predict"):
        transform_features(session, uses_parent, table=parent["table"])


def test_no_tool_can_load_the_serving_rows(tmp_path):
    session = _session(tmp_path)

    with pytest.raises(ValueError):
        session.resolve_table("prepared/serving.parquet")


def test_materialize_writes_the_serving_rows_without_target_or_identifiers(tmp_path):
    train = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4],
        "valid": [0.1, 0.2, 0.3, 0.4],
        "store_id": [10, 10, 11, 11],
        "target": [0, 1, 0, 1],
    })
    train.to_csv(tmp_path / "train.csv", index=False)
    train.drop(columns="target").to_csv(tmp_path / "test.csv", index=False)
    plan = ta.Agent(model=object()).inspect(tmp_path)
    plan.target = "target"
    plan.serving_columns = list(plan.proposed_serving_columns)
    plan.split_on = False
    plan.accept_identifier_proposals()

    plan.materialize(tmp_path / "run")
    serving = pd.read_parquet(tmp_path / "run" / "prepared" / "serving.parquet")

    assert sorted(serving.columns) == ["store_id", "valid"]
    assert len(serving) == 4


def test_a_bundle_that_cannot_predict_the_serving_table_is_reported(tmp_path):
    (tmp_path / "run_report.md").write_text("# Run Report\n")

    def refuse(_rows):
        raise KeyError("2023")

    run = SimpleNamespace(run_dir=str(tmp_path), bundle=SimpleNamespace(predict=refuse))
    plan = SimpleNamespace(serving_table="test.csv", training_table="train.csv",
                           tables={"test.csv": {"path": str(tmp_path / "test.csv")}})

    with pytest.warns(UserWarning, match="could not predict the serving table"):
        _warn_if_serving_rows_fail(run, plan)

    assert "Serving table cannot be predicted" in (tmp_path / "run_report.md").read_text()


def test_the_shipped_table_is_named_with_the_probe_choice_it_overrode():
    j = {
        "tables": [
            {"id": "t1", "path": "tables/t1.parquet", "source_table": "data/work.parquet",
             "method": "transform"},
            {"id": "t2", "path": "tables/t2.parquet", "source_table": "tables/t1.parquet",
             "method": "history"},
        ],
        "terminal_selection": {"table": "tables/t1.parquet", "chosen": "t1",
                               "probe_choice": "t2", "scores": {"t1": 3.004, "t2": 3.058}},
    }

    out = _shipped_table(None, j, "tables/t1.parquet")

    assert out["table"] == "t1"
    assert out["steps"] == ["t1: transform"]
    assert out["probe_choice"] == "t2"
    assert out["fitted_scores"] == {"t1": 3.004, "t2": 3.058}
    assert "t1, not t2" in out["note"]


def test_the_shipped_table_needs_no_note_when_the_probe_agreed():
    j = {"tables": [{"id": "t1", "path": "tables/t1.parquet",
                     "source_table": "data/work.parquet", "method": "transform"}],
         "terminal_selection": {"probe_choice": "t1", "scores": {"t1": 3.0}}}

    out = _shipped_table(None, j, "tables/t1.parquet")

    assert out == {"table": "t1", "steps": ["t1: transform"]}


def test_finalize_names_the_table_that_shipped(tmp_path):
    session = _session(tmp_path, with_serving=False, profile="minimal")
    transform_features(session, WORKS)

    out = finalize_run(session)

    tables = {t["id"] for t in journal.load(session.run_dir)["tables"]}
    assert out["shipped_table"]["table"] in tables | {"work"}
