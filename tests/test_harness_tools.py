"""The tool layer must work standalone — no LLM, no pydantic-ai."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import finalize_run, inspect_data, train_tabular_model
from tabular_agent.taskspec import TaskSpec


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "city": rng.choice(["berlin", "hamburg", "koeln"], size=n),
            "y": (x1 + 0.5 * x2 + rng.normal(scale=0.3, size=n) > 0).astype(int),
        }
    )
    data_path = str(tmp_path / "data.csv")
    df.to_csv(data_path, index=False)
    return RunSession(str(tmp_path / "run"), data_path, TaskSpec(target="y", n_splits=3))


def test_inspect_data_resolves_task_and_records_it(session):
    out = inspect_data(session)

    assert out["n_rows"] == 200
    assert out["task_type"] == "binary"
    assert out["target"]["name"] == "y"
    # The string column is reported as unusable rather than silently dropped.
    assert out["non_numeric_columns"] == ["city"]

    assert journal.load(session.run_dir)["task"] == {
        "target": "y",
        "task_type": "binary",
        "metric": out["metric"],
    }


def test_inspect_data_rejects_a_missing_target(tmp_path):
    path = str(tmp_path / "d.csv")
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="nope"))
    with pytest.raises(ValueError, match="not in"):
        inspect_data(s)


def test_train_returns_a_summary_never_a_dataframe(session):
    inspect_data(session)
    out = train_tabular_model(session, model_type="lgbm")

    assert out["experiment_id"] == "e1"
    assert out["n_features"] == 2  # city is excluded
    assert len(out["fold_metrics"]) == 3
    assert isinstance(out["cv_score"], float)
    json.dumps(out)  # the whole summary must survive the wire


def test_train_writes_artifacts_and_a_leaderboard_row(session):
    inspect_data(session)
    out = train_tabular_model(session, model_type="lgbm")

    exp_dir = os.path.join(session.run_dir, out["artifact_path"])
    assert os.path.exists(os.path.join(exp_dir, "final_model.joblib"))
    assert os.path.exists(os.path.join(exp_dir, "oof.parquet"))

    j = journal.load(session.run_dir)
    assert [r["id"] for r in j["leaderboard"]] == ["e1"]
    assert j["best"]["experiment_id"] == "e1"


def test_baseline_is_tracked_separately_from_best(session):
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    train_tabular_model(session, model_type="lgbm")

    j = journal.load(session.run_dir)
    assert j["baseline"]["experiment_id"] == "e1"
    # The baseline must never be selected as the deliverable when a real model exists.
    assert j["best"]["experiment_id"] == "e2"


def test_train_refuses_when_no_numeric_features_remain(tmp_path):
    path = str(tmp_path / "d.csv")
    pd.DataFrame({"city": ["a", "b"] * 20, "y": [0, 1] * 20}).to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=2))
    inspect_data(s)
    with pytest.raises(ValueError, match="no numeric feature columns"):
        train_tabular_model(s)


def test_finalize_packages_the_best_experiment(session):
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    train_tabular_model(session, model_type="lgbm")

    out = finalize_run(session)

    assert out["best_experiment_id"] == "e2"
    assert out["n_experiments"] == 2
    assert os.path.exists(os.path.join(session.run_dir, out["report_path"]))
    assert os.path.exists(
        os.path.join(session.run_dir, out["bundle_dir"], "pipeline.json")
    )


def test_finalize_autotrains_a_baseline_if_the_run_never_did(session):
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm")

    out = finalize_run(session)

    # The report can always answer "did this beat doing nothing?".
    assert out["baseline_score"] is not None
    assert out["best_experiment_id"] == "e1"


def test_finalize_refuses_an_empty_run(session):
    inspect_data(session)
    with pytest.raises(ValueError, match="no model has been trained"):
        finalize_run(session)


def test_bundle_predicts_on_raw_rows(session):
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm")
    out = finalize_run(session)

    from tabular_agent.io import load_table
    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))
    raw = load_table(session.data_path).drop(columns=["y"])
    preds = predict_bundle(bundle, raw)

    assert len(preds) == 200
