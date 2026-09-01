import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tabular_agent as ta
import tabular_agent.api as public_api
from tabular_agent.inspect import profile_collection


def _tables(tmp_path):
    train = pd.DataFrame(
        {
            "Store": [1, 2, 1, 2],
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "Customers": [10, 20, 11, 21],
            "Sales": [100.0, 200.0, 110.0, 210.0],
        }
    )
    test = train.drop(columns=["Customers", "Sales"]).copy()
    stores = pd.DataFrame({"Store": [1, 2], "Region": ["N", "S"]})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    stores.to_csv(tmp_path / "store.csv", index=False)
    return train, test, stores


def test_inspect_profiles_all_tables_and_only_proposes(tmp_path):
    _tables(tmp_path)
    plan = ta.Agent(model=object()).inspect(tmp_path)

    assert set(plan.tables) == {"store.csv", "test.csv", "train.csv"}
    assert plan.training_table == "train.csv"
    assert plan.serving_table == "test.csv"
    assert plan.target_candidates[0] == "Sales"
    assert plan.ordering_candidates == ["Date"]
    assert plan.proposed_serving_columns == ["Store", "Date"]
    assert plan.serving_columns is None
    assert plan.training_only_columns == ["Customers", "Sales"]
    assert len(plan.joins) == 1
    assert plan.joins[0].right_table == "store.csv"
    assert plan.joins[0].decision == "proposed"


def test_plan_refuses_every_unconfirmed_owner_decision(tmp_path):
    _tables(tmp_path)
    plan = ta.Agent(model=object()).inspect(tmp_path)

    with pytest.raises(ValueError, match="target is unconfirmed"):
        plan.validate()
    plan.target = "Sales"
    with pytest.raises(ValueError, match="serving_columns are unconfirmed"):
        plan.validate()
    plan.serving_columns = list(plan.proposed_serving_columns)
    with pytest.raises(ValueError, match="split_on is unconfirmed"):
        plan.validate()

    plan.split_on = "Date"
    with pytest.raises(ValueError, match="join proposals are unconfirmed"):
        plan.validate()
    plan.reject_joins()
    plan.validate()

    plan.serving_columns.append("Sales")
    with pytest.raises(ValueError, match="must not contain the target"):
        plan.validate()


def test_accepted_static_join_materializes_without_changing_rows(tmp_path):
    train, _, _ = _tables(tmp_path)
    plan = ta.Agent(model=object()).inspect(tmp_path)
    plan.target = "Sales"
    plan.split_on = "Date"
    plan.serving_columns = list(plan.proposed_serving_columns)
    plan.accept_joins()
    plan.serving_columns.extend(["Region"])

    path, manifest = plan.materialize(tmp_path / "run")
    prepared = pd.read_parquet(path)

    assert len(prepared) == len(train)
    assert prepared["Region"].tolist() == ["N", "S", "N", "S"]
    assert pd.api.types.is_datetime64_any_dtype(prepared["Date"])
    assert manifest["joins"][0]["decision"] == "accepted"


def test_bundle_predict_replays_static_join(monkeypatch, tmp_path):
    bundle_dir = tmp_path / "bundle"
    static = bundle_dir / "static"
    static.mkdir(parents=True)
    pd.DataFrame({"Store": [1, 2], "Region": [3.0, 4.0]}).to_csv(
        static / "store.csv", index=False
    )
    (bundle_dir / "api_plan.json").write_text(
        json.dumps(
            {
                "serving_table": "test.csv",
                "split_on": False,
                "joins": [
                    {
                        "right_table": "store.csv",
                        "left_on": "Store",
                        "right_on": "Store",
                        "columns": ["Region"],
                        "decision": "accepted",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(public_api, "load_bundle", lambda _: {"sentinel": True})
    seen = {}

    def fake_predict(bundle, frame, streams=None):
        seen["frame"] = frame
        return frame["Region"].to_numpy()

    monkeypatch.setattr(public_api, "predict_bundle", fake_predict)
    bundle = ta.Bundle(bundle_dir)
    predictions = bundle.predict(pd.DataFrame({"Store": [2, 1]}))

    assert predictions.tolist() == [4.0, 3.0]
    assert seen["frame"]["Store"].tolist() == [2, 1]


def test_agent_run_delegates_to_session_and_finalizer(monkeypatch, tmp_path):
    frame = pd.DataFrame({"x": range(20), "target": np.arange(20, dtype=float)})
    agent = ta.Agent(model=object(), run_root=tmp_path, profile="minimal")
    plan = agent.inspect(frame)
    plan.target = "target"
    plan.serving_columns = ["x"]
    plan.split_on = False
    called = {}

    def fake_run(session, prompt, model):
        called["session"] = session
        called["prompt"] = prompt
        called["model"] = model

    def fake_finalize(session):
        bundle = Path(session.run_dir) / "bundle"
        bundle.mkdir()
        (bundle / "pipeline.json").write_text("{}")
        return {"bundle_dir": "bundle"}

    class FakeRun:
        def __init__(self, run_dir):
            self.run_dir = str(run_dir)

    monkeypatch.setattr(public_api, "run_agent", fake_run)
    monkeypatch.setattr(public_api, "finalize_run", fake_finalize)
    monkeypatch.setattr(public_api, "Run", FakeRun)

    result = agent.run(plan)

    assert Path(result.run_dir).parent == tmp_path
    assert called["session"].profile == "minimal"
    assert called["session"].serving_columns == ["x"]
    assert called["session"].split_on is None
    assert called["model"] is agent.model


def test_public_exports():
    assert ta.Agent is public_api.Agent
    assert ta.DataPlan is public_api.DataPlan
    assert ta.Bundle is public_api.Bundle


def test_facade_minimal_run_finalizes_and_predicts_end_to_end(tmp_path):
    from pydantic_ai.models.test import TestModel

    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 50),
            "target": np.linspace(10.0, 20.0, 50),
        }
    )
    agent = ta.Agent(
        model=TestModel(call_tools=[]),
        run_root=tmp_path,
        profile="minimal",
    )
    plan = agent.inspect(frame)
    plan.target = "target"
    plan.serving_columns = ["x"]
    plan.split_on = False

    run = agent.run(plan)
    predictions = run.bundle.predict(frame[["x"]].iloc[:3])

    assert len(predictions) == 3
    assert np.isfinite(predictions).all()
    assert "#" in run.report()
