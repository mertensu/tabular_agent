import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tabular_agent as ta
import tabular_agent.api as public_api
from tabular_agent.encoders import FrequencyEncoder
from tabular_agent.pipeline import Pipeline
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


def test_identifier_role_is_proposed_confirmed_and_removed_before_modeling(tmp_path):
    train = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "valid": [0.1, 0.2, 0.3, 0.4],
            "store_id": [10, 10, 11, 11],
            "target": [0, 1, 0, 1],
        }
    )
    train.to_csv(tmp_path / "train.csv", index=False)
    train.drop(columns="target").to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame(
        {"PassengerId": [1, 2, 3, 4], "Tier": ["A", "B", "A", "B"]}
    ).to_csv(tmp_path / "passengers.csv", index=False)

    plan = ta.Agent(model=object()).inspect(tmp_path)

    assert plan.proposed_identifier_columns == ["PassengerId"]
    assert plan.identifier_columns is None
    plan.target = "target"
    plan.serving_columns = list(plan.proposed_serving_columns)
    plan.split_on = False
    plan.accept_joins()
    plan.serving_columns.append("Tier")
    with pytest.raises(ValueError, match="identifier_columns are unconfirmed"):
        plan.validate()

    plan.accept_identifier_proposals()
    plan.validate()
    prepared_path, manifest = plan.materialize(tmp_path / "run")
    prepared = pd.read_parquet(prepared_path)

    assert "PassengerId" not in prepared
    assert "valid" in prepared and "store_id" in prepared
    assert prepared["Tier"].tolist() == ["A", "B", "A", "B"]
    assert manifest["identifier_columns"] == ["PassengerId"]


def test_inspect_picks_up_a_data_description_into_notes(tmp_path):
    _tables(tmp_path)
    (tmp_path / "data_description.txt").write_text(
        "Store: the store id\nSales: daily revenue (target)\n"
    )
    # A generic README is a weaker candidate than a named dictionary.
    (tmp_path / "README.md").write_text("repo readme, not the data dictionary")

    plan = ta.Agent(model=object()).inspect(tmp_path)

    assert plan.notes_source == "data_description.txt"
    assert plan.notes.startswith("Store: the store id")
    assert plan.description_candidates == ["data_description.txt", "README.md"]
    # It is surfaced, with the losing candidate and how to override.
    text = plan.summary()
    assert "loaded from data_description.txt" in text
    assert "README.md" in text


def test_inspect_without_a_description_leaves_notes_empty(tmp_path):
    _tables(tmp_path)
    plan = ta.Agent(model=object()).inspect(tmp_path)
    assert plan.notes is None
    assert plan.notes_source is None
    assert plan.description_candidates == []


def test_plan_summary_tracks_open_decisions(tmp_path):
    _tables(tmp_path)
    plan = ta.Agent(model=object()).inspect(tmp_path)

    fresh = plan.summary()
    # Unconfirmed decisions are shown as open, with their proposals to guide the owner.
    assert "still to confirm" in fresh
    assert "[ ] target" in fresh
    assert "[ ] serving_columns" in fresh
    assert "[ ] split_on" in fresh
    assert "[ ] joins" in fresh
    assert "Sales" in fresh  # a target candidate is surfaced

    plan.target = "Sales"
    plan.serving_columns = list(plan.proposed_serving_columns)
    plan.split_on = "Date"
    plan.accept_joins()

    done = plan.summary()
    assert "all decisions confirmed" in done
    assert "[ ]" not in done  # nothing left open
    assert "agent.run(plan)" in done


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
                "target": "target",
                "serving_table": "test.csv",
                "split_on": False,
                "identifier_columns": ["RowId"],
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
    predictions = bundle.predict(
        pd.DataFrame({"RowId": [20, 10], "Store": [2, 1], "target": [9, 8]})
    )

    assert predictions.tolist() == [4.0, 3.0]
    assert seen["frame"]["Store"].tolist() == [2, 1]
    assert "RowId" not in seen["frame"]
    assert "target" not in seen["frame"]


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
    assert ta.capability_backlog is not None


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

    training = run.training_features()
    assert training.columns.tolist() == ["x", "target"]
    assert len(training) == 40

    assert len(predictions) == 3
    assert np.isfinite(predictions).all()
    assert "#" in run.report()
    assert "<!doctype html>" in run.report("html")
    assert run.report_path().name == "run_report.html"
    report_data = json.loads(run.report("json"))
    assert report_data["summary"]["validation_score"] is not None
    assert report_data["serving_contract"]["target"] == "target"
    assert report_data["serving_contract"]["serving_columns"] == ["x"]
    assert report_data["features"]["catalog"][0]["feature"] == "x"


def test_bundle_describes_and_replays_learned_features(monkeypatch, tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "api_plan.json").write_text(
        json.dumps({
            "target": "target",
            "serving_columns": ["x", "category"],
            "identifier_columns": ["RowId"],
            "joins": [],
            "split_on": False,
        })
    )
    raw = pd.DataFrame(
        {
            "RowId": [9, 8],
            "x": [2.0, 3.0],
            "category": ["a", "b"],
            "target": [10.0, 20.0],
        },
        index=[7, 3],
    )
    pipeline = Pipeline("regression")
    pipeline.add_transform(
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['double_x'] = df['x'] * 2\n"
        "    return df\n",
        new_columns=["double_x"],
    )
    pipeline.add_encoder(FrequencyEncoder(["category"]).fit(raw))
    payload = {
        "pipeline": pipeline,
        "models": [],
        "weights": np.array([]),
        "feature_names": ["x", "double_x", "category_freq"],
        "task_type": "regression",
        "target_transform": "none",
        "streams": {},
        "training_table": None,
    }
    monkeypatch.setattr(public_api, "load_bundle", lambda _: payload)

    bundle = ta.Bundle(bundle_dir)
    matrix = bundle.feature_table(raw)
    enriched = bundle.feature_table(raw, include_input=True)
    descriptions = bundle.describe_features().set_index("feature")

    assert matrix.index.tolist() == [7, 3]
    assert matrix.columns.tolist() == bundle.feature_names
    assert matrix["double_x"].tolist() == [4.0, 6.0]
    assert matrix["category_freq"].tolist() == [0.5, 0.5]
    assert enriched[["RowId", "target"]].equals(raw[["RowId", "target"]])
    pd.testing.assert_frame_equal(enriched[bundle.feature_names], matrix)
    assert descriptions.loc["x", "kind"] == "input"
    assert descriptions.loc["double_x", "kind"] == "row_transform"
    assert descriptions.loc["category_freq", "kind"] == "frequency"
    assert descriptions.loc["category_freq", "source_columns"] == ["category"]
    assert "df['double_x']" in descriptions.loc["double_x", "definition"]


def test_run_training_features_reads_oof_table_instead_of_replaying(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    bundle_dir = run_dir / "bundle"
    table_dir = run_dir / "tables"
    bundle_dir.mkdir(parents=True)
    table_dir.mkdir()
    (bundle_dir / "api_plan.json").write_text(json.dumps({"target": "target"}))
    fitted = pd.DataFrame(
        {"x": [1.0, 2.0], "group_te": [0.25, 0.75], "target": [0, 1]}
    )
    fitted.to_parquet(table_dir / "winner.parquet", index=False)
    payload = {
        "pipeline": Pipeline("binary"),
        "models": [],
        "weights": np.array([]),
        "feature_names": ["x", "group_te"],
        "task_type": "binary",
        "target_transform": "none",
        "streams": {},
        "training_table": "tables/winner.parquet",
    }
    monkeypatch.setattr(public_api, "load_bundle", lambda _: payload)

    run = ta.Run(run_dir)
    exported = run.training_features()

    pd.testing.assert_frame_equal(exported, fitted)
    assert run.training_features(include_target=False).columns.tolist() == ["x", "group_te"]
