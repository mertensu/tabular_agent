"""The tool layer must work standalone — no LLM, no pydantic-ai."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.io import load_table
from tabular_agent.sandbox import SandboxError
from tabular_agent.harness.tools import (
    finalize_run,
    fit_encoding,
    inspect_data,
    train_tabular_model,
    transform_features,
)
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


def test_fit_encoding_makes_a_dropped_column_usable(session):
    inspect_data(session)
    out = fit_encoding(session, ["city"], "frequency")

    assert out["new_columns"] == ["city_freq"]
    assert out["non_numeric_columns"] == ["city"]  # the raw string column stays, unused
    json.dumps(out)

    trained = train_tabular_model(session, table=out["table"])
    assert trained["n_features"] == 3  # x1, x2 + the new encoded column
    assert "city_freq" in load_table(os.path.join(session.run_dir, out["table"])).columns


def test_leaderboard_rows_say_which_table_they_trained_on(session):
    """A resuming agent must not have to open each meta.json to know that."""
    inspect_data(session)
    enc = fit_encoding(session, ["city"], "frequency")
    train_tabular_model(session, model_type="baseline")
    train_tabular_model(session, table=enc["table"], model_type="lgbm")

    rows = journal.load(session.run_dir)["leaderboard"]
    assert rows[0]["source_table"] == session.rel(session.data_path)  # the raw table
    assert rows[1]["source_table"] == enc["table"] == "tables/t1.parquet"


def test_fit_encoding_records_the_table_in_the_journal(session):
    inspect_data(session)
    out = fit_encoding(session, ["city"], "frequency")

    tables = journal.load(session.run_dir)["tables"]
    assert [t["id"] for t in tables] == ["t1"]
    assert tables[0]["path"] == out["table"]
    assert tables[0]["method"] == "frequency"


def test_target_encoding_stores_oof_values_but_a_full_fit_encoder(session):
    """The §5.2 split: honest values to train on, the all-rows encoder to serve."""
    inspect_data(session)
    out = fit_encoding(session, ["city"], "target")

    table = load_table(os.path.join(session.run_dir, out["table"]))
    lineage_path = os.path.join(session.run_dir, "tables", "t1.pipeline.json")
    with open(lineage_path) as f:
        lineage = json.load(f)
    encoder = lineage["steps"][0]["encoder"]
    assert encoder["kind"] == "target"

    # Each row's stored value was fitted without that row's fold; the serving
    # encoder saw every row. They must therefore disagree.
    full_fit = pd.Series(table["city"].map(encoder["mapping"]["city"]))
    assert not np.allclose(table["city_te"], full_fit)


def test_ordinal_encoding_needs_the_order_and_keeps_it(tmp_path):
    path = str(tmp_path / "d.csv")
    rng = np.random.default_rng(0)
    sizes = rng.choice(["small", "medium", "large"], size=60)
    rank = pd.Series(sizes).map({"small": 0, "medium": 1, "large": 2})
    pd.DataFrame({"size": sizes, "y": rank * 2.0 + rng.normal(size=60)}).to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3))
    inspect_data(s)

    with pytest.raises(ValueError, match="requires categories"):
        fit_encoding(s, ["size"], "ordinal")

    out = fit_encoding(
        s, ["size"], "ordinal", categories={"size": ["small", "medium", "large"]}
    )
    table = load_table(os.path.join(s.run_dir, out["table"]))
    assert list(table.loc[table["size"] == "large", "size_ord"].unique()) == [2.0]


def test_encoder_reaches_the_bundle_and_predicts_from_raw_rows(session):
    """The point of the lineage sidecar: raw strings in, prediction out."""
    inspect_data(session)
    enc = fit_encoding(session, ["city"], "frequency")
    train_tabular_model(session, table=enc["table"], model_type="lgbm")
    out = finalize_run(session)

    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle_dir = os.path.join(session.run_dir, out["bundle_dir"])
    bundle = load_bundle(bundle_dir)
    assert [s["type"] for s in bundle["pipeline"].to_dict()["steps"]] == ["encode"]
    assert "city_freq" in bundle["feature_names"]

    # Raw rows: the string column, never the encoded one, and no target.
    raw = load_table(session.data_path).drop(columns=["y"])
    assert len(predict_bundle(bundle, raw)) == 200


def test_fit_encoding_refuses_to_encode_the_target(session):
    inspect_data(session)
    with pytest.raises(ValueError, match="refusing to encode the target"):
        fit_encoding(session, ["y"], "frequency")


RATIO_CODE = """
import numpy as np
def transform_features(df):
    df = df.copy()
    df["ratio"] = df["x1"] / (np.abs(df["x2"]) + 1.0)
    return df
"""


def test_transform_features_derives_a_column_and_trains_on_it(session):
    inspect_data(session)
    out = transform_features(session, RATIO_CODE)

    assert out["new_columns"] == ["ratio"]
    json.dumps(out)

    table = load_table(os.path.join(session.run_dir, out["table"]))
    assert "ratio" in table.columns
    # The target rides along untouched so the new table is trainable on its own.
    assert list(table["y"]) == list(load_table(session.data_path)["y"])

    trained = train_tabular_model(session, table=out["table"])
    assert trained["n_features"] == 3  # x1, x2, ratio


def test_transform_code_cannot_see_the_target(session):
    """No target in the frame at fit time, because there is none at predict time."""
    inspect_data(session)
    code = "def transform_features(df):\n    df = df.copy()\n    df['leak'] = df['y']\n    return df\n"

    # And the failure has to *say* so — a bare KeyError reads like a typo.
    with pytest.raises(SandboxError, match="deliberately not in the frame"):
        transform_features(session, code)


def test_a_transform_may_not_name_a_column_after_the_target(session):
    inspect_data(session)
    code = "def transform_features(df):\n    df = df.copy()\n    df['y'] = df['x1']\n    return df\n"

    with pytest.raises(ValueError, match="which is the target"):
        transform_features(session, code)


def test_transform_records_the_code_not_the_columns(session):
    """The lineage stores the program; the columns are re-derived at predict."""
    inspect_data(session)
    out = transform_features(session, RATIO_CODE)

    with open(os.path.join(session.run_dir, "tables", "t1.pipeline.json")) as f:
        lineage = json.load(f)
    assert [s["type"] for s in lineage["steps"]] == ["transform"]
    assert lineage["steps"][0]["code"] == RATIO_CODE

    row = journal.load(session.run_dir)["tables"][0]
    assert row["method"] == "transform"
    assert row["code"] == RATIO_CODE


def test_transform_stacks_on_an_encoded_table(session):
    """A run branches: encode, then derive features from the encoded table."""
    inspect_data(session)
    enc = fit_encoding(session, ["city"], "frequency")
    code = (
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['x1_per_city'] = df['x1'] / df['city_freq']\n"
        "    return df\n"
    )
    out = transform_features(session, code, table=enc["table"])

    assert out["source_table"] == enc["table"]
    with open(os.path.join(session.run_dir, "tables", "t2.pipeline.json")) as f:
        lineage = json.load(f)
    assert [s["type"] for s in lineage["steps"]] == ["encode", "transform"]


def test_transformed_features_reach_the_bundle_and_predict_from_raw_rows(session):
    inspect_data(session)
    out = transform_features(session, RATIO_CODE)
    train_tabular_model(session, table=out["table"], model_type="lgbm")
    final = finalize_run(session)

    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle = load_bundle(os.path.join(session.run_dir, final["bundle_dir"]))
    assert "ratio" in bundle["feature_names"]

    # Raw rows, no target column: the stored code must run there unchanged.
    raw = load_table(session.data_path).drop(columns=["y"])
    assert len(predict_bundle(bundle, raw)) == 200


def test_transform_must_preserve_the_rows(session):
    """Dropping rows would silently misalign every feature from its label."""
    inspect_data(session)
    code = "def transform_features(df):\n    return df.head(10)\n"

    with pytest.raises(SandboxError, match="preserve the row index"):
        transform_features(session, code)


def _stalled_session(tmp_path, patience=1, max_refusals=3):
    """A run that has already trained twice with no real gain, so it is stalled."""
    from tabular_agent.harness.stop import StopPolicy

    rng = np.random.default_rng(0)
    n = 120
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "y": (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)})
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"),
        path,
        TaskSpec(target="y", n_splits=3),
        stop_policy=StopPolicy(patience=patience, max_refusals=max_refusals),
    )
    inspect_data(s)
    train_tabular_model(s, model_type="baseline")
    train_tabular_model(s, model_type="lgbm")
    # Identical model, identical table: cannot possibly be an improvement.
    train_tabular_model(s, model_type="lgbm")
    return s


def test_a_stalled_run_stops_being_allowed_to_train(tmp_path):
    s = _stalled_session(tmp_path)
    before = len(journal.load(s.run_dir)["leaderboard"])

    out = train_tabular_model(s, model_type="lgbm")

    assert out["trained"] is False
    assert out["next"] == "finalize_run"
    assert "no real improvement" in out["reason"]
    # Refused means refused: no model fitted, no leaderboard row.
    assert len(journal.load(s.run_dir)["leaderboard"]) == before


def test_a_refusal_is_a_return_value_not_an_exception(tmp_path):
    """An exception would burn the retry budget meant for malformed arguments."""
    pytest.importorskip("pydantic_ai")
    from tabular_agent.harness.agent import _retryable

    s = _stalled_session(tmp_path)
    out = _retryable(lambda: train_tabular_model(s, model_type="lgbm"))
    assert out["trained"] is False


def test_every_result_says_where_the_run_stands(session):
    inspect_data(session)
    first = train_tabular_model(session, model_type="lgbm")
    assert first["since_improvement"] == 0
    assert "best result so far" in first["note"]

    second = train_tabular_model(session, model_type="lgbm")
    assert second["since_improvement"] == 1
    assert second["since_improvement_on_table"] == 1
    assert "on this table without an improvement" in second["note"]


def test_an_exhausted_table_is_refused_but_the_run_continues(tmp_path):
    """The redirect: a table is cheap to give up on, a run is not."""
    from tabular_agent.harness.stop import StopPolicy

    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.normal(size=n)
    path = str(tmp_path / "d.csv")
    pd.DataFrame(
        {
            "x1": x1,
            "city": rng.choice(["a", "b", "c"], size=n),
            "y": (x1 + rng.normal(scale=0.4, size=n) > 0).astype(int),
        }
    ).to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"),
        path,
        TaskSpec(target="y", n_splits=3),
        stop_policy=StopPolicy(table_patience=2),
    )
    inspect_data(s)
    for _ in range(3):
        train_tabular_model(s, model_type="lgbm")  # same table, same model

    refused = train_tabular_model(s, model_type="lgbm")
    assert refused["trained"] is False
    assert refused["table_exhausted"] is True
    assert "transform_features" in refused["note"]

    # The run is emphatically not over: new features, and training resumes.
    enc = fit_encoding(s, ["city"], "frequency")
    ok = train_tabular_model(s, table=enc["table"], model_type="lgbm")
    assert ok["trained"] is True


def test_asking_for_the_same_dead_table_over_and_over_ends_the_run(tmp_path):
    from tabular_agent.harness.stop import StopPolicy

    rng = np.random.default_rng(4)
    n = 150
    x1 = rng.normal(size=n)
    path = str(tmp_path / "d.csv")
    pd.DataFrame({"x1": x1, "y": (x1 > 0).astype(int)}).to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"),
        path,
        TaskSpec(target="y", n_splits=3),
        stop_policy=StopPolicy(table_patience=1, max_refusals=2),
    )
    inspect_data(s)
    train_tabular_model(s, model_type="lgbm")
    train_tabular_model(s, model_type="lgbm")

    outs = [train_tabular_model(s, model_type="lgbm") for _ in range(3)]
    assert all(o["trained"] is False for o in outs)
    # Stuck, not redirected — so it ends the way a finished run should: with a bundle.
    assert outs[-1]["harness_finalized"] is True


def test_a_model_that_will_not_stop_gets_the_run_packaged_for_it(tmp_path):
    """The bundle is the deliverable; a loop that ends without one is the failure."""
    s = _stalled_session(tmp_path, max_refusals=2)

    first = train_tabular_model(s, model_type="lgbm")
    assert first["trained"] is False and "bundle_dir" not in first

    second = train_tabular_model(s, model_type="lgbm")
    assert second["harness_finalized"] is True
    assert os.path.exists(os.path.join(s.run_dir, second["bundle_dir"], "pipeline.json"))

    # And now the run really is over.
    with pytest.raises(RuntimeError, match="this run is finished"):
        train_tabular_model(s, model_type="lgbm")


def test_finalize_can_still_train_the_baseline_it_needs(tmp_path):
    """The stop rule must not block the harness from doing its own bookkeeping."""
    from tabular_agent.harness.stop import StopPolicy

    rng = np.random.default_rng(1)
    n = 120
    x1 = rng.normal(size=n)
    path = str(tmp_path / "d.csv")
    pd.DataFrame({"x1": x1, "y": (x1 > 0).astype(int)}).to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"),
        path,
        TaskSpec(target="y", n_splits=3),
        stop_policy=StopPolicy(patience=1),
    )
    inspect_data(s)
    train_tabular_model(s, model_type="lgbm")
    train_tabular_model(s, model_type="lgbm")  # stalls the run, still no baseline

    out = finalize_run(s)
    assert out["baseline_score"] is not None


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


def test_finalize_flags_a_table_that_was_never_trained_on(session):
    """The t5/t6 case: build, extend, train only the child — the link is unmeasured.

    The score of `t6` then cannot be split between what `t5` added and what `t6`
    added, because nothing ever scored `t5`. The run stays valid; the report has
    to stop short of claiming which features earned it.
    """
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")

    t1 = transform_features(session, RATIO_CODE)              # built...
    t2 = fit_encoding(session, ["city"], "frequency", table=t1["table"])  # ...and extended
    train_tabular_model(session, table=t2["table"], model_type="lgbm")

    out = finalize_run(session)
    assert out["unmeasured_tables"] == [t1["table"]]

    report = open(os.path.join(session.run_dir, out["report_path"])).read()
    assert "Attribution gap" in report
    assert "never trained on" in report


def test_a_fully_measured_chain_has_no_gap(session):
    inspect_data(session)
    t1 = transform_features(session, RATIO_CODE)
    train_tabular_model(session, table=t1["table"], model_type="lgbm")
    t2 = fit_encoding(session, ["city"], "frequency", table=t1["table"])
    train_tabular_model(session, table=t2["table"], model_type="lgbm")

    out = finalize_run(session)
    assert out["unmeasured_tables"] == []
    report = open(os.path.join(session.run_dir, out["report_path"])).read()
    assert "Attribution gap" not in report
    assert "How the winning feature set was built" in report


def test_attribution_walks_the_whole_chain(session):
    inspect_data(session)
    t1 = transform_features(session, RATIO_CODE)
    t2 = fit_encoding(session, ["city"], "frequency", table=t1["table"])
    train_tabular_model(session, table=t2["table"], model_type="lgbm")

    j = journal.load(session.run_dir)
    chain = journal.attribution(j, t2["table"])
    assert [s["table"] for s in chain] == [t2["table"], t1["table"]]
    assert chain[0]["measured"] and not chain[1]["measured"]
    # The raw input table is not a derived table and has no row of its own.
    assert journal.ancestry(j, session.rel(session.data_path)) == []


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


def test_an_invented_table_path_lists_the_real_ones(session):
    """The caller is a model that mistyped; show it the menu instead of a stack trace."""
    inspect_data(session)
    fit_encoding(session, ["city"], "frequency")

    with pytest.raises(ValueError, match=r"no such table: 'tables/t9.parquet'.*tables/t1.parquet"):
        train_tabular_model(session, table="tables/t9.parquet")

    assert session.available_tables() == ["tables/t1.parquet"]
