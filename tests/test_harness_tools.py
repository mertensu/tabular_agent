"""The tool layer must work standalone — no LLM, no pydantic-ai."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal, tools
from tabular_agent.harness.session import RunSession
from tabular_agent.io import load_table
from tabular_agent.sandbox import SandboxError
from tabular_agent.harness.tools import (
    explain_table,
    explore_experiment,
    finalize_run,
    fit_encoding,
    inspect_data,
    train_tabular_model,
    transform_features,
)
from tabular_agent.taskspec import TaskSpec


#: How the raw input table names itself in a result. It has no row in `tables`
#: (nothing derived it) and its path is absolute, so results label it the way the
#: agent refers to it: by passing no table at all.
RAW = "the work table"


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

    # 200 rows in, 20% held out for validation (0019) — the agent's world is
    # the work half, and `inspect_data` profiles what it can actually train on.
    assert out["n_rows"] == 160
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
    assert isinstance(out["cv_score"], float)
    json.dumps(out)  # the whole summary must survive the wire

    # The per-fold scores are on disk for the report, not in the payload: the
    # agent acts on the mean and the spread, and this dict ships every call.
    meta_path = os.path.join(session.run_dir, out["artifact_path"], "meta.json")
    with open(meta_path) as f:
        assert len(json.load(f)["fold_metrics"]) == 3


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
    raw = load_table(session.source_path).drop(columns=["y"])
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
    raw = load_table(session.source_path).drop(columns=["y"])
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


#: Every key `train_tabular_model` may return. This dict ships on *every*
#: training call, so it is the one surface where an extra field is paid for
#: dozens of times per run — the same reason the tool list is pinned in
#: test_harness_agent.py. Adding to it should be a decision, not a reflex when
#: something reads confusingly: static explanation belongs in the system prompt,
#: where it is read once.
TRAIN_RESULT_FIELDS = {
    "trained",
    "experiment_id",
    "val_score",
    # The 14th field, added deliberately (0022/0028). It is not derivable — the
    # agent has no bootstrap — and a validation score shipped without its own
    # uncertainty is what the run once used to celebrate a 0.008 gain worth
    # 0.0007. `note` carries the warning; this carries the number every time.
    "val_se",
    "cv_score",
    "fold_spread",
    "model_type",
    "n_features",
    "source_table",
    "dropped_non_numeric",
    "vs_baseline",
    "vs_source_table",
    "columns_by_table",
    "artifact_path",
    "note",
    # Present only when `vs_source_table` is null, to say what would produce it.
    "vs_source_basis",
    # Present only when the validation score could not be computed.
    "val_unavailable",
}


def test_a_training_result_carries_nothing_beyond_its_agreed_fields(session):
    inspect_data(session)
    on_raw = train_tabular_model(session, model_type="lgbm")
    t1 = transform_features(session, RATIO_CODE)
    on_child = train_tabular_model(session, table=t1["table"], model_type="lgbm")

    for out in (on_raw, on_child):
        assert set(out) <= TRAIN_RESULT_FIELDS, set(out) - TRAIN_RESULT_FIELDS

    # The caption is not dead weight on every call: it appears exactly when the
    # gain it explains is missing.
    assert ("vs_source_basis" in on_raw) is (on_raw["vs_source_table"] is None)
    assert ("vs_source_basis" in on_child) is (on_child["vs_source_table"] is None)


def test_every_result_says_where_the_run_stands(session):
    """One prose line, not a set of counters restating it — `note` carries both."""
    inspect_data(session)
    first = train_tabular_model(session, model_type="lgbm")
    assert "best result so far" in first["note"]

    second = train_tabular_model(session, model_type="lgbm")
    assert "1 experiment(s) on this table without an improvement" in second["note"]
    assert "experiments remain in this run's budget" in second["note"]


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


def test_results_are_attributed_per_tool_call_not_per_column(session):
    """Blocks are the unit of decision: the agent cannot add half a tool call."""
    inspect_data(session)
    t1 = transform_features(session, RATIO_CODE)
    t2 = fit_encoding(session, ["city"], "frequency", table=t1["table"])
    out = train_tabular_model(session, table=t2["table"], model_type="lgbm")

    by_table = {b["table"]: b for b in out["columns_by_table"]}
    assert set(by_table) == {RAW, t1["table"], t2["table"]}
    assert by_table[t1["table"]]["n_columns"] == 1  # the "ratio" column
    assert by_table[t2["table"]]["method"] == "frequency"
    # Structure only. Training attaches no per-table score: that is a
    # counterfactual, and answering it costs refits (`explain_table`).
    assert all("contribution" not in b for b in out["columns_by_table"])
    json.dumps(out)


def test_credit_is_a_refit_without_the_columns_not_an_importance(session):
    inspect_data(session)
    t1 = transform_features(session, RATIO_CODE)
    t2 = fit_encoding(session, ["city"], "frequency", table=t1["table"])
    out = explain_table(session, table=t2["table"], model_type="lgbm")

    credit = {b["table"]: b for b in out["credit_by_table"]}
    assert set(credit) == {RAW, t1["table"], t2["table"]}
    # One fit on everything plus one per contributing table withheld.
    assert out["n_fits"] == 4
    # x1/x2 generate y, so withholding them costs the score; the frequency
    # encoding of a column independent of y should be close to free.
    assert credit[RAW]["contribution"] > 0
    assert credit[t2["table"]]["contribution"] < credit[RAW]["contribution"]
    json.dumps(out)


def test_both_halves_of_an_interaction_are_credited_across_separate_calls(tmp_path):
    """The property that replaced "keep interacting features in one call".

    Under importance, a half that does nothing alone scored ~0 and invited the
    agent to abandon it before its partner existed. Leave-one-out cannot make that
    mistake: withholding either half breaks the pair, so both measure positive
    even though they arrived in different tool calls.
    """
    rng = np.random.default_rng(0)
    n = 1200
    a = rng.integers(0, 2, size=n)
    b = rng.integers(0, 2, size=n)
    # The halves must reach the model only via the transforms, so the source
    # columns are strings and get dropped as non-numeric.
    df = pd.DataFrame(
        {
            "noise": rng.normal(size=n),
            "a_str": np.where(a == 1, "on", "off"),
            "b_str": np.where(b == 1, "on", "off"),
            "y": a ^ b,  # pure interaction: neither half alone says anything about y
        }
    )
    data_path = str(tmp_path / "data.csv")
    df.to_csv(data_path, index=False)
    s = RunSession(
        str(tmp_path / "run"), data_path, TaskSpec(target="y", metric="logloss", n_splits=3)
    )

    inspect_data(s)
    half = 'def transform_features(df):\n    df = df.copy()\n    df["{new}"] = (df["{src}"] == "on").astype(int)\n    return df\n'
    t1 = transform_features(s, half.format(new="A", src="a_str"))
    t2 = transform_features(s, half.format(new="B", src="b_str"), table=t1["table"])

    credit = {
        r["table"]: r["contribution"]
        for r in explain_table(s, table=t2["table"], model_type="lgbm")["credit_by_table"]
    }
    assert credit[t1["table"]] > 0.5
    assert credit[t2["table"]] > 0.5


def test_credit_survives_a_context_reset(session):
    """It costs refits, so it is worth persisting rather than re-earning."""
    inspect_data(session)
    t1 = transform_features(session, RATIO_CODE)
    explain_table(session, table=t1["table"], model_type="lgbm")

    row = next(
        t for t in journal.load(session.run_dir)["tables"] if t["path"] == t1["table"]
    )
    assert [b["table"] for b in row["credit_by_table"]] == [RAW, t1["table"]]


def test_a_table_with_one_source_says_so_instead_of_claiming_everything(session):
    """"Without the only columns there are" is not a model; vs_baseline answers it."""
    inspect_data(session)
    out = explain_table(session, model_type="lgbm")

    assert [b["contribution"] for b in out["credit_by_table"]] == [None]
    assert "vs_baseline" in out["credit_by_table"][0]["reason"]
    assert out["n_fits"] == 1


def test_explaining_a_table_records_no_experiment_and_spends_no_budget(session):
    """The measurement must stay usable on a table training has been declined for."""
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm")
    before = journal.load(session.run_dir)

    explain_table(session, model_type="lgbm")

    after = journal.load(session.run_dir)
    assert after["leaderboard"] == before["leaderboard"]
    assert after["best"] == before["best"]


JUNK_CODE = """
def transform_features(df):
    df = df.copy()
    df["junk"] = df["noise"] * 2.0
    return df
"""


def test_credit_is_a_gain_under_either_metric_direction(tmp_path):
    """Same convention as every `vs_...` field: positive means those columns helped.

    Asserted by running the identical ablation under a lower-is-better metric and
    a higher-is-better one: the verdict per table must not depend on which end of
    the metric "good" lives at.
    """
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "noise": rng.normal(size=n),
            "y": (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int),
        }
    )
    data_path = str(tmp_path / "data.csv")
    df.to_csv(data_path, index=False)

    verdicts = {}
    for metric in ("logloss", "auc"):
        session = RunSession(
            str(tmp_path / f"run_{metric}"),
            data_path,
            TaskSpec(target="y", metric=metric, n_splits=3),
        )
        inspect_data(session)
        t1 = transform_features(session, JUNK_CODE)
        out = explain_table(session, table=t1["table"], model_type="lgbm")
        verdicts[metric] = {b["table"]: b["contribution"] for b in out["credit_by_table"]}

    for credit in verdicts.values():
        # x1 lives here and generates y, so withholding these columns wrecks the score.
        assert credit[RAW] > 0
        # A rescaled copy of an irrelevant column: the table does not need it.
        assert credit[t1["table"]] < credit[RAW]

    assert {b: c > 0 for b, c in verdicts["logloss"].items()} == {
        b: c > 0 for b, c in verdicts["auc"].items()
    }


def test_vs_source_table_exists_only_when_the_source_was_measured(session):
    """The marginal number is the reward for having trained the previous step."""
    inspect_data(session)
    t1 = transform_features(session, RATIO_CODE)

    # Parent never trained -> no marginal number available.
    t2 = fit_encoding(session, ["city"], "frequency", table=t1["table"])
    assert train_tabular_model(session, table=t2["table"])["vs_source_table"] is None

    # Train the source, then another table built on it: now it exists.
    train_tabular_model(session, table=t1["table"], model_type="lgbm")
    t3 = fit_encoding(session, ["city"], "target", table=t1["table"])
    assert train_tabular_model(session, table=t3["table"])["vs_source_table"] is not None


def test_vs_source_table_compares_the_same_fit_only(session):
    """Otherwise a model swap is credited to the features built alongside it."""
    inspect_data(session)
    t1 = transform_features(session, RATIO_CODE)
    train_tabular_model(session, table=t1["table"], model_type="lgbm")
    t2 = fit_encoding(session, ["city"], "frequency", table=t1["table"])

    # Same model as the source: the columns are the only difference, so it counts.
    same = train_tabular_model(session, table=t2["table"], model_type="lgbm")
    assert same["vs_source_table"] is not None
    # No caption when the number is there — it would say the same thing every call.
    assert "vs_source_basis" not in same

    # Different model, and different params on the same model, are both
    # different experiments — neither isolates the feature call.
    for kwargs in ({"model_type": "xgb"}, {"model_type": "lgbm", "params": {"n_estimators": 7}}):
        out = train_tabular_model(session, table=t2["table"], **kwargs)
        assert out["vs_source_table"] is None
        assert "not with this exact fit" in out["vs_source_basis"]


def test_a_missing_vs_source_table_always_says_how_to_get_one(session):
    """A bare null teaches nothing; the fix is one call the agent must be told about."""
    inspect_data(session)
    t1 = transform_features(session, RATIO_CODE)
    out = train_tabular_model(session, table=t1["table"], model_type="lgbm")

    assert out["vs_source_table"] is None
    basis = out["vs_source_basis"]
    assert "never been trained" in basis
    assert "explain_table" in basis  # the way out that needs no second fit

    # And it survives the reset, alongside the gain it explains.
    row = next(r for r in journal.load(session.run_dir)["leaderboard"] if r["id"] == out["experiment_id"])
    assert row["vs_source_basis"] == basis


def test_training_on_the_raw_table_has_no_source_to_compare_against(session):
    inspect_data(session)
    out = train_tabular_model(session, model_type="lgbm")
    assert out["vs_source_table"] is None
    assert "not derived from another table" in out["vs_source_basis"]


@pytest.mark.parametrize("metric", ["logloss", "auc"])
def test_every_gain_field_is_positive_when_better_whatever_the_metric(tmp_path, metric):
    """One sign convention for all `vs_*` fields.

    These used to disagree: `vs_baseline` was the raw difference and flipped with
    the metric, while `vs_source_table` was already normalised. A run on `logloss`
    reported the same improvement as -0.19 and +0.19 in one result dict.
    """
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "y": (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)})
    data_path = str(tmp_path / "data.csv")
    df.to_csv(data_path, index=False)
    session = RunSession(
        str(tmp_path / "run"),
        data_path,
        TaskSpec(target="y", metric=metric, n_splits=3),
    )

    # The metric's direction is a run constant, stated once here rather than
    # repeated on every training result.
    higher_is_better = inspect_data(session)["higher_is_better"]
    train_tabular_model(session, model_type="baseline")
    on_raw = train_tabular_model(session, model_type="lgbm")

    # x1 generates y, so lgbm beats the constant baseline under either metric.
    assert on_raw["vs_baseline"] > 0

    t1 = transform_features(
        session,
        'def transform_features(df):\n    df = df.copy()\n    df["x1_sq"] = df["x1"] ** 2\n    return df\n',
    )
    out = train_tabular_model(session, table=t1["table"], model_type="lgbm")
    assert out["vs_baseline"] > 0

    # vs_source_table points the same way as vs_baseline: positive iff this score
    # actually beat the source's, whichever end of the metric "beat" lives at.
    # Read off `val_score`, because that is what `vs_source_table` compares
    # (0019): the leaderboard ranks by the held-out half, so every gain field is
    # a gain in that number, not in the folds.
    beat_source = (
        out["val_score"] > on_raw["val_score"]
        if higher_is_better
        else out["val_score"] < on_raw["val_score"]
    )
    assert (out["vs_source_table"] > 0) is beat_source


def test_columns_by_table_does_not_depend_on_the_model_exposing_importances(session):
    """Provenance is a property of the table, not of whatever was fitted on it."""
    inspect_data(session)
    out = train_tabular_model(session, model_type="baseline")
    assert [b["table"] for b in out["columns_by_table"]] == [RAW]


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
    # The *whole* input file, not the work half — including the validation rows,
    # which no tool in the run ever wrote a feature for. If the bundle can score
    # those, it can score rows that did not exist when the run happened.
    raw = load_table(session.source_path).drop(columns=["y"])
    preds = predict_bundle(bundle, raw)

    assert len(preds) == 200


def test_an_invented_table_path_lists_the_real_ones(session):
    """The caller is a model that mistyped; show it the menu instead of a stack trace."""
    inspect_data(session)
    fit_encoding(session, ["city"], "frequency")

    with pytest.raises(ValueError, match=r"no such table: 'tables/t9.parquet'.*tables/t1.parquet"):
        train_tabular_model(session, table="tables/t9.parquet")

    assert session.available_tables() == ["tables/t1.parquet"]


# --------------------------------------------------------------------- 0019


def test_the_validation_half_is_not_reachable_through_any_tool(session):
    """The guarantee is 'the agent cannot', not 'the agent has no reason to'.

    `available_tables()` never listed the validation half, but listing was only a
    display convention — `resolve_table` used to accept any path that existed,
    absolute ones included. Naming the file directly is the attack, and it is the
    one an agent stumbles into by accident after seeing `data/work.parquet` in a
    leaderboard row.
    """
    inspect_data(session)
    assert os.path.exists(session.val_path)

    for attempt in ("data/val.parquet", session.val_path, "../run/data/val.parquet"):
        with pytest.raises(ValueError, match="no such table"):
            session.resolve_table(attempt)

    assert "data/val.parquet" not in session.available_tables()


def test_the_work_table_resolves_under_the_name_results_give_it(session):
    """Every leaderboard row carries `source_table`, and for the first experiments
    that string *is* the work table's path. A path the agent is shown has to be a
    path the agent can pass back, or the allowlist breaks the obvious next call."""
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm")

    named = journal.load(session.run_dir)["leaderboard"][0]["source_table"]
    assert session.resolve_table(named) == session.data_path


def test_the_split_is_stable_when_a_run_is_resumed(tmp_path):
    """A fresh process on the same run_dir must validate against the same rows.
    If it re-drew the split, every score already on the leaderboard would have
    been measured against different data than every score after the resume."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.integers(0, 2, size=200)})
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)

    first = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y"))
    original = load_table(first.val_path)

    resumed = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y"))
    pd.testing.assert_frame_equal(original, load_table(resumed.val_path))


def test_an_ordered_split_holds_back_the_latest_rows(tmp_path):
    """With a time column, validation is the *future* of work. A random split
    would let the model see past its own prediction horizon."""
    df = pd.DataFrame(
        {
            "day": list(range(200)),
            "x": np.random.default_rng(0).normal(size=200),
            "y": np.random.default_rng(1).integers(0, 2, size=200),
        }
    ).sample(frac=1, random_state=7)  # arrive out of order, as real data does
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)

    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y"), split_on="day")

    assert load_table(s.val_path)["day"].min() > load_table(s.data_path)["day"].max()
    assert s.split_record()["mode"] == "ordered"


def test_the_split_is_recorded_in_the_journal(session):
    """Two runs over the same file are only comparable if they validated against
    the same rows, and the halves live in a run dir that may be deleted."""
    inspect_data(session)

    split = journal.load(session.run_dir)["split"]
    assert (split["n_work"], split["n_val"]) == (160, 40)
    assert split["mode"] == "random"
    assert split["val_fraction"] == 0.2


def test_the_leaderboard_ranks_by_validation_not_by_cv(tmp_path):
    """The whole point of 0019. Written against hand-made rows because the two
    scores disagreeing is the case that matters and it will not happen on demand
    with real fits."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    journal.set_task(run_dir, target="y", task_type="binary", metric="logloss")

    # e1 looks better in the folds; e2 is better on the held-out half.
    journal.append_experiment(
        run_dir,
        {"id": "e1", "model_type": "lgbm", "metric": "logloss",
         "cv_score": 0.30, "val_score": 0.55, "artifact_path": "experiments/e1"},
    )
    journal.append_experiment(
        run_dir,
        {"id": "e2", "model_type": "lgbm", "metric": "logloss",
         "cv_score": 0.45, "val_score": 0.40, "artifact_path": "experiments/e2"},
    )

    assert journal.load(run_dir)["best"]["experiment_id"] == "e2"


def test_a_row_without_a_validation_score_cannot_win(tmp_path):
    """A missing score is a fact about the run, so the row stays on the
    leaderboard — but nothing unranked is allowed to be the deliverable."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    journal.set_task(run_dir, target="y", task_type="binary", metric="logloss")

    journal.append_experiment(
        run_dir,
        {"id": "e1", "model_type": "lgbm", "metric": "logloss",
         "cv_score": 0.50, "val_score": 0.50, "artifact_path": "experiments/e1"},
    )
    journal.append_experiment(
        run_dir,
        {"id": "e2", "model_type": "lgbm", "metric": "logloss", "cv_score": 0.01,
         "val_score": None, "val_unavailable": "lineage replay failed",
         "artifact_path": "experiments/e2"},
    )

    j = journal.load(run_dir)
    assert [r["id"] for r in j["leaderboard"]] == ["e1", "e2"]
    assert j["best"]["experiment_id"] == "e1"


def test_finalize_run_happens_once(session):
    """`finalize_run` picks the winner against the validation rows. Letting it run
    twice makes that a query loop, which is the thing the split exists to bound."""
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm")
    finalize_run(session)

    with pytest.raises(ValueError, match="already been finalized"):
        finalize_run(session)


def test_training_after_finalize_is_refused_even_with_budget_left(session):
    """The finalized check used to sit *after* an early return that fired whenever
    the stop rule said keep going — so an agent could finalize early, keep
    training, and finalize again against the same rows."""
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm")
    finalize_run(session)

    # The stop rule is perfectly happy here: one experiment, budget untouched.
    with pytest.raises(RuntimeError, match="this run is finished"):
        train_tabular_model(session, model_type="lgbm", params={"n_estimators": 40})


def test_a_validation_score_rides_on_every_training_result(session):
    """It is the number the run ranks by, so the agent has to see it — and
    `cv_score` stays beside it, because the gap is the signal that a gain lived
    only in the folds."""
    inspect_data(session)
    out = train_tabular_model(session, model_type="lgbm")

    assert isinstance(out["val_score"], float)
    assert isinstance(out["cv_score"], float)
    assert "val_unavailable" not in out


# ---------------------------------------------------------- explore_experiment


@pytest.fixture
def regression_session(tmp_path):
    """A regression run whose error is concentrated in one segment of a string
    column the model cannot use — the shape `explore_experiment` exists to find."""
    rng = np.random.default_rng(7)
    n = 300
    x1 = rng.normal(size=n)
    segment = rng.choice(["easy", "hard"], size=n)
    # `hard` rows carry noise the features cannot explain, so the model's error
    # lives there and nowhere else.
    noise = np.where(segment == "hard", rng.normal(scale=5.0, size=n), 0.0)
    df = pd.DataFrame({"x1": x1, "segment": segment, "y": 3 * x1 + noise})
    data_path = str(tmp_path / "reg.csv")
    df.to_csv(data_path, index=False)
    return RunSession(str(tmp_path / "run"), data_path, TaskSpec(target="y", n_splits=3))


def _explore(session, exp_id, body):
    return explore_experiment(session, exp_id, f"def explore(df):\n{body}")


def test_the_frame_carries_the_error_and_the_columns_the_model_could_not_use(
    regression_session,
):
    """The string column is the point. A model cannot train on `segment`, which is
    exactly why grouping residuals by it is how the agent learns to encode it."""
    inspect_data(regression_session)
    exp = train_tabular_model(regression_session, model_type="lgbm")

    out = _explore(regression_session, exp["experiment_id"],
                   "    return sorted(df.columns)")
    columns = out["result"]
    assert {"x1", "segment", "y", "oof_pred", "residual", "fold"} <= set(columns)

    out = _explore(
        regression_session, exp["experiment_id"],
        "    return df.groupby('segment')['residual'].apply(lambda s: float(s.abs().mean())).to_dict()",
    )
    errors = out["result"]
    assert errors["hard"] > 3 * errors["easy"]


def test_rows_line_up_with_the_folds_they_were_held_out_in(session):
    """The strongest available check that the join is not scrambled: recomputing
    the metric per fold from the frame must reproduce the fold scores the
    experiment recorded. Alignment here is positional, so a silent off-by-one
    would be invisible in every other assertion."""
    from tabular_agent.metrics import compute_metric

    inspect_data(session)
    exp = train_tabular_model(session, model_type="lgbm")
    with open(os.path.join(session.run_dir, exp["artifact_path"], "meta.json")) as f:
        meta = json.load(f)

    out = _explore(
        session, exp["experiment_id"],
        "    return {int(k): [list(v['y']), list(v['oof_pred'])]\n"
        "            for k, v in df.groupby('fold')[['y', 'oof_pred']].agg(list).to_dict('index').items()}",
    )
    recomputed = [
        compute_metric(meta["metric"], np.array(y), np.array(p))
        for _, (y, p) in sorted(out["result"].items())
    ]
    assert recomputed == pytest.approx(meta["fold_metrics"], abs=1e-9)


def test_it_sees_the_work_half_only(session):
    """200 rows in, 160 in the work half. The validation rows have no route here:
    the frame is built from the experiment's own source table."""
    inspect_data(session)
    exp = train_tabular_model(session, model_type="lgbm")

    out = _explore(session, exp["experiment_id"], "    return len(df)")
    assert out["result"] == 160


def test_printed_output_comes_back(session):
    inspect_data(session)
    exp = train_tabular_model(session, model_type="lgbm")

    out = _explore(session, exp["experiment_id"],
                   "    print('fold sizes:', df['fold'].value_counts().to_dict())\n"
                   "    return 'ok'")
    assert out["result"] == "ok"
    assert "fold sizes:" in out["stdout"]


def test_a_quiet_call_carries_no_empty_stdout_field(session):
    """0018: a field that reads the same on every call is not payload."""
    inspect_data(session)
    exp = train_tabular_model(session, model_type="lgbm")

    assert "stdout" not in _explore(session, exp["experiment_id"], "    return 1")


def test_unknown_experiment_names_the_real_ones(session):
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm")

    with pytest.raises(ValueError, match="no such experiment") as exc:
        _explore(session, "e99", "    return 1")
    assert "'e1'" in str(exc.value)


def test_it_costs_nothing_and_records_nothing(session):
    """Free like `explain_table`: it produces no model to ship, so charging it
    against the experiment budget would price looking at a failure like causing
    one."""
    inspect_data(session)
    exp = train_tabular_model(session, model_type="lgbm")
    before = journal.load(session.run_dir)

    for _ in range(5):
        _explore(session, exp["experiment_id"], "    return float(df['residual'].std())")

    after = journal.load(session.run_dir)
    assert after["leaderboard"] == before["leaderboard"]
    assert after["stop"] == before["stop"]


def test_it_still_works_once_the_run_has_stopped(session):
    """The moment the agent most needs to ask what went wrong is after being
    told to stop, so a refusal must not take the diagnostic with it."""
    inspect_data(session)
    exp = train_tabular_model(session, model_type="lgbm")
    finalize_run(session)

    assert _explore(session, exp["experiment_id"], "    return len(df)")["result"] == 160


def test_nothing_survives_between_calls(session):
    """There is no kernel. State is `run_dir`; a name bound in one call is gone
    in the next, which is what keeps a resumed run identical to a fresh one."""
    inspect_data(session)
    exp = train_tabular_model(session, model_type="lgbm")

    _explore(session, exp["experiment_id"], "    global stash\n    stash = 1\n    return 1")
    with pytest.raises(SandboxError):
        _explore(session, exp["experiment_id"], "    return stash")


def test_exploration_cannot_train_a_model(session):
    inspect_data(session)
    exp = train_tabular_model(session, model_type="lgbm")

    with pytest.raises(SandboxError, match="train_tabular_model"):
        _explore(session, exp["experiment_id"], "    import lightgbm\n    return 1")


# ------------------------------------------------------------ target transforms


@pytest.fixture
def priced_session(tmp_path):
    """A strictly positive, right-skewed target — the shape where fitting on the
    log scale is the point, and the only shape log1p/sqrt are defined on."""
    rng = np.random.default_rng(5)
    n = 300
    area = rng.uniform(30, 160, n)
    grade = rng.choice(["low", "mid", "high"], n)
    level = {"low": 700, "mid": 1400, "high": 2400}
    price = area * np.array([level[g] for g in grade]) * rng.lognormal(0, 0.25, n)
    df = pd.DataFrame({"area": area, "grade": grade, "y": price})
    path = str(tmp_path / "priced.csv")
    df.to_csv(path, index=False)
    return RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3))


def test_the_transform_is_recorded_and_the_scores_stay_comparable(priced_session):
    """Two experiments differing only by target scale must land on one leaderboard
    with scores that mean the same thing, or the run cannot choose between them."""
    inspect_data(priced_session)
    raw = train_tabular_model(priced_session, model_type="lgbm")
    log = train_tabular_model(priced_session, model_type="lgbm", target_transform="log1p")

    rows = journal.load(priced_session.run_dir)["leaderboard"]
    assert [r["target_transform"] for r in rows] == ["none", "log1p"]
    # Same metric, same rows, same units — within a factor of a few of each other
    # rather than the ~10x gap a log-scale score would show.
    assert 0.2 < raw["val_score"] / log["val_score"] < 5


def test_an_impossible_transform_is_a_correction_not_a_dead_run(session):
    """`session`'s target is a 0/1 class label. The message has to name the fix,
    because it arrives as a ModelRetry and is all the agent gets."""
    inspect_data(session)
    with pytest.raises(ValueError, match="regression only"):
        train_tabular_model(session, model_type="lgbm", target_transform="log1p")


def test_vs_source_table_does_not_compare_across_target_scales(priced_session):
    """0017's rule: a comparison must differ by exactly one thing. A log fit on a
    child table measured against a raw fit on its source would credit the
    transform to the columns."""
    inspect_data(priced_session)
    enc = fit_encoding(priced_session, ["grade"], "frequency")
    train_tabular_model(priced_session, model_type="lgbm")  # raw, on the source
    child = train_tabular_model(
        priced_session, table=enc["table"], model_type="lgbm", target_transform="log1p"
    )
    assert child["vs_source_table"] is None
    assert "not with this exact fit" in child["vs_source_basis"]


def test_the_bundle_predicts_in_the_original_units(priced_session):
    """The silent failure: a bundle that forgets the inverse returns log-dollars,
    and every value still looks like a plausible number."""
    from tabular_agent.pipeline import load_bundle, predict_bundle

    inspect_data(priced_session)
    train_tabular_model(priced_session, model_type="lgbm", target_transform="log1p")
    out = finalize_run(priced_session)

    raw_rows = load_table(priced_session.source_path).drop(columns=["y"])
    preds = predict_bundle(
        load_bundle(os.path.join(priced_session.run_dir, out["bundle_dir"])), raw_rows
    )
    truth = load_table(priced_session.source_path)["y"]
    assert preds.mean() > 0.3 * truth.mean()
    assert np.corrcoef(preds, truth)[0, 1] > 0.5


# -------------------------------------------------------------- explore_journal


def _journal_code(body):
    return f"def explore(lb, tables, run):\n{body}"


def test_the_leaderboard_arrives_as_a_frame(session):
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    enc = fit_encoding(session, ["city"], "frequency")
    train_tabular_model(session, table=enc["table"], model_type="lgbm")

    out = tools.explore_journal(session, _journal_code(
        "    return {'n': len(lb), 'models': sorted(lb['model_type']),\n"
        "            'tables': sorted(tables['method']), 'metric': run['task']['metric']}"
    ))
    assert out["result"]["n"] == 2
    assert out["result"]["models"] == ["baseline", "lgbm"]
    assert out["result"]["tables"] == ["frequency"]


def test_an_empty_run_hands_over_a_well_formed_frame(session):
    """A fresh run must not fail on a KeyError for a column that exists — the
    first question an agent asks is often asked before it has trained anything."""
    inspect_data(session)
    out = tools.explore_journal(session, _journal_code(
        "    return {'rows': len(lb), 'has_col': 'val_score' in lb.columns,\n"
        "            'filtered': len(lb[lb['val_score'] < 1])}"
    ))
    assert out["result"] == {"rows": 0, "has_col": True, "filtered": 0}


def test_params_group_instead_of_raising(session):
    """A dict cell is unhashable, so a params column of dicts would make the
    obvious query — group by hyper-parameters — an error rather than an answer."""
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm", params={"n_estimators": 30})
    train_tabular_model(session, model_type="lgbm", params={"n_estimators": 30})
    train_tabular_model(session, model_type="lgbm", params={"n_estimators": 60})

    out = tools.explore_journal(session, _journal_code(
        "    return {k: int(v) for k, v in lb.groupby('params').size().to_dict().items()}"
    ))
    assert sorted(out["result"].values()) == [1, 2]


def test_it_answers_the_question_no_single_result_can(session):
    """Which encoding method actually paid — a question spanning every experiment,
    which is the reason this tool exists rather than reading rows one at a time."""
    inspect_data(session)
    freq = fit_encoding(session, ["city"], "frequency")
    tgt = fit_encoding(session, ["city"], "target")
    train_tabular_model(session, table=freq["table"], model_type="lgbm")
    train_tabular_model(session, table=tgt["table"], model_type="lgbm")

    out = tools.explore_journal(session, _journal_code(
        "    m = lb.merge(tables, left_on='source_table', right_on='path', how='left')\n"
        "    return m.groupby('method')['val_score'].mean().round(4).to_dict()"
    ))
    assert set(out["result"]) == {"frequency", "target"}


def test_the_data_dictionary_reaches_the_agent(tmp_path):
    """The only channel by which the *meaning* of a column reaches the agent.

    Motivated by Rossmann: `Customers` correlates 0.895 with the target, is
    absent from the competition's test set, and is therefore unusable at predict
    time. Every guard in this harness holds while a model trains on it, and the
    validation half — drawn from the same file — confirms the gain. Nothing
    computable from the training data says otherwise; only the data's owner does.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "store": rng.integers(1, 20, n),
        "customers": rng.integers(50, 500, n),
        "promo": rng.integers(0, 2, n),
    })
    df["sales"] = df["customers"] * 8 + df["promo"] * 300
    path = tmp_path / "rossmann_like.csv"
    df.to_csv(path, index=False)

    spec = TaskSpec(
        target="sales",
        notes="One row per store per day. Rows where the store was closed score 0 "
              "and are excluded from the real metric.",
        column_notes={
            "customers": "Number of customers that day. NOT available at predict "
                         "time — it is not in the test set.",
            "store": "Store identifier, 1115 of them. A real categorical, not a row number.",
        },
    )
    out = inspect_data(RunSession(str(tmp_path / "run"), str(path), spec))

    # Per-column, beside the dtype and cardinality the reader is already looking at.
    assert "NOT available at predict time" in out["columns"]["customers"]["description"]
    assert "not a row number" in out["columns"]["store"]["description"]
    # A column nobody described carries no description key at all — an empty one
    # would be a line of payload per column saying nothing.
    assert "description" not in out["columns"]["promo"]
    # And what has no column to attach to.
    assert "excluded from the real metric" in out["about_this_data"]


def test_no_dictionary_means_no_extra_payload(session):
    """The common case is a dataset nobody documented, and it must cost nothing."""
    out = inspect_data(session)
    assert "about_this_data" not in out
    assert all("description" not in c for c in out["columns"].values())


def test_a_resumed_agent_is_told_the_documentation_exists(tmp_path):
    """The prompt now tells a resumed episode to skip `inspect_data`, so without
    this line the data dictionary is unreachable by an agent that was not there."""
    import numpy as np
    import pandas as pd
    from tabular_agent.harness.brief import render_brief

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    df["y"] = df["a"] * 3 + rng.normal(0, 0.1, 200)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)

    plain = RunSession(str(tmp_path / "r1"), str(path), TaskSpec(target="y"))
    documented = RunSession(str(tmp_path / "r2"), str(path),
                            TaskSpec(target="y", column_notes={"a": "the useful one"}))
    for s in (plain, documented):
        inspect_data(s)
        train_tabular_model(s, model_type="baseline")

    assert "documentation" not in render_brief(plain)
    assert "documentation" in render_brief(documented)


def test_every_experiment_carries_the_uncertainty_of_its_own_score(session):
    """0022's fix: the yardstick has to measure the quantity being ranked, and
    that quantity is `val_score`, so its standard error has to be on the row."""
    inspect_data(session)
    out = train_tabular_model(session, model_type="lgbm")

    row = journal.load(session.run_dir)["leaderboard"][-1]
    assert row["val_se"] is not None and row["val_se"] > 0
    # On the row and in the artifact, like every other score.
    with open(os.path.join(session.run_dir, row["artifact_path"], "meta.json")) as f:
        assert json.load(f)["val_se"] == row["val_se"]
    assert out["val_score"] is not None


def test_the_uncertainty_is_reproducible(session):
    """It gates the stop rule, so a noise estimate that moved between identical
    runs would make the run's control flow non-deterministic."""
    inspect_data(session)
    a = train_tabular_model(session, model_type="lgbm", params={"n_estimators": 20})
    b = train_tabular_model(session, model_type="lgbm", params={"n_estimators": 20})

    rows = journal.load(session.run_dir)["leaderboard"]
    by_id = {r["id"]: r for r in rows}
    assert by_id[a["experiment_id"]]["val_se"] == by_id[b["experiment_id"]]["val_se"]


def _panel(tmp_path, n_days=40, n_stores=6):
    """A store-day panel: many rows share each date, as in Rossmann."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", periods=n_days, freq="D")
    df = pd.DataFrame([
        {"date": d, "store": s, "promo": int(rng.integers(0, 2))}
        for d in days for s in range(n_stores)
    ])
    df["y"] = 100 + df["store"] * 10 + df["promo"] * 5 + np.arange(len(df)) * 0.01
    path = tmp_path / "panel.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_the_split_never_puts_one_date_on_both_sides(tmp_path):
    """Rows sharing a date share the promo calendar, the holiday, the weather. A
    cut at an exact row count put 603 of one Rossmann day's stores in the work
    half and 512 of the same day in validation — the model learns that day's
    level from one group and applies it to the other."""
    path = _panel(tmp_path)
    session = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y"),
                         split_on="date", val_fraction=0.2)

    work = load_table(session.data_path)
    val = load_table(session.val_path)
    assert not set(work["date"]) & set(val["date"])
    assert work["date"].max() < val["date"].min()


def test_cross_validation_forecasts_when_the_run_does(tmp_path):
    """The inner folds have to ask the question the outer split asks. Left
    random, `cv_score` measures interpolation while `val_score` measures
    forecasting, and their disagreement looks like overfitting rather than what
    it is: two different tasks."""
    import pandas as pd

    path = _panel(tmp_path)
    session = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y"),
                         split_on="date", val_fraction=0.2)
    inspect_data(session)
    out = train_tabular_model(session, model_type="lgbm", params={"n_estimators": 20})

    oof = pd.read_parquet(
        os.path.join(session.run_dir, out["artifact_path"], "oof.parquet")
    )
    # The earliest block has no prediction — nothing precedes it to train on —
    # and it must arrive as NaN, not as a zero that reads like a huge residual.
    assert oof.iloc[:, 0].isna().any()
    assert not oof.iloc[:, 0].isna().all()


def test_a_table_that_drops_the_ordering_column_is_refused(tmp_path):
    """Falling back to random folds would silently change the evaluation protocol
    mid-run, which produces a confident number nobody can interpret afterwards."""
    path = _panel(tmp_path)
    session = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y"),
                         split_on="date", val_fraction=0.2)
    inspect_data(session)
    dropped = transform_features(session, "def transform_features(df):\n"
                                          "    return df.drop(columns=['date'])")

    with pytest.raises(ValueError, match="split on 'date'"):
        train_tabular_model(session, table=dropped["table"], model_type="lgbm")


def test_no_uncertainty_is_claimed_on_an_ordered_split(tmp_path):
    """The bootstrap assumes the validation rows are exchangeable draws from one
    population. Under `split_on` they are a *later period*, chosen because it
    differs from the training rows — resampling inside it measures the wrong
    variation and would come out confidently too small.

    A too-small yardstick is worse than none, because the stop rule believes it.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    day = np.arange(n)
    df = pd.DataFrame({
        "day": day,
        "x": rng.normal(size=n),
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })
    df["y"] = df["x"] * 2 + day * 0.01 + rng.normal(0, 0.1, n) + 10
    path = tmp_path / "series.csv"
    df.to_csv(path, index=False)

    ordered = RunSession(str(tmp_path / "ordered"), str(path),
                         TaskSpec(target="y"), split_on="date")
    inspect_data(ordered)
    train_tabular_model(ordered, model_type="lgbm")
    row = journal.load(ordered.run_dir)["leaderboard"][-1]
    assert row["val_score"] is not None
    assert row["val_se"] is None
    assert journal.load(ordered.run_dir)["split"]["mode"] == "ordered"

    # The same data split at random does get one — it is the ordering that makes
    # the estimate invalid, not anything about this dataset.
    shuffled = RunSession(str(tmp_path / "random"), str(path), TaskSpec(target="y"))
    inspect_data(shuffled)
    train_tabular_model(shuffled, model_type="lgbm")
    assert journal.load(shuffled.run_dir)["leaderboard"][-1]["val_se"] is not None


def test_no_uncertainty_rather_than_a_meaningless_one(tmp_path):
    """A validation half too small to resample gets `None`, not a number. The
    stop rule falls back to `fold_spread`; a fabricated standard error would be
    trusted exactly as much as a real one."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 40  # 20% of 40 is 8 validation rows, under the floor
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = df["a"] * 2 + rng.normal(0, 0.1, n)
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)

    tiny = RunSession(str(tmp_path / "run"), str(path), TaskSpec(target="y"))
    inspect_data(tiny)
    train_tabular_model(tiny, model_type="lgbm")

    row = journal.load(tiny.run_dir)["leaderboard"][-1]
    assert row["val_score"] is not None
    assert row["val_se"] is None


def test_a_table_reports_the_columns_it_added(session):
    """The `tables` frame read a `columns` key that journal rows never carry, so
    every table reported zero columns added — a wrong answer, not an error, to
    the one question the frame exists to answer."""
    inspect_data(session)
    enc = fit_encoding(session, ["city"], "frequency")

    out = tools.explore_journal(session, "def explore(tables):\n"
                                "    r = tables.iloc[0]\n"
                                "    return {'n': int(r['n_columns']), 'cols': list(r['columns'])}")
    assert out["result"]["n"] > 0
    assert out["result"]["cols"] == journal.load(session.run_dir)["tables"][0]["new_columns"]
    assert enc["table"] == journal.load(session.run_dir)["tables"][0]["path"]


def test_it_sees_no_data_at_all(session):
    """The whole reason it needs no leakage argument. There is no frame here
    carrying a row of the dataset, and no name that could reach one."""
    inspect_data(session)
    train_tabular_model(session, model_type="lgbm")

    with pytest.raises(SandboxError) as exc:
        tools.explore_journal(session, "def explore(df):\n    return len(df)")
    msg = str(exc.value)
    assert "'df'" in msg and "lb" in msg and "tables" in msg


def test_it_costs_nothing_and_works_after_finalize(session):
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    train_tabular_model(session, model_type="lgbm")
    finalize_run(session)
    before = journal.load(session.run_dir)

    out = tools.explore_journal(session, _journal_code(
        "    print('best is', run['best']['experiment_id'])\n"
        "    return len(lb)"
    ))
    assert out["result"] == 2
    assert "best is e2" in out["stdout"]
    assert journal.load(session.run_dir) == before
