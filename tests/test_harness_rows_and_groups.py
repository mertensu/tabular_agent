"""`filter_rows` and `fit_group_stats` through the harness, end to end.

The unit tests beside this file check the two mechanisms in isolation. What
matters here is what happens to a *run*: whether the validation score is still
measured on every held-back row, and whether the bundle can predict rows that
were never in the training file's shape.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.tools import (
    _load_lineage,
    filter_rows,
    finalize_run,
    fit_group_stats,
    inspect_data,
    train_tabular_model,
    transform_features,
)
from tabular_agent.harness.session import RunSession
from tabular_agent.io import load_table
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.taskspec import TaskSpec

KEEP_OPEN = "def keep_rows(df):\n    return df['is_open'] == 1\n"


@pytest.fixture
def panel(tmp_path):
    """A store-day panel: a group key, a closed-days regime, a post-hoc column.

    Shaped to exercise all three things at once — `traffic` is measured on the
    day and so is not available when predicting it, `is_open` marks rows the
    model should not learn from, and `store` is the group key both depend on.
    """
    rng = np.random.default_rng(0)
    n = 300
    store = rng.choice(["s1", "s2", "s3", "s4"], size=n)
    level = pd.Series(store).map({"s1": 10.0, "s2": 20.0, "s3": 30.0, "s4": 40.0})
    is_open = rng.random(n) > 0.2
    traffic = level * 3 + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame(
        {
            "store": store,
            "day": np.arange(n) // 4,
            "is_open": is_open.astype(int),
            "traffic": np.where(is_open, traffic, 0.0),
            "sales": np.where(is_open, level + rng.normal(scale=1.0, size=n), 0.0),
        }
    )
    path = str(tmp_path / "panel.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def session(tmp_path, panel):
    return RunSession(
        str(tmp_path / "run"),
        panel,
        TaskSpec(target="sales", task_type="regression", metric="rmse", n_splits=3),
        # `traffic` is measured on the day being predicted, so it is in the
        # training file and not in a serving request.
        serving_columns=["store", "day", "is_open"],
    )


@pytest.fixture
def plain_session(tmp_path, panel):
    """The same data with no serving schema, for tests about filtering alone.

    Without this the unservable-column refusal fires first and every filter test
    would be measuring that instead.
    """
    return RunSession(
        str(tmp_path / "plain"),
        panel,
        TaskSpec(target="sales", task_type="regression", metric="rmse", n_splits=3),
    )


# ------------------------------------------------------------------- filters


def test_a_filter_writes_a_smaller_table_and_leaves_its_source_alone(plain_session):
    inspect_data(plain_session)
    before = len(load_table(plain_session.data_path))

    out = filter_rows(plain_session, KEEP_OPEN)

    assert out["n_rows_before"] == before
    assert out["n_dropped"] > 0
    assert len(load_table(plain_session.resolve_table(out["table"]))) == out["n_rows"]
    # Immutability (§ the system prompt's promise): the source is untouched.
    assert len(load_table(plain_session.data_path)) == before


def test_the_validation_score_still_covers_every_held_back_row(plain_session):
    """The reason a filter must not be replayed, asserted rather than argued.

    If the filter ran during scoring, the model would be graded on the ~80% of
    validation rows that survive it — a different, easier exam than the one an
    unfiltered experiment sat, and the two `val_score`s would not be comparable.
    So the exact path the scorer takes, replaying the filtered table's lineage
    over the validation half, has to come back with every row it started with.
    """
    inspect_data(plain_session)
    filtered = filter_rows(plain_session, KEEP_OPEN)["table"]
    path = plain_session.resolve_table(filtered)

    val = load_table(plain_session.val_path)
    replayed = _load_lineage(path, "regression").transform(val.drop(columns=["sales"]))

    assert len(replayed) == len(val)

    exp = train_tabular_model(plain_session, table=filtered, model_type="lgbm")
    assert exp["trained"] is True
    assert exp["val_score"] is not None


def test_a_bundle_from_a_filtered_table_predicts_every_row(plain_session):
    """Serving: a model that declines to answer for some rows is not a model."""
    inspect_data(plain_session)
    filtered = filter_rows(plain_session, KEEP_OPEN)["table"]
    train_tabular_model(plain_session, table=filtered, model_type="lgbm")
    out = finalize_run(plain_session)

    bundle = load_bundle(os.path.join(plain_session.run_dir, out["bundle_dir"]))
    serving = load_table(plain_session.val_path).drop(columns=["sales"])
    preds = predict_bundle(bundle, serving)

    # Including the closed days the model was never trained on — they were kept
    # out of the fit, not out of the world.
    assert len(preds) == len(serving)
    assert not np.isnan(preds).any()


def test_a_predicate_that_drops_nothing_is_refused(plain_session):
    inspect_data(plain_session)
    with pytest.raises(ValueError, match="kept all"):
        filter_rows(plain_session, "def keep_rows(df):\n    return df['is_open'] >= 0\n")


def test_a_predicate_that_drops_almost_everything_is_refused(plain_session):
    inspect_data(plain_session)
    with pytest.raises(ValueError, match="below the"):
        filter_rows(plain_session, "def keep_rows(df):\n    return df.index < 3\n")


def test_the_journal_records_a_filter_as_a_filter(plain_session):
    inspect_data(plain_session)
    out = filter_rows(plain_session, KEEP_OPEN)

    row = journal.load(plain_session.run_dir)["tables"][-1]
    assert row["method"] == "filter"
    # Present and empty rather than absent: every reader of `tables` looks for it.
    assert row["new_columns"] == []
    assert row["n_dropped"] == out["n_dropped"]


def test_a_filter_can_be_stacked_under_later_feature_work(session):
    """The filter is a table like any other, so building on it must work."""
    inspect_data(session)
    filtered = filter_rows(session, KEEP_OPEN)["table"]

    stats = fit_group_stats(session, by=["store"], value="traffic", table=filtered)
    dropped = transform_features(
        session,
        "def transform_features(df):\n"
        "    return df.drop(columns=['traffic'], errors='ignore')\n",
        table=stats["table"],
    )
    exp = train_tabular_model(session, table=dropped["table"], model_type="lgbm")

    assert exp["trained"] is True
    assert exp["val_score"] is not None
    # The group statistic is credited to the table that produced it, not folded
    # into the filter's (empty) contribution or the raw table's columns.
    credited = {t["table"]: t["n_columns"] for t in exp["columns_by_table"]}
    assert credited[stats["table"]] == 1


# -------------------------------------------------------------- group stats


def test_group_stats_makes_an_unavailable_column_usable_as_history(session):
    """The capability, stated as the run experiences it.

    `traffic` cannot be trained on — it will not exist at predict time and the
    harness refuses it. Its *per-store history* can be, because the lookup needs
    only `store`.
    """
    inspect_data(session)
    stats = fit_group_stats(session, by=["store"], value="traffic", stat="mean")

    # Training on the table as it stands is still refused: `traffic` itself is
    # in the feature matrix.
    refused = train_tabular_model(session, table=stats["table"], model_type="lgbm")
    assert refused["trained"] is False
    assert refused["unservable"] is True

    dropped = transform_features(
        session,
        "def transform_features(df):\n"
        "    return df.drop(columns=['traffic'], errors='ignore')\n",
        table=stats["table"],
    )
    exp = train_tabular_model(session, table=dropped["table"], model_type="lgbm")

    assert exp["trained"] is True
    trained_on = load_table(session.resolve_table(dropped["table"]))
    assert "store_traffic_mean" in trained_on.columns
    assert "traffic" not in trained_on.columns


def test_the_note_says_how_to_make_the_table_trainable(session):
    inspect_data(session)
    out = fit_group_stats(session, by=["store"], value="traffic", stat="mean")

    assert "errors='ignore'" in out["note"]
    assert out["new_columns"] == ["store_traffic_mean"]
    assert out["n_groups"] == 4


def test_the_stored_statistics_replay_on_rows_that_never_had_the_value(session):
    """The serving path, through the lineage rather than the encoder directly."""
    inspect_data(session)
    stats = fit_group_stats(session, by=["store"], value="traffic", stat="mean")
    dropped = transform_features(
        session,
        "def transform_features(df):\n"
        "    return df.drop(columns=['traffic'], errors='ignore')\n",
        table=stats["table"],
    )
    train_tabular_model(session, table=dropped["table"], model_type="lgbm")
    # This test is about replaying the terminal lineage, not whether the
    # synthetic group statistic earns selection. State that precondition under
    # the 0090 terminal contract.
    state = journal.load(session.run_dir)
    terminal = next(row for row in state["tables"] if row["path"] == dropped["table"])
    terminal["probe_vs_source_interval"] = {
        "gain": 1.0,
        "paired_se": 0.01,
        "mode": "paired_bootstrap",
    }
    journal.save(session.run_dir, state)
    out = finalize_run(session)

    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))
    serving = load_table(session.val_path).drop(columns=["sales", "traffic"])
    preds = predict_bundle(bundle, serving)

    assert len(preds) == len(serving)
    assert not np.isnan(preds).any()


def test_grouping_the_target_points_at_the_tool_that_does_it_properly(session):
    inspect_data(session)
    with pytest.raises(ValueError, match="fit_encoding"):
        fit_group_stats(session, by=["store"], value="sales", stat="mean")


def test_a_group_key_that_will_not_exist_is_refused_before_the_fit(session):
    """Refused here rather than three calls later as an unservable model."""
    inspect_data(session)
    with pytest.raises(ValueError, match="cannot group by"):
        fit_group_stats(session, by=["traffic"], value="day", stat="mean")


def test_a_column_that_is_not_there_is_named(session):
    inspect_data(session)
    with pytest.raises(ValueError, match="columns not in"):
        fit_group_stats(session, by=["store"], value="nope", stat="mean")


def test_the_out_of_fold_values_differ_from_the_fitted_ones(session):
    """The table trains on held-out values; the lineage ships the all-rows fit.

    Same arrangement as out-of-fold target encoding, and the same reason: if the
    two were equal, every row would be seeing a statistic it helped compute.
    """
    inspect_data(session)
    stats = fit_group_stats(session, by=["store"], value="traffic", stat="mean")

    table = load_table(session.resolve_table(stats["table"]))
    from tabular_agent.harness.tools import _load_lineage

    replayed = _load_lineage(session.resolve_table(stats["table"]), "regression").transform(
        table.drop(columns=["sales", "store_traffic_mean"])
    )

    assert not np.allclose(
        table["store_traffic_mean"], replayed["store_traffic_mean"], equal_nan=True
    )
