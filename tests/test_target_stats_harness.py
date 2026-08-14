"""`fit_encoding(stats=...)` through the harness, end to end.

The unit tests beside this file check the encoder in isolation. What matters
here is what happens to a *run*: whether the extra columns reach the table, and
— the one that would be silent if it broke — whether a model trained on them can
still be served, given that every one of them was computed from a column that
does not exist at predict time.
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import fit_encoding, inspect_data, train_tabular_model
from tabular_agent.io import load_table
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.taskspec import TaskSpec

ALL_STATS = ["mean", "std", "median", "count"]


@pytest.fixture
def panel(tmp_path):
    """Categories that differ in spread and in size, not only in average.

    `wide` and `tight` share a target mean, so the mean alone cannot tell them
    apart; `rare` has two rows, so its estimate should not be read like the
    others'.
    """
    rng = np.random.default_rng(0)
    rows = []
    for cat, mean, spread, n in [
        ("tight", 50.0, 1.0, 120),
        ("wide", 50.0, 25.0, 120),
        ("high", 90.0, 5.0, 120),
        ("rare", 70.0, 5.0, 2),
    ]:
        for _ in range(n):
            rows.append({"cat": cat, "noise": rng.normal(), "y": rng.normal(mean, spread)})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    path = str(tmp_path / "panel.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def session(tmp_path, panel):
    return RunSession(
        str(tmp_path / "run"),
        panel,
        TaskSpec(target="y", task_type="regression", metric="rmse", n_splits=3),
        serving_columns=["cat", "noise"],
    )


def test_one_call_writes_every_requested_statistic(session):
    inspect_data(session)
    out = fit_encoding(session, columns=["cat"], method="target", stats=ALL_STATS)

    assert out["new_columns"] == ["cat_te", "cat_te_std", "cat_te_median", "cat_te_count"]
    table = load_table(session.resolve_table(out["table"]))
    for name in out["new_columns"]:
        assert name in table.columns


def test_the_default_is_still_the_mean_alone(session):
    inspect_data(session)
    out = fit_encoding(session, columns=["cat"], method="target")
    assert out["new_columns"] == ["cat_te"]


def test_the_journal_records_which_statistics_were_asked_for(session):
    """Two target encodings of the same column differ only by this."""
    inspect_data(session)
    fit_encoding(session, columns=["cat"], method="target")
    fit_encoding(session, columns=["cat"], method="target", stats=["mean", "count"])

    rows = [r for r in journal.load(session.run_dir).get("tables", []) if r.get("method") == "target"]
    assert [r["target_stats"] for r in rows] == [["mean"], ["mean", "count"]]


def test_spread_separates_categories_the_mean_cannot(session):
    """The point of the slice, asserted on data rather than argued."""
    inspect_data(session)
    out = fit_encoding(session, columns=["cat"], method="target", stats=["mean", "std"])
    table = load_table(session.resolve_table(out["table"]))

    by_cat = table.groupby("cat")[["cat_te", "cat_te_std"]].mean()
    assert by_cat.loc["tight", "cat_te"] == pytest.approx(by_cat.loc["wide", "cat_te"], abs=3.0)
    assert by_cat.loc["tight", "cat_te_std"] < by_cat.loc["wide", "cat_te_std"] / 2


def test_count_is_the_trust_signal_a_thin_category_gets(session):
    inspect_data(session)
    out = fit_encoding(session, columns=["cat"], method="target", stats=["count"])
    table = load_table(session.resolve_table(out["table"]))
    counts = table.groupby("cat")["cat_te_count"].max()
    assert counts["rare"] < counts["tight"]


def test_a_model_trained_on_them_can_still_be_served(session):
    """The failure that would otherwise be silent.

    Every one of these columns is computed from the target, which by definition
    is not in a serving request. They survive only because the fitted statistics
    live in the table's lineage and the lookup reads `cat` alone — so this
    asserts the whole path: train, package, then predict from rows that have no
    target column at all.
    """
    inspect_data(session)
    encoded = fit_encoding(session, columns=["cat"], method="target", stats=ALL_STATS)
    trained = train_tabular_model(session, table=encoded["table"], model_type="lgbm")
    assert trained.get("unservable") is not True
    assert trained["val_score"] is not None

    from tabular_agent.harness.tools import finalize_run

    finalize_run(session)
    bundle = load_bundle(session.ctx.path("bundle", "_").rsplit("/", 1)[0])

    raw = load_table(session.data_path).drop(columns=["y"]).head(20)
    preds = predict_bundle(bundle, raw)
    assert len(preds) == 20
    assert np.isfinite(preds).all()


def test_a_category_unseen_at_predict_time_gets_a_defined_value_for_every_statistic(session):
    inspect_data(session)
    encoded = fit_encoding(session, columns=["cat"], method="target", stats=ALL_STATS)
    lineage_of = __import__(
        "tabular_agent.harness.tools", fromlist=["_load_lineage"]
    )._load_lineage
    pipeline = lineage_of(session.resolve_table(encoded["table"]), "regression")

    unseen = pd.DataFrame({"cat": ["never-seen"], "noise": [0.0]})
    out = pipeline.transform(unseen)
    for name in ["cat_te", "cat_te_std", "cat_te_median", "cat_te_count"]:
        assert np.isfinite(out[name].iloc[0]), name
    # Unseen means zero for a count, and the population value for the rest.
    assert out["cat_te_count"].iloc[0] == 0.0
    assert out["cat_te_count"].iloc[0] != out["cat_te_std"].iloc[0]


def test_stats_is_refused_on_a_method_that_never_reads_the_target(session):
    inspect_data(session)
    with pytest.raises(ValueError, match="method='target' only"):
        fit_encoding(session, columns=["cat"], method="frequency", stats=["count"])


def test_an_unknown_statistic_is_refused_before_anything_is_written(session):
    inspect_data(session)
    before = len(journal.load(session.run_dir).get("tables", []))
    with pytest.raises(ValueError, match="unknown target statistic"):
        fit_encoding(session, columns=["cat"], method="target", stats=["mean", "variance"])
    assert len(journal.load(session.run_dir).get("tables", [])) == before
