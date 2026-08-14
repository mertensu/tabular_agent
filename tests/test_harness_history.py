"""`fit_history` through the harness, end to end.

The unit tests beside this file check the mechanism in isolation. What matters
here is what happens to a *run*: whether the ordering column the run is split on
is the one history is frozen along, whether the bundle can answer rows dated
after every row it was fitted on, and whether the refusals fire where a run would
otherwise get a number that means nothing.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    finalize_run,
    fit_history,
    train_tabular_model,
)
from tabular_agent.io import load_table
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.taskspec import TaskSpec


@pytest.fixture
def panel(tmp_path):
    """A store-day panel with a per-store level and a slow drift in time."""
    rng = np.random.default_rng(0)
    days = np.arange(120)
    stores = ["s1", "s2", "s3", "s4"]
    level = {"s1": 10.0, "s2": 20.0, "s3": 30.0, "s4": 40.0}
    rows = []
    for d in days:
        for s in stores:
            rows.append(
                {
                    "store": s,
                    "day": int(d),
                    "promo": int(rng.random() > 0.7),
                    "sales": level[s] + 0.1 * d + rng.normal(scale=1.0),
                }
            )
    path = str(tmp_path / "panel.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def session(tmp_path, panel):
    return RunSession(
        str(tmp_path / "run"),
        panel,
        TaskSpec(target="sales", task_type="regression", metric="rmse", n_splits=3),
        split_on="day",
    )


@pytest.fixture
def unordered_session(tmp_path, panel):
    return RunSession(
        str(tmp_path / "run_unordered"),
        panel,
        TaskSpec(target="sales", task_type="regression", metric="rmse", n_splits=3),
    )


def test_it_adds_the_pair_and_records_the_ordering_column(session):
    out = fit_history(session, by=["store"], value="sales")

    assert out["new_columns"] == ["store_sales_last", "store_sales_last_age"]
    row = journal.load(session.run_dir)["tables"][-1]
    assert row["method"] == "history"
    assert row["order_column"] == "day"
    assert row["value_column"] == "sales"


def test_the_target_may_be_the_value_column(session):
    """The difference from `fit_group_stats`, which refuses the target outright.

    A per-group *statistic* of the target is target encoding and has its own tool.
    A per-group *most recent* target is not, and cannot be smuggled out of a
    served frame either, because the target is not in one.
    """
    out = fit_history(session, by=["store"], value="sales")

    assert "store_sales_last" in out["new_columns"]


def test_an_unordered_run_is_refused(unordered_session):
    """0048: without an ordering column there is no past to freeze history at."""
    with pytest.raises(ValueError, match="not split on an ordering column"):
        fit_history(unordered_session, by=["store"], value="sales")


def test_the_first_period_has_no_value_and_the_tool_says_so(session):
    out = fit_history(session, by=["store"], value="sales")

    assert out["n_rows_without_value"] > 0
    assert "first period" in out["note"]


def test_a_key_that_will_not_exist_at_predict_time_is_refused(tmp_path, panel):
    session = RunSession(
        str(tmp_path / "run_serving"),
        panel,
        TaskSpec(target="sales", task_type="regression", metric="rmse", n_splits=3),
        split_on="day",
        # `promo` is not in a serving request, so history keyed on it could never
        # be looked up.
        serving_columns=["store", "day"],
    )

    with pytest.raises(ValueError, match="cannot key history on"):
        fit_history(session, by=["store", "promo"], value="sales")


def test_an_iso_date_column_read_back_from_csv_works(tmp_path, panel):
    """The normal case: every table the harness loads has its dates as text, so
    the ordering column arrives as a string rather than a datetime."""
    df = pd.read_csv(panel)
    df["date"] = (
        pd.Timestamp("2015-01-01") + pd.to_timedelta(df["day"], unit="D")
    ).dt.strftime("%Y-%m-%d")
    dated = str(tmp_path / "dated.csv")
    df.drop(columns=["day"]).to_csv(dated, index=False)

    session = RunSession(
        str(tmp_path / "run_dates"),
        dated,
        TaskSpec(target="sales", task_type="regression", metric="rmse", n_splits=3),
        split_on="date",
    )

    out = fit_history(session, by=["store"], value="sales")

    table = load_table(session.resolve_table(out["table"]))
    seen = table["store_sales_last_age"].notna()
    # Days, not the nanosecond integers a naive conversion would hand back.
    assert table.loc[seen, "store_sales_last_age"].max() < 365


def test_offsets_need_the_task_to_declare_its_horizon(session):
    with pytest.raises(ValueError, match="forecast_horizon"):
        fit_history(session, by=["store"], value="sales", offsets=[3])


def test_a_declared_horizon_makes_a_long_enough_offset_legal(tmp_path, panel):
    session = RunSession(
        str(tmp_path / "run_horizon"),
        panel,
        TaskSpec(
            target="sales", task_type="regression", metric="rmse", n_splits=3,
            forecast_horizon=7,
        ),
        split_on="day",
    )

    out = fit_history(session, by=["store"], value="sales", offsets=[7], keep=10)

    assert "store_sales_lag7" in out["new_columns"]
    assert "store_sales_lag7_age" in out["new_columns"]


def test_the_bundle_predicts_rows_dated_after_every_row_it_saw(session, tmp_path):
    """The serving case history exists for: a contiguous future block, unlabelled,
    later than all stored history."""
    table = fit_history(session, by=["store"], value="sales")["table"]
    train_tabular_model(session, table=table)
    out = finalize_run(session)

    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))
    future = pd.DataFrame(
        {
            "store": ["s1", "s2", "s1"],
            "day": [130, 130, 200],
            "promo": [0, 1, 0],
        }
    )

    preds = predict_bundle(bundle, future)

    assert len(preds) == 3
    assert np.isfinite(preds).all()


def test_a_served_row_reads_history_not_its_own_batch(session):
    """0011's property at the level that matters: one row alone and the same row
    inside a batch get the same feature."""
    table = fit_history(session, by=["store"], value="sales")["table"]
    train_tabular_model(session, table=table)
    out = finalize_run(session)
    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))

    batch = pd.DataFrame(
        {"store": ["s1", "s1", "s1"], "day": [121, 122, 123], "promo": [0, 0, 0]}
    )
    alone = batch.iloc[[2]]

    assert predict_bundle(bundle, alone)[0] == pytest.approx(
        predict_bundle(bundle, batch)[2]
    )


def test_the_stored_values_are_the_frozen_ones_not_the_rows_own(session):
    """The table the model trains on holds per-fold values, so a row's own sales
    can never be the number beside it."""
    out = fit_history(session, by=["store"], value="sales")
    table = load_table(session.resolve_table(out["table"]))

    seen = table["store_sales_last"].notna()
    assert seen.any()
    # Never the row's own value: the freeze is strictly in that row's past.
    assert (table.loc[seen, "store_sales_last"] != table.loc[seen, "sales"]).all()
    assert (table.loc[seen, "store_sales_last_age"] > 0).all()
