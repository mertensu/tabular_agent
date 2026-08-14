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

    assert out["new_columns"] == ["store_sales_latest", "store_sales_latest_ago"]
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

    assert "store_sales_latest" in out["new_columns"]


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
    seen = table["store_sales_latest_ago"].notna()
    # Days, not the nanosecond integers a naive conversion would hand back.
    assert table.loc[seen, "store_sales_latest_ago"].max() < 365


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

    assert "store_sales_7_ago" in out["new_columns"]
    assert "store_sales_7_ago_exact" in out["new_columns"]


def test_the_tool_reports_what_share_of_rows_each_offset_could_answer(tmp_path, panel):
    """0050: an offset shorter than the period being predicted is answerable for
    the earliest rows and silently becomes the last-known value for the rest.

    On Rossmann v8 a 7-day offset over a 44-day block was honoured for 11% of
    rows and nothing said so. This is the line that would have said so.
    """
    session = RunSession(
        str(tmp_path / "run_answerable"),
        panel,
        TaskSpec(
            target="sales", task_type="regression", metric="rmse", n_splits=3,
            forecast_horizon=1,
        ),
        split_on="day",
    )

    out = fit_history(
        session, by=["store"], value="sales", offsets=[2, 60], keep=80
    )

    short, long = out["offsets_answerable"]["2"], out["offsets_answerable"]["60"]
    # The blocks here are ~30 days wide, so a 2-day offset is answerable only for
    # the first rows of each and a 60-day one reaches back past the boundary.
    assert short < long
    assert "WARNING" in out["note"] and "'2'" in out["note"]


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

    seen = table["store_sales_latest"].notna()
    assert seen.any()
    # Never the row's own value: the freeze is strictly in that row's past.
    assert (table.loc[seen, "store_sales_latest"] != table.loc[seen, "sales"]).all()
    assert (table.loc[seen, "store_sales_latest_ago"] > 0).all()


# --- trailing windows --------------------------------------------------------


@pytest.fixture
def forecasting_session(tmp_path, panel):
    """A session that declares a horizon, which windows require."""
    return RunSession(
        str(tmp_path / "run_windows"),
        panel,
        TaskSpec(
            target="sales", task_type="regression", metric="rmse", n_splits=3,
            forecast_horizon=1,
        ),
        split_on="day",
    )


def test_a_window_adds_its_statistic_and_its_count(forecasting_session):
    out = fit_history(
        forecasting_session, by=["store"], value="sales",
        windows=[[20, "mean"], [20, "std"]], keep=40,
    )

    assert out["new_columns"] == [
        "store_sales_latest", "store_sales_latest_ago",
        "store_sales_20_mean", "store_sales_20_std", "store_sales_20_count",
    ]
    row = journal.load(forecasting_session.run_dir)["tables"][-1]
    assert row["windows"] == [[20.0, "mean"], [20.0, "std"]]


def test_the_tool_reports_how_much_each_window_actually_held(forecasting_session):
    """`offsets_answerable`'s counterpart. A window that averages two points on the
    rows that matter is 0050's degraded lag in new clothing."""
    out = fit_history(
        forecasting_session, by=["store"], value="sales",
        windows=[[2, "mean"], [40, "mean"]], keep=60,
    )

    assert out["window_observations"]["40"] > out["window_observations"]["2"]
    # The 2-day window spans less than the gap for all but the block's first rows.
    assert "WARNING" in out["note"] and "'2'" in out["note"]


def test_a_window_no_fold_can_fill_says_so_without_predicting_a_cost(
    forecasting_session,
):
    """The failure the count cannot see, and the expensive one.

    Past the *longest* training prefix no fold can fill the window while a served
    row fills it completely, so each fold's column summarises a different number
    of observations. Measured on Rossmann: per-fold medians ramping 20/39/59/78/93
    against a served 101, costing 0.093 — while the count read *higher* than a
    healthy shorter window's.
    """
    out = fit_history(
        forecasting_session, by=["store"], value="sales",
        windows=[[500, "mean"]], keep=600,
    )

    assert out["longest_training_prefix"] < 500
    assert "every fold" in out["note"]
    assert "not settled" in out["note"]


def test_a_window_only_the_earliest_folds_cannot_fill_is_noted_more_briefly(
    forecasting_session,
):
    """Between the two prefixes the later folds still settle near what serving
    delivers, so this is a note rather than a warning — Rossmann's 364-day window
    sat here and was the best of the sweep."""
    out = fit_history(
        forecasting_session, by=["store"], value="sales",
        windows=[[60, "mean"]], keep=120,
    )

    assert out["shortest_training_prefix"] < 60 <= out["longest_training_prefix"]
    assert "shortest training prefix" in out["note"]
    assert "every fold" not in out["note"]


def test_a_window_inside_every_fold_prefix_is_left_alone(forecasting_session):
    out = fit_history(
        forecasting_session, by=["store"], value="sales",
        windows=[[10, "mean"]], keep=40,
    )

    assert 10 <= out["shortest_training_prefix"]
    assert "training prefix" not in out["note"]


def test_a_window_shorter_than_the_horizon_is_refused_through_the_tool(tmp_path, panel):
    session = RunSession(
        str(tmp_path / "run_short_window"),
        panel,
        TaskSpec(
            target="sales", task_type="regression", metric="rmse", n_splits=3,
            forecast_horizon=14,
        ),
        split_on="day",
    )

    with pytest.raises(ValueError, match="shorter than the declared forecast horizon"):
        fit_history(session, by=["store"], value="sales", windows=[[7, "mean"]])


def test_windows_need_the_task_to_declare_its_horizon(session):
    with pytest.raises(ValueError, match="forecast_horizon"):
        fit_history(session, by=["store"], value="sales", windows=[[20, "mean"]])


def test_a_windowed_bundle_predicts_rows_after_every_row_it_saw(forecasting_session):
    """The serving case, for the columns slice 3 adds: the window has to be
    computable from stored history alone, on rows later than all of it."""
    table = fit_history(
        forecasting_session, by=["store"], value="sales",
        windows=[[30, "mean"], [30, "median"]], keep=40,
    )["table"]
    train_tabular_model(forecasting_session, table=table)
    out = finalize_run(forecasting_session)

    bundle = load_bundle(os.path.join(forecasting_session.run_dir, out["bundle_dir"]))
    future = pd.DataFrame(
        {"store": ["s1", "s2", "s1"], "day": [125, 130, 200], "promo": [0, 1, 0]}
    )

    preds = predict_bundle(bundle, future)

    assert len(preds) == 3
    assert np.isfinite(preds).all()


def test_a_served_window_is_the_mean_of_the_rows_own_stretch_of_stored_history(
    forecasting_session, panel
):
    """Recomputed by hand from the raw data, because a window is exactly the kind
    of feature that can be wrong by a plausible amount without looking wrong."""
    out = fit_history(
        forecasting_session, by=["store"], value="sales",
        windows=[[30, "mean"]], keep=40,
    )
    table = load_table(forecasting_session.resolve_table(out["table"]))
    assert "store_sales_30_mean" in table.columns

    from tabular_agent.encoders import HistoryEncoder

    raw = pd.read_csv(panel)
    encoder = HistoryEncoder(
        ["store"], "sales", "day", windows=[(30, "mean")], keep=40
    ).fit(raw)
    served = encoder.transform(pd.DataFrame({"store": ["s1"], "day": [125]}))

    # Stored history runs to day 119, so [95, 125) is days 95-119 of store s1.
    expected = raw[
        (raw["store"] == "s1") & (raw["day"] >= 95) & (raw["day"] < 125)
    ]["sales"].mean()

    assert served["store_sales_30_mean"].iloc[0] == pytest.approx(expected)
    assert served["store_sales_30_count"].iloc[0] == 25.0
