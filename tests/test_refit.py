"""Refit before shipping: the lineage and the shipped models, on work + validation.

Two properties per step type. Refitting a table's lineage on the *work* rows
reproduces the tool's own table and lineage exactly — the refit reuses the fit
code, it does not reimplement it. Refitting on work + validation moves the
fitted state onto the rows the run held back. Then finalize: what ships is the
refit, the sealed slice stays out of it and scores it once, beside the
work-only record.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal, tools
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    _load_lineage,
    _run_streams,
    filter_rows,
    finalize_run,
    fit_encoding,
    fit_future_covariate_window,
    fit_group_stats,
    fit_history,
    fit_stream_aggregate,
    inspect_data,
    transform_features,
)
from tabular_agent.io import load_table, save_parquet
from tabular_agent.metrics import compute_metric
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.refit import RefitError, refit_lineage
from tabular_agent.taskspec import TaskSpec

TRANSFORM = """
def transform_features(df):
    df = df.copy()
    df["x2"] = df["x"] * 2
    return df
"""
KEEP = """
def keep_rows(df):
    return df["x"] < 2.0
"""


def _refit(session, table_rel, rows):
    spec = session.resolved()
    lineage = _load_lineage(session.resolve_table(table_rel), spec.task_type)
    return refit_lineage(lineage, rows, spec, order=session.split_on,
                         streams=_run_streams(session))


def _assert_reproduces(session, out, tmp_path):
    """Refitting on the rows the tool fitted on gives the tool's table and lineage."""
    path = session.resolve_table(out["table"])
    table, pipeline = _refit(session, out["table"], load_table(session.data_path))
    # Through parquet, as the tool's table went, so dtypes compare like for like.
    roundtrip = str(tmp_path / "refit_roundtrip.parquet")
    save_parquet(table, roundtrip)
    pd.testing.assert_frame_equal(load_table(roundtrip), load_table(path))
    stored = _load_lineage(path, session.resolved().task_type).to_dict()
    assert json.dumps(pipeline.to_dict(), sort_keys=True, default=str) == json.dumps(
        stored, sort_keys=True, default=str
    )


def _work_and_val(session):
    return pd.concat(
        [load_table(session.data_path), load_table(session.val_path)], ignore_index=True
    )


# ------------------------------------------------------------- random split


@pytest.fixture
def iid_session(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    city = rng.choice(["a", "b", "c", "d"], n)
    size = rng.choice(["s", "m", "l"], n)
    x = rng.normal(size=n)
    effect = pd.Series(city).map({"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}).to_numpy()
    y = effect + x + pd.Series(size).map({"s": 0, "m": 1, "l": 2}).to_numpy() \
        + rng.normal(scale=0.3, size=n)
    path = tmp_path / "iid.csv"
    pd.DataFrame({"city": city, "size": size, "x": x, "y": y}).to_csv(path, index=False)
    session = RunSession(str(tmp_path / "iid_run"), str(path), TaskSpec(target="y", n_splits=3),
                         holdout_fraction=0.1)
    inspect_data(session)
    return session


def test_every_random_split_step_refits_to_the_tools_own_result(iid_session, tmp_path):
    s = iid_session
    steps = [
        lambda t: transform_features(s, TRANSFORM, table=t),
        lambda t: fit_encoding(s, ["city"], "frequency", table=t),
        lambda t: fit_encoding(s, ["size"], "ordinal", table=t,
                               categories={"size": ["s", "m", "l"]}),
        lambda t: fit_encoding(s, ["city"], "target", table=t, stats=["mean", "std"]),
        lambda t: fit_group_stats(s, ["city"], "x", "median", table=t),
        lambda t: filter_rows(s, KEEP, table=t),
    ]
    table = None
    for step in steps:
        out = step(table)
        _assert_reproduces(s, out, tmp_path)
        table = out["table"]


def test_refit_on_work_and_validation_moves_the_fitted_state(iid_session):
    s = iid_session
    out = fit_encoding(s, ["city"], "target", stats=["mean"])
    out = filter_rows(s, KEEP, table=out["table"])
    rows = _work_and_val(s)
    table, pipeline = _refit(s, out["table"], rows)

    # The filter is re-evaluated on the new rows; nothing sealed is among them.
    assert len(table) == int((rows["x"] < 2.0).sum())
    assert len(rows) == len(load_table(s.data_path)) + len(load_table(s.val_path))
    [encode, filt] = pipeline.steps
    assert filt["type"] == "filter" and filt["train_only"]
    work_encoder = _load_lineage(s.resolve_table(out["table"]), "regression").steps[0]
    assert encode["encoder"]["mapping"] != work_encoder["encoder"]["mapping"]
    # Out-of-fold on the refit rows, not the final encoder's in-sample values.
    final = pipeline.transform(table.drop(columns=["y"]))
    assert not np.allclose(table["city_te"].to_numpy(), final["city_te"].to_numpy())


def test_a_step_that_stored_lags_is_refused(iid_session):
    from tabular_agent.pipeline import Pipeline

    lineage = Pipeline.from_dict({"task_type": "regression", "steps": [{
        "type": "encode",
        "encoder": {"kind": "history", "by": ["city"], "value": "y", "order": "x",
                    "offsets": [1], "keep": None, "windows": [], "latest": True,
                    "history": {}, "frozen_at": None},
    }]})
    with pytest.raises(RefitError, match="0072"):
        refit_lineage(lineage, load_table(iid_session.data_path), iid_session.resolved(),
                      order="x")


# ------------------------------------------------------------ ordered split


@pytest.fixture
def forecast_paths(tmp_path):
    rng = np.random.default_rng(1)
    days = pd.date_range("2017-01-01", periods=240)
    stores = ["a", "b", "c"]
    df = pd.DataFrame({
        "store": np.tile(stores, len(days)),
        "date": np.repeat(days, len(stores)),
    })
    df["weekday"] = df["date"].dt.dayofweek
    df["temp"] = 10 + 5 * np.sin(np.arange(len(df)) / 40) + rng.normal(0, 1, len(df))
    level = df["store"].map({"a": 20.0, "b": 30.0, "c": 40.0})
    trend = np.arange(len(df)) / len(df) * 15  # the latest rows run higher
    df["sales"] = level + trend + df["weekday"] + 0.3 * df["temp"] + rng.normal(0, 1, len(df))
    table = tmp_path / "panel.csv"
    df.to_csv(table, index=False)

    rows = []
    for _, r in df.iterrows():
        for notice, party in ((14, 2.0), (3, 4.0)):
            rows.append({"store": r["store"], "date": r["date"],
                         "booked_at": r["date"] - pd.Timedelta(days=notice),
                         "party": party})
    stream = tmp_path / "bookings.csv"
    pd.DataFrame(rows).to_csv(stream, index=False)
    return str(table), str(stream)


def _forecast_session(tmp_path, paths, name="forecast_run", **kw):
    table, stream = paths
    session = RunSession(
        str(tmp_path / name), table,
        TaskSpec(target="sales", n_splits=3, forecast_horizon=7.0),
        split_on="date", future_known=["temp"], workflow="forecast",
        streams={"bookings": {"path": stream, "event_time": "date",
                              "observed_at": "booked_at"}},
        **kw,
    )
    inspect_data(session)
    return session


def test_every_ordered_step_refits_to_the_tools_own_result(tmp_path, forecast_paths):
    s = _forecast_session(tmp_path, forecast_paths)
    steps = [
        lambda t: fit_encoding(s, ["store"], "target", table=t),
        lambda t: fit_group_stats(s, ["store"], "temp", table=t),
        lambda t: fit_history(s, ["store"], "sales", windows=[[60, "mean"]], table=t),
        lambda t: fit_stream_aggregate(s, "bookings", ["store"], "party", [7.0], table=t),
        lambda t: fit_future_covariate_window(
            s, "temp", [{"family": "period", "unit": "block", "stat": "mean"},
                        {"family": "trailing", "window": 7, "stat": "mean"}],
            by=["store"], table=t),
    ]
    table = None
    for step in steps:
        out = step(table)
        _assert_reproduces(s, out, tmp_path)
        table = out["table"]


def test_refit_history_is_frozen_at_the_end_of_validation(tmp_path, forecast_paths):
    s = _forecast_session(tmp_path, forecast_paths)
    out = fit_history(s, ["store"], "sales", windows=[[60, "mean"]])
    rows = _work_and_val(s)
    _, pipeline = _refit(s, out["table"], rows)
    work_frozen = _load_lineage(s.resolve_table(out["table"]), "regression") \
        .steps[0]["encoder"]["frozen_at"]
    refit_frozen = pipeline.steps[0]["encoder"]["frozen_at"]
    assert refit_frozen > work_frozen


# --------------------------------------------------------------- finalize


def test_finalize_ships_the_refit_and_scores_it_once_beside_the_work_fit(
    tmp_path, forecast_paths, monkeypatch,
):
    s = _forecast_session(tmp_path, forecast_paths, holdout_fraction=0.15)
    out = fit_encoding(s, ["store"], "target")
    out = fit_history(s, ["store"], "sales", windows=[[60, "mean"]], table=out["table"])
    # Selection is not under test: this validation period is too short for the
    # paired interval 0090 requires, so pin the table that ships.
    monkeypatch.setattr(tools, "_best_table_path", lambda session, j: out["table"])
    result = finalize_run(s)

    n_work = len(load_table(s.data_path))
    n_val = len(load_table(s.val_path))
    holdout_rows = load_table(s.holdout_path)
    assert result["fitted_on"]["fitted_on"] == "work+validation"
    assert result["fitted_on"]["n_rows"] == n_work + n_val

    bundle_dir = os.path.join(s.run_dir, result["bundle_dir"])
    bundle = load_bundle(bundle_dir)
    assert bundle["fitted_on"]["fitted_on"] == "work+validation"
    assert bundle["training_table"] == "refit/" + os.path.basename(out["table"])
    assert bundle["input_dtypes"]["weekday"] == "int64"
    assert "sales" not in bundle["input_dtypes"]
    trained = load_table(os.path.join(s.run_dir, bundle["training_table"]))
    assert len(trained) == n_work + n_val
    # The sealed slice is the latest period and none of it was fitted on.
    assert trained["date"].max() < holdout_rows["date"].min()
    history = [st for st in bundle["pipeline"].steps
               if st["type"] == "encode" and st["encoder"]["kind"] == "history"][0]
    assert history["encoder"]["frozen_at"] > _load_lineage(
        s.resolve_table(out["table"]), "regression").steps[-1]["encoder"]["frozen_at"]

    record = journal.load(s.run_dir)["holdout"]
    refit = record["shipped_refit"]
    assert refit["n_rows_fitted"] == n_work + n_val
    assert result["holdout_score"] == refit["holdout_score"]
    # The refit's sealed score is of exactly what the bundle predicts.
    preds = predict_bundle(bundle, holdout_rows.drop(columns=["sales"]),
                           streams=_run_streams(s))
    assert compute_metric(s.resolved().metric, holdout_rows["sales"], preds) == pytest.approx(
        refit["holdout_score"], abs=1e-5)
    # The work-only record is still there for the optimism series.
    [work_fit] = [e for e in record["scores"] if e["shipped"]]
    assert result["work_fit_holdout_score"] == work_fit["holdout_score"]
    assert refit["vs_work_fit"] == pytest.approx(
        work_fit["holdout_score"] - refit["holdout_score"], abs=1e-5)

    report = open(os.path.join(s.run_dir, result["report_path"])).read()
    assert "refitted on training + validation rows" in report
    # The refit table is not reachable by an agent tool: it holds validation rows.
    with pytest.raises(ValueError):
        s.resolve_table(bundle["training_table"])


def test_a_failed_refit_ships_the_work_fit_and_says_why(tmp_path, iid_session, monkeypatch):
    def broken(*args, **kwargs):
        raise RefitError("deliberately broken")

    monkeypatch.setattr(tools, "refit_lineage", broken)
    fit_encoding(iid_session, ["city"], "frequency")
    result = finalize_run(iid_session)

    assert result["fitted_on"]["fitted_on"] == "work"
    assert "deliberately broken" in result["fitted_on"]["refit_unavailable"]
    bundle = load_bundle(os.path.join(iid_session.run_dir, result["bundle_dir"]))
    assert bundle["training_table"].startswith("tables/")
    record = journal.load(iid_session.run_dir)["holdout"]
    assert "refit_unavailable" in record["shipped_refit"]
    assert result["holdout_score"] == result["work_fit_holdout_score"]


def test_a_forecast_validation_period_spans_whole_horizons(tmp_path, forecast_paths):
    """5% of 240 days is 12 days; with a 7-day horizon validation becomes 14 days."""
    table, stream = forecast_paths
    s = RunSession(str(tmp_path / "whole"), table,
                   TaskSpec(target="sales", n_splits=3, forecast_horizon=7.0),
                   split_on="date", workflow="forecast", val_fraction=0.05)
    val = load_table(s.val_path)
    assert val["date"].nunique() == 14
    assert load_table(s.data_path)["date"].max() < val["date"].min()


def test_a_forecast_validation_period_spans_at_least_two_horizons(tmp_path, forecast_paths):
    """5% of 240 days is 12 days; with a 10-day horizon one would be nearest, two is the floor."""
    table, stream = forecast_paths
    s = RunSession(str(tmp_path / "two"), table,
                   TaskSpec(target="sales", n_splits=3, forecast_horizon=10.0),
                   split_on="date", workflow="forecast", val_fraction=0.05)
    assert load_table(s.val_path)["date"].nunique() == 20


def test_data_too_short_for_two_horizons_validates_on_one(tmp_path, forecast_paths):
    table, stream = forecast_paths
    s = RunSession(str(tmp_path / "one"), table,
                   TaskSpec(target="sales", n_splits=3, forecast_horizon=120.0),
                   split_on="date", workflow="forecast", val_fraction=0.05)
    assert load_table(s.val_path)["date"].nunique() == 120
