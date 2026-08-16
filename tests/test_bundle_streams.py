"""Shipping a model that reads a second input, and refusing to ship one blind.

The bundle is where 0062's central choice becomes visible. The encoder stores a
*recipe*, so the stream has to be handed back at predict time — and the price of
that is a bundle which cannot answer on its own. The alternative was a bundle
carrying its own copy of the aggregate, which needs no argument here at all and
is frozen the moment it is written; 0050 is the record of a frozen anchor costing
a working model two thirds of its score.

So the two properties under test are: a bundle handed a stream that has **moved
on** uses the new records, and a bundle handed **nothing** says so loudly, naming
the columns its file has to carry rather than failing somewhere inside a
transform.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    finalize_run,
    fit_stream_aggregate,
    inspect_data,
)
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.taskspec import TaskSpec


@pytest.fixture
def table_path(tmp_path):
    rng = np.random.default_rng(1)
    days = pd.date_range("2017-01-01", periods=120)
    df = pd.DataFrame({
        "store": np.tile(["a", "b"], 120),
        "visit_date": np.repeat(days, 2),
        "weekday": np.repeat(days.dayofweek, 2),
    })
    df["visitors"] = 20 + df["weekday"] + rng.normal(0, 1.0, 240)
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def stream_frame():
    """One booking per store-day, made 30 days ahead."""
    days = pd.date_range("2017-01-01", periods=200)
    rows = [
        {"store": store, "visit_date": day,
         "booked_at": day - pd.Timedelta(days=30), "party": 4.0}
        for store in ("a", "b") for day in days
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def stream_file(tmp_path, stream_frame):
    """Only the first 120 days — the file as it stood when the model was fitted."""
    path = tmp_path / "bookings.csv"
    cutoff = pd.Timestamp("2017-01-01") + pd.Timedelta(days=119)
    stream_frame[stream_frame["visit_date"] <= cutoff].to_csv(path, index=False)
    return str(path)


@pytest.fixture
def shipped(tmp_path, table_path, stream_file):
    """A finalized run whose winning table came from the stream."""
    session = RunSession(
        str(tmp_path / "run"), table_path,
        TaskSpec(target="visitors", forecast_horizon=7.0),
        split_on="visit_date",
        streams={"bookings": {
            "path": stream_file, "event_time": "visit_date",
            "observed_at": "booked_at",
        }},
    )
    inspect_data(session)
    fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[7],
    )
    out = finalize_run(session)
    return session, out


def test_the_manifest_says_which_stream_and_which_columns(shipped):
    """A year later the caller has the file and not the run that read it.

    Which of two timestamps was the observation time is exactly the fact that
    cannot be recovered by looking — and reading the wrong one builds the feature
    out of the future without raising.
    """
    session, out = shipped
    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))

    assert bundle["streams"]["bookings"]["event_time"] == "visit_date"
    assert bundle["streams"]["bookings"]["observed_at"] == "booked_at"
    assert bundle["streams"]["bookings"]["columns"] == ["store", "party"]
    # And in the finalize result, which is what a shipping script reads.
    assert out["required_streams"] == ["bookings"]


def test_a_bundle_handed_no_stream_refuses_and_says_what_it_needs(shipped):
    session, out = shipped
    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))
    serving = pd.DataFrame({
        "store": ["a"], "visit_date": pd.to_datetime(["2017-06-01"]), "weekday": [3],
    })

    with pytest.raises(ValueError, match="cannot predict without them"):
        predict_bundle(bundle, serving)


def test_the_stream_handed_in_at_predict_time_is_the_one_that_is_used(
    shipped, stream_frame
):
    """The whole reason the recipe is stored instead of the numbers.

    The fitted file stopped at day 120. A row on day 160 is answerable only from
    records that arrived afterwards — and it is answered, because the aggregate is
    recomputed rather than looked up.
    """
    session, out = shipped
    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))
    served_day = pd.Timestamp("2017-01-01") + pd.Timedelta(days=160)
    serving = pd.DataFrame({
        "store": ["a"], "visit_date": [served_day], "weekday": [served_day.dayofweek],
    })

    preds = predict_bundle(bundle, serving, streams={"bookings": stream_frame})

    assert len(preds) == 1
    assert np.isfinite(preds).all()
    # The column really was filled from the newer records, not left empty.
    features = bundle["pipeline"].transform(serving, streams={"bookings": stream_frame})
    assert features["bookings_store_party_lead7_count"].iloc[0] == 1.0


def test_a_stale_stream_answers_nothing_and_does_not_pretend_otherwise(shipped):
    """Handing back the training-time file is the mistake the message warns about.

    It does not raise — a stream legitimately has gaps — so what protects the
    caller is the count column reading zero rather than a number.
    """
    session, out = shipped
    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))
    # The copy pinned in the run directory: exactly the file the fit saw.
    stale = pd.read_parquet(
        os.path.join(session.run_dir, "data", "streams", "bookings.parquet")
    )
    served_day = pd.Timestamp("2017-01-01") + pd.Timedelta(days=160)
    serving = pd.DataFrame({
        "store": ["a"], "visit_date": [served_day], "weekday": [served_day.dayofweek],
    })

    features = bundle["pipeline"].transform(serving, streams={"bookings": stale})

    assert features["bookings_store_party_lead7_count"].iloc[0] == 0.0


def test_a_bundle_with_no_stream_needs_no_argument(tmp_path, table_path):
    """Every bundle written before this existed keeps predicting exactly as it did."""
    from tabular_agent.harness.tools import fit_encoding

    session = RunSession(
        str(tmp_path / "plain"), table_path, TaskSpec(target="visitors"),
        split_on="visit_date",
    )
    inspect_data(session)
    fit_encoding(session, columns=["store"], method="frequency")
    out = finalize_run(session)
    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))

    assert bundle["streams"] == {}
    assert "required_streams" not in out
    raw = pd.read_csv(table_path).drop(columns=["visitors"]).head(5)
    assert len(predict_bundle(bundle, raw)) == 5
