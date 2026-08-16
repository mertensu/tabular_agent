"""The stream registry: how a run declares a second input, and what it is told back.

A stream is the one kind of input the harness cannot recognise by looking at it.
Two datetime columns are two datetime columns; which of them is *when a record is
about* and which is *when it became knowable* is a fact only the data's owner
has, and swapping them builds a feature out of the future. So it is **declared**,
the way `serving_columns` is, and the tests here are mostly about the declaration
failing loudly rather than quietly doing nothing.

The other half is what comes back. A second input the agent is not told about is
a second input that does not exist, and the single number that decides whether it
is usable — how far ahead of the event its records typically arrived — is one no
tool could otherwise compute, because no tool returns rows from a stream.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.brief import render_brief
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import fit_encoding, inspect_data
from tabular_agent.inspect import profile_stream
from tabular_agent.taskspec import TaskSpec


@pytest.fixture
def table_path(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.date_range("2017-01-01", periods=60)
    df = pd.DataFrame({
        "store": np.tile(["a", "b"], 60),
        "visit_date": np.repeat(days, 2),
        "holiday": rng.integers(0, 2, 120),
    })
    df["visitors"] = 20 + df["holiday"] * 5 + rng.normal(0, 2, 120)
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def stream_path(tmp_path):
    """Bookings: a third of them made on the day, the rest well ahead."""
    days = pd.date_range("2017-01-01", periods=60)
    event = np.repeat(days, 3)
    notice = np.tile([0, 10, 30], 60)
    df = pd.DataFrame({
        "store": np.tile(["a", "b", "a"], 60),
        "visit_date": event,
        "booked_at": event - pd.to_timedelta(notice, unit="D"),
        "party": np.tile([2.0, 3.0, 4.0], 60),
    })
    path = tmp_path / "bookings.csv"
    df.to_csv(path, index=False)
    return str(path)


def make_session(tmp_path, table_path, streams=None, name="run"):
    return RunSession(
        str(tmp_path / name), table_path, TaskSpec(target="visitors"),
        split_on="visit_date", streams=streams,
    )


@pytest.fixture
def declared(tmp_path, table_path, stream_path):
    return make_session(tmp_path, table_path, streams={
        "bookings": {
            "path": stream_path,
            "event_time": "visit_date",
            "observed_at": "booked_at",
            "notes": "one row per booking",
        }
    })


# --- the declaration, and what it refuses ------------------------------------


def test_a_run_without_streams_has_none(tmp_path, table_path):
    session = make_session(tmp_path, table_path)

    assert session.streams == {}
    assert session.stream_names() == []
    assert session.stream_record() == {}


def test_both_times_have_to_be_named(tmp_path, table_path, stream_path):
    """The one fact no profiler recovers, so its absence is an error not a default."""
    with pytest.raises(ValueError, match="observed_at"):
        make_session(tmp_path, table_path, streams={
            "bookings": {"path": stream_path, "event_time": "visit_date"}
        })


def test_the_two_times_may_not_be_one_column(tmp_path, table_path, stream_path):
    with pytest.raises(ValueError, match="not a stream in this sense"):
        make_session(tmp_path, table_path, streams={
            "bookings": {
                "path": stream_path, "event_time": "visit_date",
                "observed_at": "visit_date",
            }
        })


def test_an_unread_key_is_refused_rather_than_ignored(tmp_path, table_path, stream_path):
    """A declaration that looks like it said something and did not is the worst case.

    ``by`` in particular is tempting to write here and would be wrong: which
    columns to group and aggregate on is the agent's call at tool time (0029),
    not the run owner's.
    """
    with pytest.raises(ValueError, match="which this harness does not read"):
        make_session(tmp_path, table_path, streams={
            "bookings": {
                "path": stream_path, "event_time": "visit_date",
                "observed_at": "booked_at", "by": ["store"],
            }
        })


def test_the_name_has_to_survive_a_column_name_and_a_filename(
    tmp_path, table_path, stream_path
):
    with pytest.raises(ValueError, match="letters, digits and underscores"):
        make_session(tmp_path, table_path, streams={
            "air reserve/v2": {
                "path": stream_path, "event_time": "visit_date",
                "observed_at": "booked_at",
            }
        })


def test_asking_for_a_stream_that_was_not_declared_lists_the_ones_that_were(declared):
    with pytest.raises(ValueError, match="no such stream: 'hpg'"):
        declared.stream_spec("hpg")


# --- materialising it --------------------------------------------------------


def test_the_stream_is_copied_into_the_run_directory(declared):
    """0003: a run directory has to be movable, so its inputs live inside it."""
    path = declared.stream_path("bookings")

    assert path == os.path.join(
        declared.run_dir, "data", "streams", "bookings.parquet"
    )
    assert os.path.exists(path)
    assert len(pd.read_parquet(path)) == 180


def test_copying_it_twice_does_not_re_read_the_source(declared, stream_path):
    """A run is a fixed experiment against fixed data; the file is pinned once."""
    first = declared.stream_path("bookings")
    os.remove(stream_path)

    assert declared.stream_path("bookings") == first


def test_a_time_column_the_file_does_not_have_is_caught(tmp_path, table_path, stream_path):
    session = make_session(tmp_path, table_path, streams={
        "bookings": {
            "path": stream_path, "event_time": "visit_date",
            "observed_at": "created_at",
        }
    })

    with pytest.raises(ValueError, match="declares \\['created_at'\\]"):
        session.stream_path("bookings")


# --- what the agent is told --------------------------------------------------


def test_inspect_data_describes_the_stream_and_how_it_attaches(declared):
    out = inspect_data(declared)

    assert set(out["streams"]) == {"bookings"}
    reported = out["streams"]["bookings"]
    assert reported["n_records"] == 180
    # The overlap is the set of possible join keys, which is the difference
    # between "there is a second file" and "here is how it attaches".
    assert reported["shared_with_table"] == ["store", "visit_date"]
    assert reported["notes"] == "one row per booking"
    assert "fit_stream_aggregate" in out["streams_note"]


def test_inspect_data_reports_how_much_notice_the_records_gave(declared):
    """The number that decides whether a stream survives being restricted.

    A third of these bookings were made on the day. Those are exactly the records
    a forecast issued any distance ahead cannot have — and the ones a naive
    collapse would hand to every training row.
    """
    reported = inspect_data(declared)["streams"]["bookings"]

    assert reported["notice"]["median"] == 10.0
    assert reported["notice_zero_share"] == pytest.approx(1 / 3, abs=1e-3)


def test_a_run_without_streams_says_nothing_about_them(tmp_path, table_path):
    out = inspect_data(make_session(tmp_path, table_path))

    assert "streams" not in out
    assert "streams_note" not in out


def test_the_declaration_reaches_the_journal(declared):
    inspect_data(declared)
    recorded = journal.load(declared.run_dir)["streams"]

    assert recorded["bookings"]["observed_at"] == "booked_at"


def test_the_brief_names_the_second_inputs(declared):
    """A resumed agent is told to skip `inspect_data`, so the brief has to say it."""
    inspect_data(declared)
    fit_encoding(declared, columns=["store"], method="frequency")
    brief = render_brief(declared)

    assert "Second inputs declared: bookings" in brief
    assert "fit_stream_aggregate" in brief


def test_a_run_without_streams_gets_no_such_line(tmp_path, table_path):
    session = make_session(tmp_path, table_path)
    inspect_data(session)
    fit_encoding(session, columns=["store"], method="frequency")

    assert "Second inputs" not in render_brief(session)


# --- the profile itself ------------------------------------------------------


def test_a_stream_sharing_no_column_with_the_table_is_flagged():
    stream = pd.DataFrame({
        "shop": ["a"], "on": pd.to_datetime(["2017-01-05"]),
        "at": pd.to_datetime(["2017-01-01"]), "n": [3.0],
    })

    profile = profile_stream(stream, "on", "at", table_columns=["store", "date"])

    assert profile["shared_with_table"] == []
    assert "nothing to join it on" in profile["warning"]


def test_an_empty_stream_profiles_without_falling_over():
    stream = pd.DataFrame({"on": [], "at": [], "n": []})

    profile = profile_stream(stream, "on", "at")

    assert profile["n_records"] == 0
    assert "notice" not in profile


def test_records_observed_after_their_own_event_count_as_no_notice():
    """A record logged the day *after* the visit gives negative notice.

    It is not an error — late-arriving records are the ordinary case in an
    IBNR-shaped feed — but it belongs with the zero-notice ones, because a
    forecast cannot have it either.
    """
    stream = pd.DataFrame({
        "on": pd.to_datetime(["2017-01-05"] * 4),
        "at": pd.to_datetime(["2017-01-01", "2017-01-05", "2017-01-06", "2017-01-03"]),
        "n": [1.0, 1.0, 1.0, 1.0],
    })

    profile = profile_stream(stream, "on", "at")

    # Notices of 4, 0, -1 and 2 days.
    assert profile["notice_zero_share"] == 0.5
    assert profile["notice"]["median"] == 1.0
