"""`fit_stream_aggregate`: the second input, made reachable by the agent.

The encoder is tested in `test_stream_aggregate.py`; what is tested here is the
*tool* — the refusals it puts in front of the encoder, the numbers it reports
back, and the fact that a table built from a stream can be scored and replayed at
all, which needs the run's stream to reach every place a lineage is replayed.

The refusal worth reading twice is the one on short leads. It does not prevent a
leak — the restriction is backward from each row whatever the lead — it prevents
a column built from knowledge no forecast row could have. Validation cannot catch
that on its own, because validation is drawn from the same uncut file.
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import fit_stream_aggregate, inspect_data
from tabular_agent.pipeline import Pipeline
from tabular_agent.taskspec import TaskSpec


@pytest.fixture
def table_path(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.date_range("2017-01-01", periods=90)
    df = pd.DataFrame({
        "store": np.tile(["a", "b"], 90),
        "visit_date": np.repeat(days, 2),
        "weekday": np.repeat(days.dayofweek, 2),
    })
    df["visitors"] = 20 + df["weekday"] + rng.normal(0, 1.5, 180)
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def stream_file(tmp_path):
    """Three bookings per store-day: one made 30 days out, one 7, one same-day."""
    days = pd.date_range("2017-01-01", periods=90)
    rows = []
    for store in ("a", "b"):
        for day in days:
            for notice, party in ((30, 2.0), (7, 3.0), (0, 5.0)):
                rows.append({
                    "store": store,
                    "visit_date": day,
                    "booked_at": day - pd.Timedelta(days=notice),
                    "party": party,
                })
    path = tmp_path / "bookings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def make_session(tmp_path, table_path, stream_file, horizon=7.0, name="run"):
    return RunSession(
        str(tmp_path / name), table_path,
        TaskSpec(target="visitors", forecast_horizon=horizon),
        split_on="visit_date",
        streams={"bookings": {
            "path": stream_file, "event_time": "visit_date",
            "observed_at": "booked_at",
        }},
    )


@pytest.fixture
def session(tmp_path, table_path, stream_file):
    s = make_session(tmp_path, table_path, stream_file)
    inspect_data(s)
    return s


def col(lead, stat):
    return f"bookings_store_party_lead{lead:g}_{stat}"


# --- what it produces --------------------------------------------------------


def test_the_lead_decides_what_the_column_holds(session):
    """Same stream, two leads, two genuinely different features."""
    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party",
        leads=[7, 30], stats=["sum"],
    )
    built = pd.read_parquet(
        session.run_dir + "/" + out["table"].replace("\\", "/")
    )

    # At a lead of 7 the same-day booking is gone; at 30 only the earliest is left.
    assert set(built[col(7, "sum")].dropna()) == {5.0}
    assert set(built[col(30, "sum")].dropna()) == {2.0}
    assert out["records_per_row"] == {"7": 2.0, "30": 1.0}
    assert out["rows_answered"] == {"7": 1.0, "30": 1.0}


def test_every_statistic_arrives_with_its_count(session):
    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party",
        leads=[7], stats=["sum", "mean"],
    )

    assert out["new_columns"] == [col(7, "sum"), col(7, "mean"), col(7, "count")]


def test_the_table_is_scored_and_can_be_replayed(session):
    """The lineage has to reach the stream, or the table is unservable and unscored.

    This is the test that would have caught the plumbing being missing: without
    the run's stream reaching `Pipeline.transform`, replaying the lineage raises
    and the table silently ends up with no probe score at all.
    """
    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[7],
    )

    assert out.get("probe_score") is not None
    assert not out.get("unservable")


def test_the_lineage_names_the_stream_it_needs(session):
    """Read off the steps, so it cannot drift from what a replay actually needs."""
    import json
    import os

    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[7],
    )
    table_path = os.path.join(session.run_dir, out["table"])
    with open(os.path.splitext(table_path)[0] + ".pipeline.json") as f:
        pipeline = Pipeline.from_dict(json.load(f))

    assert pipeline.required_streams() == ["bookings"]
    # And the recipe alone: no aggregate, nothing that grows with the stream.
    stored = pipeline.steps[-1]["encoder"]
    assert stored["kind"] == "stream_aggregate"
    assert "party" not in str(stored.get("mapping", ""))


def test_the_journal_records_which_stream_and_which_leads(session):
    fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[7, 30],
    )
    row = journal.load(session.run_dir)["tables"][-1]

    assert row["method"] == "stream_aggregate"
    assert row["stream"] == "bookings"
    assert row["leads"] == [7.0, 30.0]


def test_the_grain_it_measured_is_reported(session):
    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[7],
    )

    assert out["event_grain"] == 1.0
    assert out["matched_rows"] == 1.0
    assert out["n_records"] == 540


def test_the_block_span_is_reported_so_a_lead_can_be_chosen_against_it(session):
    """The horizon is a block's *floor*; nothing else tells the agent its width."""
    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[7],
    )

    assert out["block_span"] > 0
    assert "floor" in out["note"]


# --- the refusals ------------------------------------------------------------


def test_a_lead_below_the_declared_horizon_is_refused(session):
    with pytest.raises(ValueError, match="shorter than the declared forecast horizon"):
        fit_stream_aggregate(
            session, stream="bookings", by=["store"], value="party", leads=[1],
        )


def test_leads_need_the_horizon_to_have_been_declared(tmp_path, table_path, stream_file):
    session = make_session(tmp_path, table_path, stream_file, horizon=None)
    inspect_data(session)

    with pytest.raises(ValueError, match="declare how far ahead it forecasts"):
        fit_stream_aggregate(
            session, stream="bookings", by=["store"], value="party", leads=[7],
        )


def test_no_lead_at_all_is_the_collapse_this_tool_replaces(session):
    with pytest.raises(ValueError, match="at least one lead time"):
        fit_stream_aggregate(
            session, stream="bookings", by=["store"], value="party", leads=[],
        )


def test_an_undeclared_stream_lists_the_declared_ones(session):
    with pytest.raises(ValueError, match="no such stream: 'hpg'"):
        fit_stream_aggregate(
            session, stream="hpg", by=["store"], value="party", leads=[7],
        )


def test_a_key_missing_from_the_stream_says_which_side_has_it(session):
    """"No rows matched" would send the caller to the data; the answer is a name."""
    with pytest.raises(ValueError, match="columns not in stream 'bookings'"):
        fit_stream_aggregate(
            session, stream="bookings", by=["weekday"], value="party", leads=[7],
        )


def test_an_unordered_run_has_no_before_and_after(tmp_path, table_path, stream_file):
    session = RunSession(
        str(tmp_path / "unordered"), table_path,
        TaskSpec(target="visitors", forecast_horizon=7.0),
        streams={"bookings": {
            "path": stream_file, "event_time": "visit_date",
            "observed_at": "booked_at",
        }},
    )
    inspect_data(session)

    with pytest.raises(ValueError, match="not split on an ordering column"):
        fit_stream_aggregate(
            session, stream="bookings", by=["store"], value="party", leads=[7],
        )


def test_an_unknown_statistic_is_refused(session):
    with pytest.raises(ValueError, match="unknown statistic"):
        fit_stream_aggregate(
            session, stream="bookings", by=["store"], value="party",
            leads=[7], stats=["max"],
        )


def test_a_key_that_will_not_exist_at_predict_time_is_refused(
    tmp_path, table_path, stream_file
):
    session = RunSession(
        str(tmp_path / "served"), table_path,
        TaskSpec(target="visitors", forecast_horizon=7.0),
        split_on="visit_date",
        serving_columns=["visit_date", "weekday"],
        streams={"bookings": {
            "path": stream_file, "event_time": "visit_date",
            "observed_at": "booked_at",
        }},
    )
    inspect_data(session)

    with pytest.raises(ValueError, match="cannot key a stream aggregate on"):
        fit_stream_aggregate(
            session, stream="bookings", by=["store"], value="party", leads=[7],
        )


# --- the diagnostics that make a bad lead visible ----------------------------


def test_a_lead_the_data_cannot_support_is_reported_not_hidden(session):
    """A lead past every record's notice does not fail; it empties the column.

    Nothing about that is an error — it is a legitimate answer to "what could a
    forecast 90 days out have seen" — but it has to be visible, or the table
    scores as a feature and is one of absences.
    """
    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[90],
    )

    assert out["rows_answered"]["90"] == 0.0
    # No median to report when nothing was answered — an absent key rather than a
    # zero, since "the typical answered row held 0 records" is not a fact.
    assert "90" not in out["records_per_row"]
    assert "WARNING" in out["note"]


def test_the_two_shares_are_reported_on_denominators_that_can_be_compared(session):
    """`rows_answered` against `matched_rows` is the comparison that means something.

    A feed reaching a tenth of the rows makes any median-over-all-rows read 0
    whatever the lead — which describes the feed, not the choice. So the share is
    over every row (comparable with `matched_rows`) and the median is over the
    rows that got an answer.
    """
    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[7],
    )

    assert out["rows_answered"]["7"] <= out["matched_rows"]
    assert "matched_rows" in out["note"]


def test_a_healthy_lead_carries_no_warning(session):
    out = fit_stream_aggregate(
        session, stream="bookings", by=["store"], value="party", leads=[7],
    )

    assert "WARNING" not in out["note"]
