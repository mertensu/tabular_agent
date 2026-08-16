"""StreamAggregateEncoder: a second input, restricted to what was knowable in time.

The property under test throughout is **not** "the aggregate is right". It is
*which records were allowed into it*. A stream carries two times — when a record
is about, and when it became knowable — and collapsing it into a column keeps the
first and throws away the second, which produces a feature that is systematically
richer in training than at serving by an amount that grows with the horizon.
Nothing else in this harness can see that: no fold is crossed, no encoder is
misfitted, the column exists at predict time, and validation confirms it.

So the tests that matter are the ones that pin down the *boundary*:

* a record observed too late is out, at every lead (0062);
* the count travels with every statistic, because a sum over one record and a
  sum over forty are the same number under the same name;
* nothing is stored — the stream is handed back in at transform time, so a
  stream that has moved on since the fit is the one that gets used (0010/0050);
* what a row reads does not depend on the batch it arrives in (0011).
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.encoders import (
    ENCODER_CLASSES,
    StreamAggregateEncoder,
)


D1, D2, D3 = "2017-03-01", "2017-03-02", "2017-03-03"


@pytest.fixture
def table():
    """Three stores over three days. Store "c" appears in no stream record."""
    return pd.DataFrame(
        {
            "store": np.repeat(["a", "b", "c"], 3),
            "visit_date": pd.to_datetime([D1, D2, D3] * 3),
            "visitors": np.arange(9.0),
        }
    )


@pytest.fixture
def stream():
    """Bookings, each with the day it is *for* and the day it was *made*.

    The three records for ``a`` on the 3rd are the whole point: booked 11 days,
    2 days and 0 days ahead, so each lead admits a different prefix of them.
    """
    return pd.DataFrame(
        {
            "store": ["a", "a", "a", "a", "b"],
            "visit_date": pd.to_datetime([D3, D3, D3, D1, D2]),
            "booked_at": pd.to_datetime(
                ["2017-02-20", D1, D3, "2017-02-01", D2]
            ),
            "party": [5.0, 2.0, 1.0, 4.0, 7.0],
        }
    )


def encoder(leads=(1,), stats=("sum",)):
    return StreamAggregateEncoder(
        "reserve", ["store"], "party", "visit_date", "booked_at", "visit_date",
        leads=leads, stats=stats,
    )


def col(lead, stat):
    return f"reserve_store_party_lead{lead:g}_{stat}"


# --- the boundary the whole encoder exists to draw ---------------------------


def test_a_lead_admits_only_records_observed_early_enough(table, stream):
    """Same bucket, four leads, four different answers — and that is the feature."""
    enc = encoder(leads=[0, 1, 7, 30], stats=["sum"])
    out = enc.fit(table, stream=stream).transform(table, stream)
    # Store "a" on the 3rd: bookings made 11 days, 2 days and 0 days ahead.
    row = out[(out["store"] == "a") & (out["visit_date"] == D3)].iloc[0]

    assert (row[col(0, "sum")], row[col(0, "count")]) == (8.0, 3.0)
    assert (row[col(1, "sum")], row[col(1, "count")]) == (7.0, 2.0)
    assert (row[col(7, "sum")], row[col(7, "count")]) == (5.0, 1.0)
    # Nothing was knowable a month out, and the bucket exists, so that is a real
    # zero rather than a missing value.
    assert (row[col(30, "sum")], row[col(30, "count")]) == (0.0, 0.0)


def test_the_collapsed_column_is_what_a_lead_of_zero_would_give(table, stream):
    """The defect, reproduced: lead 0 is the naive collapse, and it is bigger."""
    enc = encoder(leads=[0, 1], stats=["sum"])
    out = enc.fit(table, stream=stream).transform(table, stream)
    naive = stream.groupby(["store", "visit_date"])["party"].sum()

    row = out[(out["store"] == "a") & (out["visit_date"] == D3)].iloc[0]
    assert row[col(0, "sum")] == naive.loc[("a", pd.Timestamp(D3))]
    assert row[col(1, "sum")] < row[col(0, "sum")]


def test_a_record_observed_after_the_row_is_never_read(table, stream):
    """Why there is no fold loop: the restriction is backward per row already.

    Store ``b``'s single booking was made on the day of the visit, so every
    positive lead excludes it — whichever half of the data the row is in and
    whatever else the file contains.
    """
    enc = encoder(leads=[1], stats=["sum"])
    out = enc.fit(table, stream=stream).transform(table, stream)
    row = out[(out["store"] == "b") & (out["visit_date"] == D2)].iloc[0]

    assert row[col(1, "count")] == 0.0
    assert row[col(1, "sum")] == 0.0


def test_the_key_and_the_event_time_both_have_to_match(table, stream):
    """A booking for another store, or another day, is not this row's booking."""
    # 28 is exactly how far ahead `a`'s booking for the 1st was made, so this
    # also pins the endpoint down as inclusive: a record knowable at the moment
    # the forecast is issued is knowable.
    enc = encoder(leads=[28], stats=["sum"])
    out = enc.fit(table, stream=stream).transform(table, stream)

    # `a` on the 1st has that booking; `b` on the 1st has none, though `b` does
    # appear in the stream on another day.
    a_first = out[(out["store"] == "a") & (out["visit_date"] == D1)].iloc[0]
    b_first = out[(out["store"] == "b") & (out["visit_date"] == D1)].iloc[0]
    assert (a_first[col(28, "sum")], a_first[col(28, "count")]) == (4.0, 1.0)
    assert (b_first[col(28, "sum")], b_first[col(28, "count")]) == (0.0, 0.0)


# --- absence, which is two different facts -----------------------------------


def test_a_key_the_stream_never_mentions_is_missing_not_zero(table, stream):
    """"Not in this feed" and "nothing booked that day" must not read alike."""
    enc = encoder(leads=[1], stats=["sum", "mean"])
    out = enc.fit(table, stream=stream).transform(table, stream)

    absent = out[out["store"] == "c"]
    assert absent[col(1, "count")].isna().all()
    assert absent[col(1, "sum")].isna().all()

    # ...against a store that *is* in the feed and simply had a quiet day.
    quiet = out[(out["store"] == "a") & (out["visit_date"] == D2)].iloc[0]
    assert quiet[col(1, "count")] == 0.0
    assert quiet[col(1, "sum")] == 0.0
    # A mean over nothing has no honest value, unlike a total.
    assert np.isnan(quiet[col(1, "mean")])


def test_a_row_with_no_ordering_value_reads_nothing(table, stream):
    served = pd.DataFrame(
        {"store": ["a"], "visit_date": pd.to_datetime([None])}
    )
    out = encoder(leads=[1]).fit(table, stream=stream).transform(served, stream)

    assert out[col(1, "count")].isna().all()
    assert out[col(1, "sum")].isna().all()


# --- what is emitted, and with what ------------------------------------------


def test_every_lead_carries_its_own_count(table, stream):
    enc = encoder(leads=[1, 7], stats=["sum", "mean", "std"])

    assert enc.output_columns == [
        col(1, "sum"), col(1, "mean"), col(1, "std"), col(1, "count"),
        col(7, "sum"), col(7, "mean"), col(7, "std"), col(7, "count"),
    ]
    out = enc.fit(table, stream=stream).transform(table, stream)
    assert set(enc.output_columns) <= set(out.columns)


def test_the_mean_and_the_spread_come_off_the_same_prefix(table, stream):
    enc = encoder(leads=[0], stats=["mean", "std"])
    out = enc.fit(table, stream=stream).transform(table, stream)
    row = out[(out["store"] == "a") & (out["visit_date"] == D3)].iloc[0]

    parties = np.array([5.0, 2.0, 1.0])
    assert row[col(0, "mean")] == pytest.approx(parties.mean())
    assert row[col(0, "std")] == pytest.approx(parties.std(ddof=1))


def test_one_record_has_no_spread_and_says_so(table, stream):
    """NaN rather than the zero a population formula hands back (0049)."""
    enc = encoder(leads=[7], stats=["sum", "std"])
    out = enc.fit(table, stream=stream).transform(table, stream)
    row = out[(out["store"] == "a") & (out["visit_date"] == D3)].iloc[0]

    assert row[col(7, "count")] == 1.0
    assert row[col(7, "sum")] == 5.0
    assert np.isnan(row[col(7, "std")])


def test_the_stream_name_is_in_the_column_name(table, stream):
    """Two feeds with the same key and value columns must not collide."""
    other = StreamAggregateEncoder(
        "partner", ["store"], "party", "visit_date", "booked_at", "visit_date",
        leads=[1],
    )
    assert other.output_columns[0] == "partner_store_party_lead1_sum"
    assert other.output_columns[0] != encoder(leads=[1]).output_columns[0]


# --- the grain, which the library owns and does not delegate ------------------


def test_a_stream_stamped_finer_than_the_table_lands_on_the_right_row(table, stream):
    """The ordinary case, and the library's job rather than a prep precondition.

    Bookings carry a time of day; the table has one row per store-day. Nobody
    has to know to floor the stream first — and if they had to, forgetting would
    produce a well-formed column of zeros rather than an error.
    """
    fine = stream.assign(visit_date=stream["visit_date"] + pd.Timedelta(hours=18))
    enc = encoder(leads=[1], stats=["sum"])
    out = enc.fit(table, stream=fine).transform(table, fine)
    row = out[(out["store"] == "a") & (out["visit_date"] == D3)].iloc[0]

    assert enc.grain_ == 1.0
    assert (row[col(1, "sum")], row[col(1, "count")]) == (7.0, 2.0)


def test_the_lead_keeps_full_resolution_when_the_bucket_does_not(table, stream):
    """Flooring decides *which row*; it must not blunt *what was knowable*.

    The booking made at 06:00 on the 2nd is one day and 18 hours before an 18:00
    visit on the 3rd, so a lead of exactly one day still admits it — while the
    bucket it lands in is the whole of the 3rd.
    """
    fine = stream.assign(
        visit_date=stream["visit_date"] + pd.Timedelta(hours=18),
        booked_at=stream["booked_at"] + pd.Timedelta(hours=6),
    )
    served = pd.DataFrame({
        "store": ["a"], "visit_date": [pd.Timestamp(D3) + pd.Timedelta(hours=18)],
    })
    enc = encoder(leads=[1], stats=["sum"]).fit(table, stream=fine)
    out = enc.transform(served, fine)

    # 02-20 06:00 and 03-01 06:00 are in; 03-03 06:00 is 12 hours before the visit.
    assert (out[col(1, "sum")].iloc[0], out[col(1, "count")].iloc[0]) == (7.0, 2.0)


def test_a_missing_day_does_not_move_the_grain(stream):
    """Gaps in the calendar are normal; the commonest spacing is still the grain."""
    holidays_missing = pd.DataFrame({
        "store": ["a"] * 4,
        "visit_date": pd.to_datetime(["2017-03-01", "2017-03-02", "2017-03-05",
                                      "2017-03-06"]),
    })
    enc = encoder(leads=[1]).fit(holidays_missing, stream=stream)

    assert enc.grain_ == 1.0


def test_a_ragged_spine_does_not_move_the_grain(stream):
    """0054's raggedness: a store with no row on a day it did not report.

    The gaps are read across the whole table, so two stores each reporting every
    other day — on opposite days — still describe a daily grid.
    """
    ragged = pd.DataFrame({
        "store": ["a", "b", "a", "b"],
        "visit_date": pd.to_datetime([D1, D2, D3, "2017-03-04"]),
    })
    enc = encoder(leads=[1]).fit(ragged, stream=stream)

    assert enc.grain_ == 1.0


def test_irregular_rows_are_refused_rather_than_guessed_at(stream):
    """No spacing to read off means no bucket width, and saying so is the point."""
    irregular = pd.DataFrame({
        "store": ["a"] * 4,
        "visit_date": pd.to_datetime([
            "2017-03-01 00:00", "2017-03-01 09:30", "2017-03-02 00:00",
            "2017-03-04 17:15",
        ]),
    })

    with pytest.raises(ValueError, match="cannot measure the grain"):
        encoder(leads=[1]).fit(irregular, stream=stream)


def test_a_declared_grain_outranks_the_measurement(stream):
    """The refusal above has an answer, and it is a declaration, not a workaround."""
    irregular = pd.DataFrame({
        "store": ["a"] * 4,
        "visit_date": pd.to_datetime([
            "2017-03-01 00:00", "2017-03-01 09:30", "2017-03-02 00:00",
            "2017-03-04 17:15",
        ]),
    })
    enc = StreamAggregateEncoder(
        "reserve", ["store"], "party", "visit_date", "booked_at", "visit_date",
        leads=[1], event_grain=1.0,
    ).fit(irregular, stream=stream)

    assert enc.grain_ == 1.0
    # Both rows on the 1st fall in one bucket, which is what declaring a day means.
    out = enc.transform(irregular, stream)
    assert out[col(1, "count")].iloc[0] == out[col(1, "count")].iloc[1]


def test_a_table_with_one_date_has_no_spacing_to_read(stream):
    single = pd.DataFrame({"store": ["a", "b"], "visit_date": pd.to_datetime([D1, D1])})

    with pytest.raises(ValueError, match="one distinct value"):
        encoder(leads=[1]).fit(single, stream=stream)


def test_the_grid_is_replayed_and_not_re_derived(table, stream):
    """A serving frame is one row past the end of history; re-measuring is impossible.

    This is why the grid is stored as a width and a phase rather than as the
    dates that were seen — and why it travels through ``to_dict``.
    """
    enc = encoder(leads=[1], stats=["sum"]).fit(table, stream=stream)
    restored = ENCODER_CLASSES[enc.kind].from_dict(enc.to_dict())
    served = pd.DataFrame(
        {"store": ["a"], "visit_date": pd.to_datetime(["2018-01-01"])}
    )

    assert (restored.grain_, restored.origin_) == (enc.grain_, enc.origin_)
    # One row on a date the fit never saw: it buckets, finds nothing, and says so
    # as a zero rather than falling over.
    out = restored.transform(served, stream)
    assert out[col(1, "count")].iloc[0] == 0.0


def test_transform_before_the_grid_exists_refuses(table, stream):
    with pytest.raises(ValueError, match="no grid yet"):
        encoder(leads=[1]).transform(table, stream)


def test_a_declared_grain_must_be_positive():
    with pytest.raises(ValueError, match="event_grain must be positive"):
        StreamAggregateEncoder(
            "reserve", ["store"], "party", "visit_date", "booked_at", "visit_date",
            leads=[1], event_grain=0,
        )


# --- the properties that make this a tool rather than free code ---------------


def test_the_answer_does_not_depend_on_the_batch(table, stream):
    """0011's stability check, which a merge inside transform_features fails."""
    enc = encoder(leads=[1], stats=["sum"]).fit(table, stream=stream)
    batch = enc.transform(table, stream)
    alone = pd.concat(
        [enc.transform(table.iloc[[i]], stream) for i in range(len(table))]
    )

    pd.testing.assert_frame_equal(batch, alone)


def test_nothing_is_stored_so_a_moved_on_stream_is_the_one_that_is_used(table, stream):
    """0010 applied to an aggregate, and 0050's frozen anchor avoided.

    Materialising the aggregate at fit time needs no plumbing and works only
    while the file already contains the future. Here a booking that arrives
    after the fit is read at transform time, which is what serving looks like.
    """
    enc = encoder(leads=[1], stats=["sum"]).fit(table, stream=stream)
    # The recipe, plus the grid — which is measured and so has to be replayed.
    # No aggregate, no key list, nothing whose size grows with the stream.
    assert enc.to_dict().keys() == {
        "kind", "stream", "by", "value", "event_time", "observed_at", "order",
        "leads", "stats", "event_grain", "grain", "origin",
    }

    later = pd.concat([stream, pd.DataFrame({
        "store": ["a"], "visit_date": pd.to_datetime([D3]),
        "booked_at": pd.to_datetime(["2017-02-25"]), "party": [9.0],
    })], ignore_index=True)
    row = enc.transform(table, later)
    row = row[(row["store"] == "a") & (row["visit_date"] == D3)].iloc[0]

    assert (row[col(1, "sum")], row[col(1, "count")]) == (16.0, 3.0)


def test_transform_without_the_stream_refuses_loudly(table, stream):
    enc = encoder(leads=[1]).fit(table, stream=stream)

    with pytest.raises(ValueError, match="handed in again at predict time"):
        enc.transform(table)


def test_the_recipe_survives_a_round_trip(table, stream):
    enc = encoder(leads=[1, 7], stats=["sum", "std"]).fit(table, stream=stream)
    restored = ENCODER_CLASSES[enc.kind].from_dict(enc.to_dict())

    assert isinstance(restored, StreamAggregateEncoder)
    assert restored.to_dict() == enc.to_dict()
    pd.testing.assert_frame_equal(
        restored.transform(table, stream), enc.transform(table, stream)
    )


# --- refusals ----------------------------------------------------------------


def test_a_stream_with_no_usable_records_is_refused(table, stream):
    empty = stream.assign(booked_at=pd.NaT)

    with pytest.raises(ValueError, match="no usable records"):
        encoder(leads=[1]).fit(table, stream=empty)


def test_fit_needs_the_stream(table):
    with pytest.raises(ValueError, match="needs the stream itself"):
        encoder(leads=[1]).fit(table)


def test_the_key_columns_must_carry_the_same_names_on_both_sides(table, stream):
    renamed = stream.rename(columns={"store": "shop"})

    with pytest.raises(ValueError, match="same names"):
        encoder(leads=[1]).fit(table, stream=renamed)


def test_a_negative_lead_is_refused():
    with pytest.raises(ValueError, match="observed \\*after\\*"):
        encoder(leads=[-1])


def test_a_stream_aggregate_without_a_lead_is_the_collapsed_column():
    with pytest.raises(ValueError, match="at least one lead"):
        encoder(leads=[])


def test_the_two_times_may_not_be_one_column():
    with pytest.raises(ValueError, match="not bitemporal"):
        StreamAggregateEncoder(
            "reserve", ["store"], "party", "booked_at", "booked_at", "visit_date",
            leads=[1],
        )


def test_an_extreme_over_a_handful_of_records_is_off_the_menu():
    with pytest.raises(ValueError, match="min/max and median"):
        encoder(leads=[1], stats=["max"])


def test_the_value_may_not_be_a_key_or_a_time():
    with pytest.raises(ValueError, match="value column"):
        StreamAggregateEncoder(
            "reserve", ["store"], "booked_at", "visit_date", "booked_at",
            "visit_date", leads=[1],
        )


def test_by_must_name_something():
    with pytest.raises(ValueError, match="at least one column"):
        StreamAggregateEncoder(
            "reserve", [], "party", "visit_date", "booked_at", "visit_date",
            leads=[1],
        )
