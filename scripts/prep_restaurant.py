"""Turn the Recruit Restaurant Visitor Forecasting download into one table.

Second forecasting dataset, chosen so that every conclusion this project has
drawn about ordered splits stops resting on Rossmann alone. It is the same
*shape* of problem — one row per entity per day, forecast a block of later dates
— at a twentieth of the size, with a different metric and a genuinely multi-file
join. What it stresses that Rossmann does not is written up in point 4.

Five things happen here, and each is a decision rather than a chore:

**1. Six files become one.** The harness takes a single table and has no join
tool, deliberately, since a join is a decision about grain rather than a feature
transform. So:

* `air_visit_data.csv` is the spine — one row per (store, date) the store
  reported, 252,108 rows over 2016-01-01 to 2017-04-22, and the target.
* `air_store_info.csv` joins on `air_store_id` (genre, area, lat/lon).
* `date_info.csv` joins on the date (day of week, holiday flag), and covers the
  forecast window too.
* `air_reserve.csv` is at a *different grain* — one row per reservation — so it
  is aggregated to (store, date) before it can join at all. See point 3.
* `hpg_reserve.csv` is keyed by a different store identifier, and
  `store_id_relation.csv` maps it onto `air_store_id` for the **150 stores of
  829** that appear on both platforms. Everything hpg-derived is therefore
  missing for 82% of rows, which is a fact about the data and not a defect.

**2. Absent days stay absent.** A store with no row on a date did not report
that date — closed, or not yet on the platform. Inserting zero-visitor rows would
be inventing observations, and the competition does not ask for those dates back.
So the table is ragged in time, which the Rossmann table is not: the harness's
ordered split and every window feature meet a spine with gaps here.

**3. The reservation aggregate is kept, and it carries a trap that Rossmann's
`Customers` does not.** For each (store, date) this writes the sum of
`reserve_visitors`, the number of reservations, and their mean lead time in days.
Unlike `Customers` the column *does* exist at prediction time — the competition
ships reservations for the forecast window. What differs is **how the column is
made**:

    the reservation stream is cut at 2017-04-22 23:00, the moment the
    forecast is issued

so a training row can count a table booked the same morning — 30% of training
reservations have a lead time of zero days — while **no row in the forecast
window can, ever**. Minimum lead time in the forecast window is 1 day at its
near edge and 39 days at its far edge. The column is thus systematically larger
in training than at serving time, by an amount that grows with the forecast
horizon.

That is a train/serve skew no guard in this harness detects: no fold boundary is
crossed, no encoder is misfitted, and validation confirms the column is
excellent because validation is drawn from the same uncut stream. It is 0047's
finding — the horizon is a column, not a constant — arriving in a *feature*
rather than in a lag.

Following 0029 and the `Customers` precedent, the column stays and the data
dictionary states the fact plainly. What the agent does with it is the agent's
decision, and if it trains on it happily we will have learned something about the
dictionary channel that a pre-cleaned table could not have taught us.

**3a. Both reservation files are also written out as *streams*, uncollapsed.**
Since 0062 the harness has a second input channel: a file at a finer grain whose
records carry both the time they are **about** (`visit_datetime`) and the time
they became **knowable** (`reserve_datetime`). `fit_stream_aggregate` can then
aggregate them per store-day *restricted to a stated lead*, which is the honest
form of the column point 3 describes.

The collapsed columns stay beside them, deliberately (0029): the harness supplies
both and the agent chooses. Handing over only the honest version would answer the
question this dataset was kept for.

**The streams are written with their timestamps untouched.** An earlier draft
floored `visit_datetime` to the visit date so it would match the table's grain,
and that was wrong for a reason bigger than this script (0063): a precondition the
library never states or checks is a trap for anyone else using it. The encoder now
measures the table's own row spacing and buckets both sides onto it, so the raw
file is what it should be given.

**4. `visit_date` is kept as a column so the run can order its split on it.**
Training ends 2017-04-22 and the forecast window is 2017-04-23 to 2017-05-31.
Pass ``split_on="visit_date"`` or the leaderboard measures interpolation.

**5. The metric is `rmsle`,** which is what the competition grades, and unlike
Rossmann it is the metric auto-detection would have picked anyway. It is passed
explicitly regardless, because a run should not depend on a guess agreeing with
the truth.

Not done here, and deliberately left open:

* **`air_store_id` stays a string.** 829 identifiers, present in the forecast
  window, obviously predictive: it wants `fit_encoding`, which is the agent's
  call.
* **`air_area_name` is left whole.** It is a hierarchy flattened into one string
  ("Tōkyō-to Setagaya-ku Taishidō"), and splitting off the prefecture is a
  feature decision, not a parsing one.
* **No reservation column is filled.** Missing means no reservation was
  recorded, which is *usually* a true zero and, for the 515 air stores absent
  from the reservation file entirely, closer to "this store does not take
  bookings". Distinguishing those is imputation, and the harness has tools for
  it.
"""
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "restaurant_visitor")
OUT = os.path.join(RAW, "train_joined.parquet")
TEST_OUT = os.path.join(RAW, "test_joined.parquet")

#: The two reservation files, uncollapsed, keyed so they can join the spine.
#: Written by this script rather than pointed at in place, because the hpg one
#: genuinely needs a join to carry `air_store_id` at all — and because a run
#: should declare one kind of thing, not "this csv for one and that parquet for
#: the other".
AIR_STREAM_OUT = os.path.join(RAW, "air_reserve_stream.parquet")
HPG_STREAM_OUT = os.path.join(RAW, "hpg_reserve_stream.parquet")

#: Kept in the table, not removed — see point 3 of the module docstring.
LEAKING_COLUMNS: list[str] = []

#: The competition's published data dictionary, transcribed, plus — for the
#: reservation aggregates, which are ours and not the competition's — a statement
#: of what was summed and over what.
#:
#: The discipline 0029 sets is that this channel carries what the *data's owner*
#: knows and no amount of profiling recovers. It is not where a modeller's
#: conclusions go. "Genre is probably the strongest categorical" was written here
#: in a first draft and removed: that is the agent's to find.
#:
#: The one long entry, on `air_reserve_visitors`, earns its length because the
#: fact it states is unobtainable from this table: the *cut* in the reservation
#: stream is visible only by comparing training rows against the forecast window,
#: and the forecast window's reservations are not in this file.
COLUMN_NOTES = {
    "air_store_id": "the restaurant's identifier on AirREGI/Restaurant Board",
    "visit_date": "the date of the visits being counted",
    "visitors": "the number of visitors to the restaurant on that date (this is "
                "what you are predicting)",
    "air_genre_name": "the type of food the restaurant serves",
    "air_area_name": "the restaurant's area, as prefecture, ward and district in "
                     "one string",
    "latitude": "latitude of the area the restaurant is in, not of the "
                "restaurant itself",
    "longitude": "longitude of the area the restaurant is in, not of the "
                 "restaurant itself",
    "day_of_week": "the day of the week of visit_date",
    "holiday_flg": "whether visit_date is a holiday in Japan",
    "air_reserve_visitors": "total party size reserved on this store-date "
                            "through AirREGI, summed over reservations. The "
                            "reservation file is cut at the moment the forecast "
                            "is issued, so a row inside the training period can "
                            "include a table booked the same day (30% of "
                            "training reservations were booked with zero days' "
                            "notice) while a row in the forecast window can only "
                            "include bookings made 1 to 39 days ahead, more of "
                            "them the further out the date. The column exists at "
                            "prediction time; it is not made the same way there.",
    "air_reserve_count": "the number of AirREGI reservations for this store-date; "
                         "same timing caveat as air_reserve_visitors",
    "air_reserve_lead_days": "mean days between booking and visit over this "
                             "store-date's AirREGI reservations; same timing "
                             "caveat as air_reserve_visitors",
    "hpg_reserve_visitors": "total party size reserved on this store-date through "
                            "Hot Pepper Gourmet, for the 150 of 829 stores known "
                            "to be listed on both platforms; same timing caveat "
                            "as air_reserve_visitors",
    "hpg_reserve_count": "the number of Hot Pepper Gourmet reservations for this "
                         "store-date; same coverage and timing caveats",
    "hpg_reserve_lead_days": "mean days between booking and visit over this "
                             "store-date's Hot Pepper Gourmet reservations; same "
                             "coverage and timing caveats",
    "hpg_genre_name": "the store's genre as described on Hot Pepper Gourmet, "
                      "which classifies differently from AirREGI; known only for "
                      "the stores listed on both platforms",
    "hpg_area_name": "the store's area as described on Hot Pepper Gourmet; known "
                     "only for the stores listed on both platforms",
}

#: The run-level channel, kept short: anything derivable from the table is left
#: out, because handing over a conclusion drawn from two profiled columns is
#: modelling dressed as documentation.
#:
#: What survives is what no profile shows — that a row is a store-day, that the
#: task is forecasting forward rather than interpolating, and that a *missing*
#: store-day is a day the store did not report rather than a day with no
#: visitors. That last one is not derivable at all: the rows are not there to
#: profile.
ABOUT = (
    "Daily visitor counts for Japanese restaurants: one row per restaurant per "
    "day it reported, 2016-01-01 to 2017-04-22. A restaurant has no row on days "
    "it did not report — closed, or not yet on the platform — so the series are "
    "ragged rather than continuous. The task is forecasting visitor counts on "
    "later dates than any in this table, so a model that works by interpolating "
    "between days it has already seen will not do what is being asked of it."
)

#: What the competition grades. Passed explicitly rather than left to detection:
#: a run should not depend on the guess happening to agree.
METRIC = "rmsle"

#: The split has to respect time, or the leaderboard measures interpolation while
#: the competition measures forecasting.
SPLIT_ON = "visit_date"

#: Days between the end of history and the earliest row the competition asks for:
#: training stops 2017-04-22, the forecast window opens 2017-04-23.
#:
#: **It is the block's floor, not its width** (0047). The far edge is 39 days out,
#: so a lead of 1 is honest for the first day of the window and more generous than
#: the last day could ever be. Both are legitimate leads and the harness reports
#: `block_span` beside them; which to use is the agent's call (0029).
FORECAST_HORIZON = 1.0

#: The run's second inputs (0062). Declared here because which column is *when a
#: booking is for* and which is *when it was made* is a fact about the download
#: that no profiler recovers — and the wrong way round builds a feature out of the
#: future without failing.
#:
#: The notes are the owner's channel, same discipline as `COLUMN_NOTES`: what the
#: file is and how it was keyed, not what to do with it.
STREAMS = {
    "air_reserve": {
        "path": AIR_STREAM_OUT,
        "event_time": "visit_datetime",
        "observed_at": "reserve_datetime",
        "notes": "one row per AirREGI reservation, as downloaded: the party size "
                 "booked, the datetime of the visit it is for, and the datetime "
                 "the booking was made. Cut at the moment the forecast is issued, "
                 "so nothing booked after 2017-04-22 is in it.",
    },
    "hpg_reserve": {
        "path": HPG_STREAM_OUT,
        "event_time": "visit_datetime",
        "observed_at": "reserve_datetime",
        "notes": "one row per Hot Pepper Gourmet reservation, re-keyed onto "
                 "air_store_id through store_id_relation.csv — so it covers only "
                 "the 150 of 829 stores listed on both platforms. Same cut as "
                 "air_reserve.",
    },
}

#: The public leaderboard, downloaded from the competition. Used only to turn a
#: score into a rank, which is the unit every conclusion in this project is
#: actually about — 0.60 rmsle means nothing on its own, "rank 1400 of 2149" does.
#:
#: Lower is better here, unlike Porto Seguro's, and that difference is why
#: `leaderboard_rank` exists rather than each caller sorting it themselves.
#: **Both boards, and the private one is the default** — 0061. Every Rossmann
#: script hardcoded the public median while quoting private scores beside it, and
#: 0060 had made the identical mistake on this dataset an hour earlier. The public
#: board here is a third of the test rows; the private one decides the competition
#: and is what a submission is finally judged on.
LEADERBOARD_PUBLIC = os.path.join(
    RAW, "recruit-restaurant-visitor-forecasting-publicleaderboard-2026-08-14T18:31:32.csv"
)
LEADERBOARD_PRIVATE = os.path.join(
    RAW, "recruit-restaurant-visitor-forecasting-privateleaderboard-2026-08-14T18:31:06.csv"
)

#: Kept for callers written before both boards were read. Points at the public
#: file, which is what it always was.
LEADERBOARD_CSV = LEADERBOARD_PUBLIC


def leaderboard(private: bool = True) -> dict:
    """Best / top-10% / median off a board, read from disk rather than typed.

    Private by default, for `leaderboard_rank`'s reason. The first hand-typed
    version of these numbers on this dataset was wrong by 0.046 on the median,
    which is twice the whole distance from median to first place.
    """
    path = LEADERBOARD_PRIVATE if private else LEADERBOARD_PUBLIC
    if not os.path.exists(path):
        return {}
    s = pd.read_csv(path)["Score"].sort_values().reset_index(drop=True)
    n = len(s)
    return {"best": round(float(s.iloc[0]), 5),
            "top 10%": round(float(s.iloc[n // 10]), 5),
            "median": round(float(s.iloc[n // 2]), 5)}


def leaderboard_rank(score: float, private: bool = True) -> str:
    """Where an rmsle would have placed. **Pass a score from the same board.**

    Approximate on purpose — our holdout is not the competition's test set, so
    this is an order-of-magnitude reading and gets reported with a tilde. What it
    is for is answering "above or below the median", which sixteen runs could not
    answer without one of these files.

    The default is private because that is what a submission is finally judged on;
    passing a public score here silently answers a different question, which is
    the mistake this helper exists to make harder (0061).
    """
    path = LEADERBOARD_PRIVATE if private else LEADERBOARD_PUBLIC
    if not os.path.exists(path):
        return "no leaderboard file"
    s = pd.read_csv(path)["Score"].sort_values().reset_index(drop=True)
    rank = int((s < score).sum()) + 1
    board = "private" if private else "public"
    return f"rank ~{rank}/{len(s)} (top {rank / len(s):.0%}, {board})"


#: First date of the competition's forecast window. Everything at or after this
#: is prediction; everything before it is history — including, importantly, the
#: last reservation this dataset records.
FORECAST_START = pd.Timestamp("2017-04-23")


def _reserve_aggregate(path: str, key: str, prefix: str) -> pd.DataFrame:
    """Collapse a reservation stream onto (store, date), which is the table's grain.

    This is the step the harness cannot do — `fit_history` aggregates *within* a
    table, and there is no tool that takes a second file at a finer grain. Three
    statistics, chosen to be the least opinionated summary that is not lossy in an
    obvious way: how many people, how many bookings, and how far ahead. 0046 is
    the standing warning that a mean alone throws away the group.

    ``lead_days`` is measured from the booking to the *date* of the visit rather
    than to its hour, so "booked this morning for tonight" is zero days rather
    than a fraction. That is the resolution the joined table has.
    """
    r = pd.read_csv(path, parse_dates=["visit_datetime", "reserve_datetime"])
    r["visit_date"] = r["visit_datetime"].dt.normalize()
    r["lead_days"] = (
        r["visit_date"] - r["reserve_datetime"].dt.normalize()
    ).dt.days
    agg = (
        r.groupby([key, "visit_date"])
        .agg(
            **{
                f"{prefix}_reserve_visitors": ("reserve_visitors", "sum"),
                f"{prefix}_reserve_count": ("reserve_visitors", "size"),
                f"{prefix}_reserve_lead_days": ("lead_days", "mean"),
            }
        )
        .reset_index()
    )
    return agg


#: What a stream file has to carry, in the order the run declares them.
STREAM_COLUMNS = ["air_store_id", "visit_datetime", "reserve_datetime",
                  "reserve_visitors"]


def _hpg_rekeyed() -> pd.DataFrame:
    """Hot Pepper Gourmet reservations, carrying `air_store_id`.

    The relation table is the only bridge between the two platforms' identifiers
    and covers 150 of 829 stores, so this is an inner join and the result is a
    stream about a fifth of the estate. That is a fact about the data; a left join
    would produce records keyed on nothing.
    """
    rel = pd.read_csv(os.path.join(RAW, "store_id_relation.csv"))
    hpg = pd.read_csv(os.path.join(RAW, "hpg_reserve.csv"),
                      parse_dates=["visit_datetime", "reserve_datetime"])
    return hpg.merge(rel, on="hpg_store_id", how="inner", validate="many_to_one")


def write_streams() -> dict[str, pd.DataFrame]:
    """The two reservation files as streams — uncollapsed, timestamps untouched.

    Nothing is aligned to the table's grain here on purpose. The encoder measures
    the table's own row spacing and buckets both sides onto it (0063), so a
    booking stamped 18:00 finds its store-day without this script knowing that a
    row is a day. Handing over a pre-floored file would work and would put a rule
    in a place nobody using the library can read.

    The **cut is preserved**, which is the whole reason this dataset is worth
    running: the file stops at the moment the forecast is issued, so the records a
    forecast row could not have are genuinely absent rather than filtered out here.
    """
    air = pd.read_csv(os.path.join(RAW, "air_reserve.csv"),
                      parse_dates=["visit_datetime", "reserve_datetime"])
    hpg = _hpg_rekeyed()
    written = {}
    for name, frame, path in (
        ("air_reserve", air, AIR_STREAM_OUT),
        ("hpg_reserve", hpg, HPG_STREAM_OUT),
    ):
        out = frame[STREAM_COLUMNS].copy()
        out.to_parquet(path, index=False)
        written[name] = out
    return written


def _join_all(spine: pd.DataFrame) -> pd.DataFrame:
    """Attach every side table to a (air_store_id, visit_date) spine.

    Shared between the training file and the forecast file on purpose. A bundle
    rebuilds its features from raw columns, so anything done differently between
    the two becomes a silent train/serve skew — the exact failure the lineage
    replay exists to prevent, and one this dataset already supplies a natural
    example of (point 3 of the module docstring).
    """
    n = len(spine)

    store = pd.read_csv(os.path.join(RAW, "air_store_info.csv"))
    out = spine.merge(store, on="air_store_id", how="left", validate="many_to_one")

    dates = pd.read_csv(os.path.join(RAW, "date_info.csv"),
                        parse_dates=["calendar_date"])
    out = out.merge(dates, left_on="visit_date", right_on="calendar_date",
                    how="left", validate="many_to_one")
    out = out.drop(columns=["calendar_date"])

    air = _reserve_aggregate(os.path.join(RAW, "air_reserve.csv"),
                             "air_store_id", "air")
    out = out.merge(air, on=["air_store_id", "visit_date"], how="left",
                    validate="one_to_one")

    # hpg is keyed by its own store id; the relation table is the only bridge, and
    # it covers 150 of 829 stores. Mapping first, then aggregating, keeps the
    # aggregate at the air grain — two hpg ids mapping to one air id would
    # otherwise produce two rows to join against one.
    hpg = _hpg_rekeyed()
    hpg["visit_date"] = hpg["visit_datetime"].dt.normalize()
    hpg["lead_days"] = (
        hpg["visit_date"] - hpg["reserve_datetime"].dt.normalize()
    ).dt.days
    hpg_agg = (
        hpg.groupby(["air_store_id", "visit_date"])
        .agg(
            hpg_reserve_visitors=("reserve_visitors", "sum"),
            hpg_reserve_count=("reserve_visitors", "size"),
            hpg_reserve_lead_days=("lead_days", "mean"),
        )
        .reset_index()
    )
    out = out.merge(hpg_agg, on=["air_store_id", "visit_date"], how="left",
                    validate="one_to_one")

    # hpg's own description of the store, which classifies genre and area
    # differently from AirREGI. Kept because the disagreement between two
    # descriptions of the same restaurant is real information; known only for the
    # linked stores, which is what the dictionary says.
    rel = pd.read_csv(os.path.join(RAW, "store_id_relation.csv"))
    hpg_store = pd.read_csv(os.path.join(RAW, "hpg_store_info.csv"))
    hpg_store = hpg_store.merge(rel, on="hpg_store_id", how="inner",
                                validate="one_to_one")
    hpg_store = hpg_store.drop(columns=["hpg_store_id", "latitude", "longitude"])
    hpg_store = hpg_store.drop_duplicates(subset=["air_store_id"])
    out = out.merge(hpg_store, on="air_store_id", how="left",
                    validate="many_to_one")

    assert len(out) == n, "a join changed the row count"
    return out


def prep_test() -> pd.DataFrame:
    """The competition's forecast rows, put through the *same* joins.

    `sample_submission.csv` is the only statement of which (store, date) pairs are
    asked for, and its `id` is the two of them glued with an underscore — which
    the store id also contains, so the split has to come from the right.
    """
    sub = pd.read_csv(os.path.join(RAW, "sample_submission.csv"))
    parts = sub["id"].str.rsplit("_", n=1, expand=True)
    spine = pd.DataFrame({
        "id": sub["id"],
        "air_store_id": parts[0],
        "visit_date": pd.to_datetime(parts[1]),
    })
    joined = _join_all(spine)
    joined.to_parquet(TEST_OUT, index=False)
    return joined


def main() -> int:
    needed = [
        "air_visit_data.csv", "air_store_info.csv", "date_info.csv",
        "air_reserve.csv", "hpg_reserve.csv", "hpg_store_info.csv",
        "store_id_relation.csv", "sample_submission.csv",
    ]
    for name in needed:
        p = os.path.join(RAW, name)
        if not os.path.exists(p):
            print(f"missing {p} — unzip the competition data first", file=sys.stderr)
            return 1

    visits = pd.read_csv(os.path.join(RAW, "air_visit_data.csv"),
                         parse_dates=["visit_date"])
    print(f"visits {visits.shape}, {visits.air_store_id.nunique()} stores")
    print(f"dates {visits.visit_date.min().date()} -> {visits.visit_date.max().date()}")

    joined = _join_all(visits)

    dropped = [c for c in LEAKING_COLUMNS if c in joined.columns]
    joined = joined.drop(columns=dropped)

    joined.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  {joined.shape}")
    print(f"columns: {list(joined.columns)}")

    print(f"\ndescribed in COLUMN_NOTES: {len(COLUMN_NOTES)} of {joined.shape[1]} columns")
    undescribed = [c for c in joined.columns if c not in COLUMN_NOTES]
    if undescribed:
        print(f"NOT described: {undescribed}")

    test = prep_test()
    print(f"wrote {TEST_OUT}  {test.shape}")
    print(f"  in train and not in test: "
          f"{sorted(set(joined.columns) - set(test.columns))}")

    # The skew point 3 documents, printed rather than asserted: it is a property
    # of the competition's data, and a future re-download that no longer had it
    # would make the dictionary entry wrong.
    got_air = joined["air_reserve_count"].notna().mean()
    got_hpg = joined["hpg_reserve_count"].notna().mean()
    print(f"\nrows with an AirREGI reservation: {got_air:.1%}")
    print(f"rows with an hpg reservation:     {got_hpg:.1%}")
    tr_same_day = (joined["air_reserve_lead_days"] == 0).sum()
    te_same_day = (test["air_reserve_lead_days"] == 0).sum()
    print(f"store-days whose only bookings were same-day: "
          f"{tr_same_day} in train, {te_same_day} in the forecast window")

    streams = write_streams()
    print()
    for name, frame in streams.items():
        notice = (
            frame["visit_datetime"].dt.normalize()
            - frame["reserve_datetime"].dt.normalize()
        ).dt.days
        print(f"wrote {STREAMS[name]['path']}  {frame.shape}")
        # The quantity the whole capability turns on, printed rather than
        # asserted: it is a property of this download, and a future one without
        # the cut would make the dictionary entry wrong rather than the code.
        print(f"  {frame.air_store_id.nunique()} stores, "
              f"booked with 0 days' notice: {(notice <= 0).mean():.1%}, "
              f"median notice {notice.median():.0f} days")

    print("\nto start a run:")
    print("    from scripts.prep_restaurant import (ABOUT, COLUMN_NOTES, "
          "FORECAST_HORIZON,")
    print("                                         METRIC, SPLIT_ON, STREAMS)")
    print("    RunSession(run_dir, OUT,")
    print(f"               TaskSpec(target='visitors', metric='{METRIC}',")
    print("                        notes=ABOUT, column_notes=COLUMN_NOTES,")
    print(f"                        forecast_horizon={FORECAST_HORIZON}),")
    print(f"               split_on='{SPLIT_ON}', streams=STREAMS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
