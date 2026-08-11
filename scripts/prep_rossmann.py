"""Turn the Rossmann competition download into one table this harness can take.

Three things have to happen before a run on this dataset means anything, and each
of them is a decision rather than a chore:

**1. Two files become one.** `train.csv` (1.0M rows, one per store per day) and
`store.csv` (1115 rows, one per store) join on `Store`. The harness takes a single
table and has no join tool — deliberately, since a join is a modelling decision
about grain, not a feature transform — so it happens here, once, in the open.

**2. `Customers` is kept, and it is a trap on purpose.** It correlates 0.895 with
`Sales` and **does not exist in `test.csv`**: you cannot know today's customer count
when forecasting tomorrow's sales. Trained on, it is a leak that every guard in this
harness permits — no fold boundary crossed, no encoder misfitted — and that the
validation half *confirms*, because validation is drawn from the same file and has
the column too. It is the `Id` failure from house-prices run 1 (0019's gotcha) with
a far bigger number attached.

An earlier version of this script dropped it. That was the harness doing the
agent's work (0029): whether to use a column is a modelling decision, and deciding
it here means the run measures our judgement rather than the agent's. So the column
stays and the dictionary says plainly that it is unavailable at prediction time.

If the agent trains on it anyway, `val_score` will look excellent and we will have
learned that the data-dictionary channel does not change behaviour — which is worth
more than a clean number, and cannot be learned from a file that never had the
column.

**3. `Date` is kept as a column so the run can order its split on it.** Train ends
2015-07-31 and test begins 2015-08-01, so this is a forecast, not a random split.
Pass `split_on="Date"` or the leaderboard measures the wrong thing entirely.

**4. Zero-sales rows are kept, and the metric is `rmspe`.** 17% of rows are days a
store was closed and sold nothing. RMSPE does not score them — the relative error
of zero is undefined — and that exclusion lives inside the metric, so the rows stay
visible to the model. Deleting them here would be the harness deciding something
the agent should work out: `inspect_data` reports what the metric measures,
including that zero-valued rows are unscored, and what to do about closed days is
then a modelling decision with the information needed to make it.

Not done here, and deliberately left open:

* **`Store` is left as an integer.** 1115 identifiers, genuinely predictive, and
  unlike `Id` it is present in the test set. It wants `fit_encoding`, which is the
  agent's call to make.
* **No `val_se` on this dataset.** The bootstrap assumes exchangeable validation
  rows and an ordered split gives a later *period*; the harness withholds the
  estimate rather than reporting one that is confidently too small, and the stop
  rule falls back to `fold_spread`.
"""
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "rossmann_store_sales")
OUT = os.path.join(RAW, "train_joined.parquet")

#: Kept in the table, not removed. See point 2 of the module docstring: which
#: columns to use is the agent's decision, and the dictionary gives it what it
#: needs to make this one.
LEAKING_COLUMNS: list[str] = []

#: The competition's published data dictionary, transcribed and nothing more.
#:
#: The discipline here is the point, and the first draft of this file broke it.
#: This is what the *data's owner* documented — the meaning of a column, which no
#: amount of profiling recovers. It is not the place for what a modeller thinks
#: about the column. Entries like "a real categorical, not a row number" or
#: "arguably ordered, and the order is basic < extra < extended" were written here
#: and removed again: they are conclusions the agent is supposed to reach, and
#: putting them in the dictionary is doing its work while calling it documentation.
#:
#: `DayOfWeek` and `Date` have no entry, faithfully — the source says "most of the
#: fields are self-explanatory" and describes only the rest. `Id` is described
#: there and absent here, because that column does not exist in this table.
COLUMN_NOTES = {
    "Store": "a unique Id for each store",
    "Sales": "the turnover for any given day (this is what you are predicting)",
    # Verbatim from the competition, plus the one fact the competition states by
    # omission: the column is in the training file and not in the test file. That
    # is not an opinion about modelling, it is what the data is — and it is
    # unobtainable from the training table, which is the whole reason this channel
    # exists.
    "Customers": "the number of customers on a given day. This column exists in "
                 "the training data only: it is measured on the same day as the "
                 "target and is NOT available at prediction time.",
    "Open": "an indicator for whether the store was open: 0 = closed, 1 = open",
    "StateHoliday": "indicates a state holiday. Normally all stores, with few "
                    "exceptions, are closed on state holidays. Note that all "
                    "schools are closed on public holidays and weekends. "
                    "a = public holiday, b = Easter holiday, c = Christmas, 0 = None",
    "SchoolHoliday": "indicates if the (Store, Date) was affected by the closure "
                     "of public schools",
    "StoreType": "differentiates between 4 different store models: a, b, c, d",
    "Assortment": "describes an assortment level: a = basic, b = extra, c = extended",
    "CompetitionDistance": "distance in meters to the nearest competitor store",
    "CompetitionOpenSinceMonth": "gives the approximate month of the time the "
                                 "nearest competitor was opened",
    "CompetitionOpenSinceYear": "gives the approximate year of the time the "
                                "nearest competitor was opened",
    "Promo": "indicates whether a store is running a promo on that day",
    "Promo2": "Promo2 is a continuing and consecutive promotion for some stores: "
              "0 = store is not participating, 1 = store is participating",
    "Promo2SinceWeek": "describes the calendar week when the store started "
                       "participating in Promo2",
    "Promo2SinceYear": "describes the year when the store started participating "
                       "in Promo2",
    "PromoInterval": "describes the consecutive intervals Promo2 is started, "
                     "naming the months the promotion is started anew. E.g. "
                     '"Feb,May,Aug,Nov" means each round starts in February, May, '
                     "August, November of any given year for that store",
}

#: What the run-level channel says, and it is short because almost everything a
#: first draft put here failed one of two tests.
#:
#: **Derivable?** Then leave it out. "About 17% of rows are days a store was closed
#: and sold nothing" is `Sales == 0` next to `Open == 0`, sitting in the data, and
#: `inspect_data` profiles both columns. The agent is separately told — by
#: `metric_measures` — that rmspe does not score zero-valued rows. Handing over the
#: conclusion drawn from those two facts is doing the modelling and calling it
#: documentation.
#:
#: **Actionable?** Then keep it; otherwise it is prose for a human. "The per-store
#: table has been joined on already" answers a question no agent can ask, since it
#: never sees `store.csv`. What matters about `Customers` is a fact about one
#: column, so it rides on that column in `COLUMN_NOTES` rather than sitting here
#: where it would have to be matched up by name.
#:
#: What survives is the one thing that is neither: that a row is a store-day and
#: the task is to forecast forward rather than interpolate. The split record says
#: `mode: ordered`, which is the mechanism; this says what it is *for*.
ABOUT = (
    "Daily sales for Rossmann drugstores: one row per store per day, "
    "2013-01-01 to 2015-07-31. The task is forecasting sales on later dates than "
    "any in this table, so a model that works by interpolating between days it has "
    "already seen will not do what is being asked of it."
)

#: What the competition actually grades. Passed explicitly because auto-detection
#: would choose `rmsle` for a non-negative target — a reasonable default, and the
#: wrong objective here.
METRIC = "rmspe"

#: The split has to respect time or the leaderboard measures interpolation while
#: the competition measures forecasting.
SPLIT_ON = "Date"


def main() -> int:
    train_csv = os.path.join(RAW, "train.csv")
    store_csv = os.path.join(RAW, "store.csv")
    for p in (train_csv, store_csv):
        if not os.path.exists(p):
            print(f"missing {p} — download the competition data first", file=sys.stderr)
            return 1

    # `low_memory=False`: StateHoliday holds both the string "0" and the integer 0
    # in different chunks, and a chunked read types the column differently
    # depending on where the chunk boundaries fall.
    train = pd.read_csv(train_csv, low_memory=False, parse_dates=["Date"])
    store = pd.read_csv(store_csv, low_memory=False)

    print(f"train {train.shape}, store {store.shape}")
    print(f"dates {train.Date.min().date()} -> {train.Date.max().date()}")

    # Left join: every sales row is kept even if its store is somehow absent from
    # the store table. An inner join would drop rows silently, which is the wrong
    # failure — a missing store should show up as missing columns, not fewer rows.
    joined = train.merge(store, on="Store", how="left", validate="many_to_one")
    assert len(joined) == len(train), "the join changed the row count"

    dropped = [c for c in LEAKING_COLUMNS if c in joined.columns]
    joined = joined.drop(columns=dropped)

    # StateHoliday's mixed "0"/0 is a genuine ambiguity in the source, not a
    # parsing artefact: settle it here rather than leaving the agent to discover
    # that one category is two.
    joined["StateHoliday"] = joined["StateHoliday"].astype(str)

    joined.to_parquet(OUT, index=False)

    print(f"dropped as unavailable at predict time: {dropped}")
    print(f"wrote {OUT}  {joined.shape}")
    print(f"columns: {list(joined.columns)}")
    print(f"\ndescribed in COLUMN_NOTES: {len(COLUMN_NOTES)} of {joined.shape[1]} columns")
    undescribed = [c for c in joined.columns if c not in COLUMN_NOTES]
    if undescribed:
        print(f"NOT described: {undescribed}")

    zero = (joined["Sales"] == 0).mean()
    print(f"\nzero-sales rows kept: {zero:.1%} (rmspe does not score them)")
    print("\nto start a run:")
    print("    from scripts.prep_rossmann import ABOUT, COLUMN_NOTES, METRIC, SPLIT_ON")
    print("    RunSession(run_dir, OUT,")
    print(f"               TaskSpec(target='Sales', metric='{METRIC}',")
    print("                        notes=ABOUT, column_notes=COLUMN_NOTES),")
    print(f"               split_on='{SPLIT_ON}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
