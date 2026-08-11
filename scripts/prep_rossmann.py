"""Turn the Rossmann competition download into one table this harness can take.

Three things have to happen before a run on this dataset means anything, and each
of them is a decision rather than a chore:

**1. Two files become one.** `train.csv` (1.0M rows, one per store per day) and
`store.csv` (1115 rows, one per store) join on `Store`. The harness takes a single
table and has no join tool — deliberately, since a join is a modelling decision
about grain, not a feature transform — so it happens here, once, in the open.

**2. `Customers` is dropped, and this is the important one.** It correlates 0.895
with `Sales` and **does not exist in `test.csv`**: you cannot know today's customer
count when forecasting tomorrow's sales. Left in, it is a leak that every guard in
this harness permits — no fold boundary is crossed, no encoder is misfitted — and
that the validation half *confirms*, because the validation half is drawn from the
same file and has the column too. It is the `Id` failure from house-prices run 1
(0019's gotcha) with a far bigger number attached.

Nothing computable from `train.csv` reveals this. The only evidence is the
competition's own data dictionary and the column list of a test set the harness
never sees. That is what `column_notes` below is for.

**3. `Date` is kept as a column so the run can order its split on it.** Train ends
2015-07-31 and test begins 2015-08-01, so this is a forecast, not a random split.
Pass `split_on="Date"` or the leaderboard measures the wrong thing entirely.

Not done here, and deliberately left as open decisions:

* **RMSPE is not implemented** (`metrics.py` has no such metric), so a run will
  default to `rmsle` on this non-negative target. Different objective from Kaggle's.
* **Zero-sales rows are kept.** RMSPE ignores them; 17% of rows are zero sales and
  every one of them is a closed store (`Open == 0`). Whether to drop them changes
  what the model is *for* — forecasting turnover given the store is open is a
  different task from forecasting turnover including closures.
* **`Store` is left as an integer.** 1115 identifiers, genuinely predictive, and
  unlike `Id` it is present in the test set. It wants `fit_encoding`, which is the
  agent's call to make.
"""
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "rossmann_store_sales")
OUT = os.path.join(RAW, "train_joined.parquet")

#: Present in `train.csv`, absent from `test.csv`. Not a feature — an outcome
#: measured at the same time as the target.
LEAKING_COLUMNS = ["Customers"]

#: The competition's data dictionary, in the harness's channel. Only the columns
#: whose meaning is not recoverable from the name and the numbers: an entry that
#: restates the column name costs tokens on every run and teaches nothing.
COLUMN_NOTES = {
    "Store": "Store identifier, 1115 of them. A real categorical — the same store "
             "appears in the data you will be scored on, so store identity is a "
             "usable signal rather than a row number.",
    "Sales": "The turnover for one store on one day. This is the target.",
    "Open": "Was the store open that day: 0 = closed, 1 = open. Sales are 0 on "
            "every closed day.",
    "StateHoliday": "a = public holiday, b = Easter, c = Christmas, 0 = none. "
                    "Most stores close on state holidays.",
    "SchoolHoliday": "Whether this (store, date) was affected by public schools "
                     "being closed.",
    "StoreType": "Four different store models: a, b, c, d. Nominal, not ordered.",
    "Assortment": "Assortment level: a = basic, b = extra, c = extended. Arguably "
                  "ordered, and the order is basic < extra < extended.",
    "CompetitionDistance": "Metres to the nearest competitor store. Missing where "
                           "there is no known competitor, which is not the same as zero.",
    "CompetitionOpenSinceMonth": "Month the nearest competitor opened, with "
                                 "CompetitionOpenSinceYear. Approximate.",
    "CompetitionOpenSinceYear": "Year the nearest competitor opened. Together with "
                                "the month and Date this gives how long the "
                                "competitor has been open — which is the quantity "
                                "that matters, and it is not a column yet.",
    "Promo": "Whether the store ran a promotion that day.",
    "Promo2": "Whether the store takes part in Promo2, a continuing promotion: "
              "0 = not participating, 1 = participating.",
    "Promo2SinceWeek": "Calendar week the store joined Promo2.",
    "Promo2SinceYear": "Year the store joined Promo2.",
    "PromoInterval": "Months in which a Promo2 round restarts, e.g. "
                     "'Feb,May,Aug,Nov'. A comma-separated list, not a category — "
                     "it means something only next to the month of Date.",
    "DayOfWeek": "1 = Monday through 7 = Sunday.",
    # Addressed to the agent, which cannot set `split_on` — that is the run's
    # argument, fixed before any tool runs. What the agent *can* act on is that
    # the column is a date, that the split already respects it, and that a
    # datetime is dropped before training unless it is turned into numbers.
    "Date": "The day this row describes. The validation half is the *latest* "
            "dates, not a random sample, so a model that only works by "
            "interpolating between days it has seen will score badly. Dropped "
            "before training as a non-numeric column — day of week, month, year "
            "and day-of-year have to be derived from it to be usable.",
}

ABOUT = (
    "Rossmann drugstore sales: one row per store per day, 2013-01-01 to 2015-07-31. "
    "The real competition is scored on the following six weeks, so the task is "
    "forecasting forward in time, not interpolating. Scoring uses RMSPE and "
    "**ignores every day with zero sales** — 17% of these rows, all of them days "
    "the store was closed. The per-store table has been joined on already. One "
    "column of the original data, Customers, has been removed: it is measured on "
    "the same day as the target and does not exist at prediction time."
)


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
