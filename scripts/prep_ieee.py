"""Turn the IEEE-CIS Fraud Detection download into one table this harness can take.

The fifth dataset, and the reason for it is that the other two large boards are
spent (0045, 0066): the harness currently has no way to check its own answer. This
competition is finished and **both leaderboards are on disk**, so one honest run
gets one honest reading.

Everything dataset-specific about IEEE-CIS lives in this file (`CLAUDE.md`).
Five things happen here, and each is a decision rather than a chore:

**1. Two files become one, and the join is LEFT.** `train_transaction.csv`
(590,540 rows) carries the target; `train_identity.csv` (144,233 rows) carries
network and device information for **24% of transactions only**. An inner join
would silently throw away three quarters of the data. So the join is left, and
the 76% of rows with no identity record get NaN across all 41 identity columns —
which is what the data means: the identity system had nothing on that transaction.
The harness has no join tool on purpose (a join is a decision about grain), so it
happens here, once, in the open.

**2. The test identity file spells its columns differently, and that is a trap.**
`train_identity.csv` has `id_01`; `test_identity.csv` has `id-01`. Hyphens. It is
a genuine quirk of the released files, and left alone it produces exactly the
failure 0055 records: a bundle whose stored lineage asks for `id_01` at predict
time, on rows that carry `id-01`, refused by 0030 after the run is over. Both
sides are normalised to underscores here, in the one place that defines what a row
of this dataset looks like.

**3. `TransactionDT` is a timedelta, not a timestamp, and it stays as it is.** The
competition says so explicitly: it is seconds from an undisclosed reference
datetime. So the rows are ordered and the distances between them are real, but no
calendar date is recoverable — which day is a Monday is unknown, and nothing
external can be joined to it. `SPLIT_ON="TransactionDT"` orders the split; the
column is left in seconds rather than converted to days, because a conversion here
would put this file's arithmetic between the agent and the units every other
number in the run is quoted in (0051 forbids guessing units; this is the same
argument for not silently changing them).

**4. The 30-day gap is declared, in the ordering column's units.** Training ends
at 15,811,131 and the test block starts at 18,403,224 — a gap of **2,592,093
seconds, 30.0 days**, with no rows released in between. That is what
`FORECAST_HORIZON` says, and it is 2,592,000 rather than 30 because
`TaskSpec.forecast_horizon` is expressed in the ordering column's units. Getting
that wrong by a factor of 86,400 would not raise anything; it would quietly make
every lag offset look answerable.

**5. Nothing is typed away from what the source says.** The competition *declares*
49 columns categorical (see `DECLARED_CATEGORICAL`), and 18 of them arrive as
numeric dtype — `card1` is an int with 13,553 distinct values. Casting those to
string here would force the agent's hand: strings are dropped by the fit unless
encoded, so the cast would remove label encoding from the menu without saying so.
Casting them is available and sanctioned as prep's job, and it is deliberately
**not done**. What this file does instead is transcribe the declaration into
`COLUMN_NOTES`, where it is a fact about the data the agent can act on or not.
See 0074 for the capability that does not exist yet.

Not done here, and deliberately left open:

* **`TransactionID` is kept.** It exists in the test file, so it is genuinely
  available at predict time and 0030 has no objection to it. What makes it
  interesting is stated in `COLUMN_NOTES` as a fact and not as advice: it
  increases with `TransactionDT` (r = 0.998) and every test value is larger than
  every training value. This is the `Id` trap from house-prices run 1 (0019) and
  the `Customers` trap from Rossmann, in its third shape — and, as there, whether
  to use a column is the agent's decision (0029).
* **The 339 `V` columns are kept in full.** They are Vesta's engineered features
  and heavily redundant; deciding that here would be feature selection performed
  by the prep script.
* **No sampling.** 590,540 rows x 434 columns is what the competition is. If the
  run cannot afford it, that is a reading about the harness, which is the point.
"""
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "fraud_detection")
OUT = os.path.join(RAW, "train_joined.parquet")
TEST_OUT = os.path.join(RAW, "test_joined.parquet")

#: The 49 columns the competition's data page declares categorical, transcribed
#: and nothing more. **31 of them are strings and 18 are numeric dtype**, which is
#: a fact about the CSV rather than about the data: `card1` is an arbitrary
#: identifier that happens to have been written as an integer, and `ProductCD` is
#: an arbitrary identifier that happens to have been written as a letter.
#:
#: This list is transcription, not analysis. What follows from it — whether to
#: encode `card1`, whether to leave it as an ordered integer, whether `id_19`'s
#: 522 levels are worth the columns — is the agent's to work out.
DECLARED_CATEGORICAL = (
    ["ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 10)]
    + ["DeviceType", "DeviceInfo"]
    + [f"id_{i:02d}" for i in range(12, 39)]
)

#: The competition's published data dictionary, transcribed. It is unusually thin
#: on purpose — the host wrote *"the field names are masked and a pairwise
#: dictionary will not be provided for privacy protection and contract
#: agreement"* — and the thinness is faithful. Inventing meanings for the masked
#: columns here would be handing over guesses in the one channel the agent is
#: entitled to trust.
#:
#: The families (`C*`, `D*`, `M*`, `V*`, `id_*`) get one entry each rather than
#: one per column: 434 entries of near-identical text is 0018's budget spent on
#: repetition. Only the columns whose meaning is not recoverable from the name and
#: the profile are described.
_NOTES: dict[str, str] = {
    "TransactionID": (
        "the identifier of the transaction. Present in the test file too, so it "
        "is available at prediction time. It increases with TransactionDT "
        "(r = 0.998) and every value in the test block is larger than every value "
        "in this table."
    ),
    "isFraud": "the target: whether the transaction was reported fraudulent.",
    "TransactionDT": (
        "a timedelta from a given reference datetime, NOT an actual timestamp. "
        "The unit is seconds. The reference datetime is not disclosed, so no "
        "calendar date, weekday or month is recoverable from it; differences "
        "between rows are exact."
    ),
    "TransactionAmt": "transaction payment amount in USD.",
    "ProductCD": "product code: the product for each transaction.",
    "card1": "payment card information, such as card type, card category, issue "
             "bank, country, etc. (card1-card6; the individual meanings are masked)",
    "card2": "payment card information (masked; see card1).",
    "card3": "payment card information (masked; see card1).",
    "card4": "payment card information (masked; see card1).",
    "card5": "payment card information (masked; see card1).",
    "card6": "payment card information (masked; see card1).",
    "addr1": "address (addr1, addr2; the individual meanings are masked).",
    "addr2": "address (masked; see addr1).",
    "dist1": "distance (dist1, dist2; the individual meanings are masked).",
    "dist2": "distance (masked; see dist1).",
    "P_emaildomain": "purchaser email domain.",
    "R_emaildomain": "recipient email domain.",
    "DeviceType": "identity information: the kind of device (see id_01).",
    "DeviceInfo": "identity information: the device (see id_01).",
}

#: The masked families, one sentence each, expanded over every member. 339 `V`
#: columns saying the same thing is repetition (0018), and leaving 338 of them
#: undescribed is worse: an undescribed column reads as one nobody thought about,
#: where what is true here is that nobody outside Vesta *can* describe it. The
#: repetition buys the reader the right question — "is this column masked, or did
#: the dictionary just not reach it" — on whichever column they happen to be
#: looking at.
_FAMILIES = [
    (
        [f"C{i}" for i in range(1, 15)],
        "C1-C14 are counts, such as how many addresses are found to be associated "
        "with the payment card. The actual meaning is masked.",
    ),
    (
        [f"D{i}" for i in range(1, 16)],
        "D1-D15 are timedeltas, such as days between the previous transaction and "
        "this one. The actual meaning is masked.",
    ),
    (
        [f"M{i}" for i in range(1, 10)],
        "M1-M9 are matches, such as whether the name on the card matches the name "
        "on the address. The actual meaning is masked.",
    ),
    (
        [f"V{i}" for i in range(1, 340)],
        "V1-V339 are features engineered by Vesta, including ranking, counting and "
        "other entity relations. The actual meanings are masked.",
    ),
    (
        [f"id_{i:02d}" for i in range(1, 39)],
        "id_01-id_38 are identity information: network connection information (IP, "
        "ISP, proxy) and digital signature (user agent, browser, OS, version) "
        "associated with the transaction, collected by Vesta's fraud protection "
        "system and its digital security partners. The field names are masked and "
        "no dictionary is provided. Present for 24% of transactions; missing "
        "everywhere else because there was no identity record, not because a value "
        "was lost.",
    ),
]

#: The declaration rides on the column it is about, appended to whatever else is
#: known about that column.
#:
#: An earlier draft put it in one `_declared_categorical` entry listing all 49
#: names. That entry would never have been shown to anyone: `profile_dataframe`
#: attaches a note to its column and drops any key that is not one, so a summary
#: entry is a note written to nobody. Worth recording because it looks like it
#: works — the dictionary is well-formed and the text is correct.
DECLARED_NOTE = " The competition declares this column categorical."

COLUMN_NOTES = {
    **{c: note for members, note in _FAMILIES for c in members},
    **_NOTES,
}
COLUMN_NOTES = {
    c: note + (DECLARED_NOTE if c in set(DECLARED_CATEGORICAL) else "")
    for c, note in COLUMN_NOTES.items()
}

#: What the run-level channel says. Two facts, both published by the competition
#: host, and neither recoverable from the table:
#:
#: **The labelling rule.** The host stated that a chargeback report on a card
#: labels not only the reported transaction but every later transaction on that
#: card, account, email or billing address. That makes the label a property of an
#: entity over time rather than of an isolated row, which is a fact about what is
#: being predicted.
#:
#: **The forecast geometry.** Predictions are asked for on a block that starts 30
#: days after this table ends, so a model that works by interpolating between
#: transactions it has already seen is not doing what is asked. Same sentence as
#: Rossmann's, because it is the same situation and the wording earned its keep.
ABOUT = (
    "Vesta payment transactions with an identity record attached where one "
    "exists: one row per transaction, ordered by TransactionDT, spanning 182 "
    "days. The target is whether a transaction was reported fraudulent. The "
    "competition host states that the label follows the entity, not the "
    "transaction: when a chargeback is reported on a card, that transaction and "
    "every later transaction on the same card, account, email address or billing "
    "address are labelled fraudulent, so a fraudulent row is often the "
    "consequence of an earlier report rather than an independent event. "
    "Predictions are asked for on a block of transactions beginning 30 days after "
    "the last row in this table, so a model that works by interpolating between "
    "transactions it has already seen will not do what is being asked of it."
)

#: What the competition actually grades: area under the ROC curve. Passed
#: explicitly rather than detected — auto-detection would choose `logloss` for a
#: binary target, which is a reasonable default and the wrong objective here.
METRIC = "auc"

#: The split has to respect time. The competition's test block is a later period
#: with a gap; a random split would measure interpolation.
SPLIT_ON = "TransactionDT"

#: The gap between the end of this table and the first row a prediction is asked
#: for, **in the ordering column's units** — seconds, not days (0051). Measured
#: from the files: test starts at 18,403,224, training ends at 15,811,131.
#:
#: Rounded down to the whole day it is reported as: the exact difference is
#: 2,592,093 s, and the extra 93 seconds is the distance to the first *observed*
#: test transaction rather than to the start of the forecast block. The field is
#: documented as a floor, so rounding up would be the direction that lies.
FORECAST_HORIZON = 30 * 24 * 3600  # 2_592_000

#: Both boards, read from the files rather than typed. 0061 is the record of this
#: project comparing a private score against a public median for weeks; the fix is
#: that no constant is transcribed at all.
LEADERBOARD_PUBLIC = os.path.join(
    RAW, "ieee-fraud-detection-publicleaderboard-2026-08-17T17:24:32.csv"
)
LEADERBOARD_PRIVATE = os.path.join(
    RAW, "ieee-fraud-detection-privateleaderboard-2026-08-17T17:24:28.csv"
)


def leaderboard(private: bool = True) -> dict:
    """Best / top-10% / median off a board. Private by default — it is the one
    that decides the competition, and the one our submissions are scored on.

    **AUC is a higher-is-better metric**, unlike every other dataset in this repo,
    so the sort is descending and `top 10%` is the 10th percentile from the top.
    """
    path = LEADERBOARD_PRIVATE if private else LEADERBOARD_PUBLIC
    if not os.path.exists(path):
        return {}
    s = pd.read_csv(path)["Score"].sort_values(ascending=False).reset_index(drop=True)
    n = len(s)
    return {"best": round(float(s.iloc[0]), 5),
            "top 10%": round(float(s.iloc[n // 10]), 5),
            "median": round(float(s.iloc[n // 2]), 5),
            "teams": n}


def leaderboard_rank(score: float, private: bool = True) -> str:
    """Where an AUC would have placed. **Pass a score from the same board.**

    Higher is better here, so the rank counts the teams that scored *above* the
    given number — the opposite of the rmspe/rmsle helpers in the other prep
    scripts, which is exactly the sort of asymmetry that gets copy-pasted wrong.
    """
    path = LEADERBOARD_PRIVATE if private else LEADERBOARD_PUBLIC
    if not os.path.exists(path):
        return "no leaderboard file"
    s = pd.read_csv(path)["Score"]
    rank = int((s > score).sum()) + 1
    board = "private" if private else "public"
    return f"rank ~{rank}/{len(s)} (top {rank / len(s):.0%}, {board})"


def _normalise_identity(identity: pd.DataFrame) -> pd.DataFrame:
    """`id-01` -> `id_01`. See point 2 of the module docstring.

    Applied to both files rather than only to the test one, so that the naming is
    this script's and not inherited from whichever file happened to be read.
    """
    return identity.rename(columns={c: c.replace("-", "_") for c in identity.columns})


def _join(transaction: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Left join on TransactionID, with the row count asserted.

    An inner join would drop 76% of the transactions and would do it silently; a
    missing identity record must show up as missing columns, not as fewer rows.
    """
    identity = _normalise_identity(identity)
    joined = transaction.merge(
        identity, on="TransactionID", how="left", validate="one_to_one"
    )
    assert len(joined) == len(transaction), "the join changed the row count"
    return joined


def prep_test() -> pd.DataFrame:
    """The competition's test rows, put through the *same* join and normalisation.

    Kept here rather than in the runner so there is one definition of what a row
    of this dataset looks like. A bundle rebuilds its features from raw columns,
    so anything done differently between the two files becomes a train/serve skew
    the lineage replay cannot see (0055).

    The only difference from the training file is the data's own: there is no
    `isFraud` column, because that is the submission.
    """
    transaction = pd.read_csv(os.path.join(RAW, "test_transaction.csv"),
                              low_memory=False)
    identity = pd.read_csv(os.path.join(RAW, "test_identity.csv"), low_memory=False)
    joined = _join(transaction, identity)
    joined.to_parquet(TEST_OUT, index=False)
    return joined


def main() -> int:
    paths = {n: os.path.join(RAW, f"{n}.csv") for n in
             ("train_transaction", "train_identity", "test_transaction",
              "test_identity")}
    for p in paths.values():
        if not os.path.exists(p):
            print(f"missing {p} — download the competition data first", file=sys.stderr)
            return 1

    transaction = pd.read_csv(paths["train_transaction"], low_memory=False)
    identity = pd.read_csv(paths["train_identity"], low_memory=False)
    print(f"transaction {transaction.shape}, identity {identity.shape} "
          f"({len(identity) / len(transaction):.0%} of transactions)")

    joined = _join(transaction, identity)
    joined.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  {joined.shape}")

    test = prep_test()
    print(f"wrote {TEST_OUT}  {test.shape}")
    print(f"  in train and not in test: "
          f"{sorted(set(joined.columns) - set(test.columns))}")
    print(f"  in test and not in train: "
          f"{sorted(set(test.columns) - set(joined.columns))}")

    # The two numbers FORECAST_HORIZON is derived from, recomputed from the files
    # every time this runs, so the constant cannot drift away from the data.
    gap = int(test["TransactionDT"].min() - joined["TransactionDT"].max())
    def span_days(s: pd.Series) -> float:
        return float(s.max() - s.min()) / 86400

    print(f"\nordering on {SPLIT_ON}: train "
          f"{joined['TransactionDT'].min()} -> {joined['TransactionDT'].max()} "
          f"({span_days(joined['TransactionDT']):.1f} days)")
    print(f"  test  {test['TransactionDT'].min()} -> {test['TransactionDT'].max()} "
          f"({span_days(test['TransactionDT']):.1f} days)")
    print(f"  gap {gap} s = {gap / 86400:.2f} days "
          f"| FORECAST_HORIZON declared {FORECAST_HORIZON}")
    assert FORECAST_HORIZON <= gap, "the declared horizon must not exceed the real gap"

    print(f"\ntarget: {joined['isFraud'].mean():.4%} positive "
          f"({int(joined['isFraud'].sum())} of {len(joined)})")

    declared = [c for c in DECLARED_CATEGORICAL if c in joined.columns]
    numeric = [c for c in declared if pd.api.types.is_numeric_dtype(joined[c])]
    strings = [c for c in declared if c not in numeric]
    print(f"\ndeclared categorical: {len(declared)} columns — "
          f"{len(strings)} stored as strings (dropped by the fit unless encoded), "
          f"{len(numeric)} as numbers (kept, and split as ordered quantities)")
    print("  widest: " + ", ".join(
        f"{c}({joined[c].nunique()})"
        for c in sorted(declared, key=lambda c: -joined[c].nunique())[:5]))
    missing_declared = [c for c in DECLARED_CATEGORICAL if c not in joined.columns]
    if missing_declared:
        print(f"  !! declared but not in the table: {missing_declared}")

    described = [c for c in joined.columns if c in COLUMN_NOTES]
    print(f"\ndescribed in COLUMN_NOTES: {len(described)} of {joined.shape[1]} "
          "columns")
    undescribed = [c for c in joined.columns if c not in COLUMN_NOTES]
    if undescribed:
        print(f"  NOT described: {undescribed}")

    # Not a defect of this file, and it is the reason it is printed here. The
    # profile the agent reads is capped at 100 columns, and this table has 434 in
    # the competition's own order — so every identity column, including the 29
    # declared categorical ones, falls off the end of `inspect_data`. Reordering
    # the columns here would hide that behind a prep script. See the runner's
    # registered predictions.
    print(f"\ncolumn order is the competition's: {list(joined.columns)[:6]} ... "
          f"{list(joined.columns)[-4:]}")
    print(f"  column 100 of {joined.shape[1]} is {joined.columns[99]!r} — "
          "inspect_data's profile stops there (MAX_REPORTED_COLUMNS)")

    print("\nto start a run:")
    print("    from scripts.prep_ieee import ABOUT, COLUMN_NOTES, METRIC, SPLIT_ON")
    print("    RunSession(run_dir, OUT,")
    print(f"               TaskSpec(target='isFraud', metric='{METRIC}',")
    print("                        notes=ABOUT, column_notes=COLUMN_NOTES,")
    print(f"                        forecast_horizon={FORECAST_HORIZON}),")
    print(f"               split_on='{SPLIT_ON}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
