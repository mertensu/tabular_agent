# 0029 — The harness owns the exam; folds respect the split

- **Date:** 2026-08-11
- **Status:** accepted, implemented
- **Related:** [0012](0012-harness-owns-the-stop.md) (the same boundary, drawn for
  stopping), [0019](0019-the-leaderboard-is-a-public-leaderboard.md) (which
  introduced the split this makes consistent),
  [0022](../../../docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md) /
  [0028](0028-the-yardstick-is-fixed-the-bar-is-not.md) (the yardstick this
  affects on ordered runs)

## Two questions that turned out to be one

Rossmann — a store-day panel, 1M rows, forecast forward in time — raised both.

**Ulf's question:** *"we are doing preprocessing which could in principle at least
be part of the agent's work. so the final result will depend on what we do now
which is kinda not what i want."* Correct, and the discomfort is the right
instinct: every setup decision we make is a decision the agent did not, and a
result that depends on our choices measures us as much as it.

**The technical question:** should a split be by time, or should it group by a
variable like store?

They resolve together, because the answer to both is the same line.

## The line

**The harness owns the evaluation protocol. The agent owns the modelling.**

Anything determining *how we measure* is fixed before the run starts, because an
agent that can change the exam can pass it without learning anything — the same
argument 0012 makes for the stop rule and 0019 makes for the validation half.
Anything determining *what the model sees* belongs to the agent, and taking it
away confounds the result.

Sorted that way, Rossmann's setup splits cleanly:

| decision | owner | why |
|---|---|---|
| metric (`rmspe`) | harness | the metric *is* the task definition |
| `split_on`, `val_fraction` | harness | the split is the exam |
| CV fold strategy | harness | ditto |
| column descriptions | harness | the data owner's knowledge, unobtainable otherwise |
| CSV typing artefacts | harness | I/O, not modelling |
| **which columns to use** | **agent** | including dropping one that leaks |
| **joining a dimension table** | **agent**, in principle | it is feature engineering |

The practical consequence: `Customers` — which correlates 0.895 with the target
and does not exist in the competition's test set — goes **back into the data**,
with the dictionary saying what it is. Whether the agent drops it becomes a
measured result rather than something we decided on its behalf. If it trains on
it, `val_score` will look excellent and we will have learned the dictionary
channel does not work, which is worth more than a clean number.

The join stays in prep, because there is no join tool. That is a real confound and
is recorded as one rather than explained away.

## Time or grouping?

**Hold out whatever deployment will hand you that is new.** New time → order the
split on time. New entities → group on the entity. Both → both.

For Rossmann the measurement settles it: 1115 stores in the work half, 1115 in
validation, **zero stores that appear only in validation**. Kaggle's test set is
the same stores on later dates. Grouping on `Store` would answer "how well do I
predict a store I have never seen", which is a real question about a different
product and would make the score worse for reasons unconnected to the task.

## What was actually wrong, and fixed

Ordering the *outer* split while leaving the *inner* folds random is incoherent,
and it was what the harness did.

**1. Cross-validation now forecasts when the run does.** `make_folds` takes an
`order=`; given one it builds forward-chaining folds — train on a prefix, validate
on the block that follows. Random k-fold on a panel lets a model predict Wednesday
having trained on Tuesday and Thursday *of the same store*: it interpolates rather
than forecasts, and scores far too well for it.

Both strategies are kept, and neither is a default to be embarrassed about.
Exchangeable rows are genuinely better served by random folds — every row gets an
out-of-fold prediction, every fold trains on the same amount of data, and the
estimate has lower variance. The strategy follows `split_on`, which is the
statement that the rows are ordered.

Measured on Rossmann: `cv_score` 0.371 against `val_score` 0.314. Under random
folds CV would have been the *optimistic* number; now it is the pessimistic one,
because early folds train on a fraction of the history. The optimism is gone and
what replaces it is honest.

**2. The split cuts on a value, not a row count.** At 1115 stores per day, cutting
at exactly 20% of rows put 603 of 2015-01-30's stores in the work half and 512 of
the same day in validation. Rows sharing a day share the promo calendar, the
holiday, the weather; the model learns that day's level from one group and applies
it to the other. 0.25% of validation rows, and exactly the leak the ordered split
exists to prevent.

**3. A table that drops the ordering column is refused.** Falling back to random
folds because a derived table happened to lose `Date` would silently change the
protocol mid-run.

## What it costs

**Good**

- `cv_score` and `val_score` now measure the same task, so their gap means
  overfitting rather than a change of question.
- `fold_spread` becomes a spread of *forecast* errors — which matters more than it
  looks, because 0028 withholds `val_se` on ordered splits and `fold_spread` is
  then the stop rule's only yardstick. It was measuring interpolation noise while
  the leaderboard ranked extrapolation.
- The ownership line is written down, so the next dataset's setup is a lookup
  rather than an argument.

**Bad**

- **Not every row gets an out-of-fold prediction.** The earliest block is never
  validated; on Rossmann that is 160,559 of 965,919 rows. They arrive as NaN in
  `oof.parquet` and as `fold = -1` in `explore_experiment`'s frame. Anything that
  assumed one prediction per row has to carry that, and `explore_experiment`'s
  docstring promise of positional alignment is now "aligned, some missing".
- **Out-of-fold target encoding leaves that block NaN too**, so on an ordered run
  roughly `1/(n_splits+1)` of training rows have no encoded value for a
  target-encoded column. The missingness is time-correlated, which a tree can
  learn as "early period". Not fixed: every alternative computes the value from
  the row's own future or from itself.
- **Fold training sets are unequal**, so `fold_spread` mixes noise with "how much
  does more history help". Still better than measuring the wrong quantity.
- **`val_fraction` becomes approximate** under an ordered split — it lands on the
  nearest value boundary. Negligible at 1115 rows per day, coarse on a column with
  few distinct values, which is now an error rather than a silent bad split.
- The join is still ours, and Rossmann results carry that asterisk.

## Where it lives

- `cv.py` — `make_folds(order=)`, `boundary_cut`, `_boundary_cuts`.
- `harness/session.py` — the value-boundary work/validation split.
- `harness/tools.py` — `_fold_order`, threaded into training, encoding,
  `explain_table` and the residual frame.
- `models.py` — `oof` initialised NaN rather than zero.
