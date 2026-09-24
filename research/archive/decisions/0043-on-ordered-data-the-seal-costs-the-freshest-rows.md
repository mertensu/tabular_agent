# 0043 — On ordered data the seal costs the freshest rows, not a random tenth

- **Date:** 2026-08-13
- **Status:** accepted (a correction)
- **Corrects:** [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md) —
  its decision stands; its "What it costs" section is incomplete
- **Related:** [0023](0023-the-third-split-has-evidence-now.md),
  [0029](0029-the-harness-owns-the-exam.md),
  [0042](0042-the-terminal-argmax-shipped-the-worst-model.md)

## What 0041 got wrong

0041 priced the third split in **rows**: "on house-prices the work half drops
1168 → 876 at 0.2. The agent builds on 25% less data." That is the whole cost
under a *random* split, where the rows removed are a uniform sample and the only
thing lost is sample size.

Under an **ordered** split it is not the whole cost, and the difference is not
small. The slices are `work | val | holdout` in time order, so the sealed rows are
not a tenth of the data — they are **the most recent tenth**, and on a forecasting
task recency is most of what a row is worth.

## The measurement

Rossmann v7, `val_fraction=0.05`, `holdout_fraction=0.05`:

| | period |
|---|---|
| work — everything the shipped model trained on | 2013-01-01 → **2015-05-01** |
| validation | 2015-05-02 → 2015-06-15 |
| sealed holdout | 2015-06-16 → **2015-07-31** |
| **Kaggle's test set** | **2015-08-01 → 2015-09-17** |

The competition supplies training data through 2015-07-31 and asks for the six
weeks after it. The model this run shipped **has never seen May, June or July**.
It is forecasting three to five months past the end of its own data, against a
field whose submissions forecast six weeks.

The row count says the seal cost 5%. The horizon says it doubled the forecast
distance. Those are not the same statement, and only the second one predicts what
happens on the leaderboard.

## What follows

- **A sealed ordered run systematically understates what the harness could
  ship.** Any Kaggle number from `_runs/rossmann_v7` is a reading on a
  deliberately handicapped model, and comparing it to v5 — whose work half ran
  ~6 weeks later — is comparing two different forecast horizons, not two
  harnesses. v5's Kaggle 0.135 is **not** a fair control for v7's submission.
- **`holdout_fraction` is a different knob on ordered data than on random data.**
  On exchangeable rows it trades sample size for an honest number, and 0.2 is
  cheap. On a time series it trades *recency*, and the same 0.2 can be very
  expensive — Rossmann at 0.2/0.2 would end the work half in October 2014, ten
  months before the test period.
- **This does not argue for putting the seal anywhere else.** 0041 rejected a
  middle holdout because sealed rows closer in time to training would be the
  optimistic ones, and the gap would measure the wrong sign of the wrong thing.
  That still holds. The cost is real and it is the price of the number being
  meaningful.

## What to do about it, and what was rejected

**Nothing automatic.** The temptation is a rule — shrink `holdout_fraction` when
`split_on` is set, or refit the shipped model on work+val before packaging. Both
were considered and neither is taken:

- **Shrinking the sealed slice on ordered data** makes the honesty number noisier
  exactly where 0042 showed it was most needed: v7's seal caught a 0.011 error in
  model choice that 50,175 validation rows could not.
- **Refitting the winner on work+val before packaging** is the obvious fix for the
  handicap and it is a leak of a subtle kind: the sealed score would then describe
  a *different model* from the one that ships, and the gap between `val_score` and
  the sealed score — the only thing 0023 bought — would stop being a statement
  about the shipped artifact. If it is ever done, the sealed score has to be
  reported as belonging to the pre-refit model and the bundle marked as not the
  thing that was measured. That is a decision with its own record, not a patch.

What is right for now is that **the driver states the horizon** it is handing the
model, so a reading is never taken against a run with a different one.

## What it costs

**Good**

- Stops the next reader concluding that the harness got worse when what changed
  was the forecast distance.
- Names the asymmetry between the two split modes, which is otherwise invisible:
  `holdout_fraction=0.05` means two quite different things depending on whether
  `split_on` is set, and nothing in the parameter's name says so.

**Bad**

- It is a caveat, not a fix. Sealed ordered runs remain handicapped, and every
  external score from one has to be read with this record in hand.
- It makes cross-run comparison on ordered data harder rather than easier: v5, v6
  and v7 now differ in work-half *end date* as well as in harness version.
