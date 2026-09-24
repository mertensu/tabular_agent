# 0050 — The anchor is what matters, not the history

- **Date:** 2026-08-14
- **Status:** a measurement, from Rossmann v8 — the first run with `fit_history`.
  **Revises the scope of [0049](0049-the-last-observation-is-a-sample-of-size-one.md)
  without contradicting its measurements**, and adds a rule 0047 did not have. No
  code changed; one reporting script was corrected.
- **Related:** [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md)
  (whose `forecast_horizon` decision directly enabled this run's best table, and
  whose age-measurement decision is what made the failure diagnosable),
  [0048](0048-a-target-lag-is-not-target-encoding.md) (whose serving-symmetry
  claim turns out to hold *exactly* for offsets and only approximately for
  `_last`), [0049](0049-the-last-observation-is-a-sample-of-size-one.md),
  [0044](../../../docs/decisions/0044-what-ships-is-the-average-not-the-argmax.md) (the blend beat the
  argmax on the seal again), [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md)
  (the drift/selection confound this run also cannot separate),
  `scripts/run_rossmann_v8_history.py`.

## The result in one table

25 tables, 10.8 minutes, one conversation. The agent called `fit_history` five
times. **The four best tables in the run are history tables. So is the worst.**

| table | offsets | probe | vs parent |
|---|---|---|---|
| t24 | `[364, 728]` | **0.13669** | +0.021707 |
| t25 | `[182, 364, 728]` | 0.13884 | +0.019556 |
| t22 | `[364]` | 0.13915 | +0.019249 |
| t23 | `[7, 30, 364]` | 0.14227 | +0.016127 |
| t20 — best non-history | | 0.15840 | |
| **t7** | **`[]` — last-known only** | **0.46396** | **−0.263936** |

Same tool, same key (`Store`), same value (`Sales`), same ordering column. The
entire spread between the best table in the run and the worst is **one
parameter**.

## Why: the anchor, not the history

`_last` and a lag are not two settings of one feature. They differ in **what the
value is anchored to**, and everything follows from that.

| | `_last` | `lag364` |
|---|---|---|
| anchored to | the **freeze boundary** — one date, shared by every served row | **the row itself** — a different date per row |
| median, train vs val | 6289 vs **9697** | 6217 vs **6332** |
| age, train vs val | 73 vs 24 | **364 vs 364** |
| day-of-week | whatever the boundary happens to be | **preserved** (364 = 52 weeks) |

Because `_last` is anchored to the boundary, **every served row inherits whatever
was true on that one date**, and one date is never typical. 0049 caught this as
zeros: the boundary was a public holiday and 98% of stores were shut. v8 shows
the more general and more alarming version. The agent had already filtered
`Open == 1`, so **there were no zeros at all** — and `_last` was still ruinous,
because the last *open* day before a holiday is a pre-holiday shopping spike:

> median sales on 2015-04-30: **9,697**. Overall median: **6,337**. **+53%.**

The filter fixed the symptom and not the disease. A column that looks perfectly
healthy on inspection — no zeros, no NaNs, sensible range — destroyed a model
that was working, taking its parent from 0.200 to 0.464.

**Generally: a boundary-anchored feature has a sample size of one *per serving
batch*, not per row.** Every row shares the draw. A row-anchored feature draws
independently for each row, which is why it survives a bad date instead of being
defined by one.

This is why 0048's serving symmetry holds *exactly* for offsets — age 364 in
training and 364 at serving — and only in shape for `_last`, whose age ran 73 in
training against 24 at serving. The scale mismatch 0049 called structural is
structural **only for the boundary-anchored case**.

## The rule 0047 was missing: the floor makes it legal, the ceiling makes it useful

`forecast_horizon` is a *minimum*, and 0047 is right that it is: an offset below
the floor cannot be answered for **any** served row. But that is a statement about
legality, and this run shows it is not the statement a caller needs.

The validation block spans **44 days**. Asked for three offsets on the same table:

| offset asked | share of validation rows that actually got it | median age delivered |
|---|---|---|
| `lag7` | **11.2%** | 24 |
| `lag30` | 51.5% | 30 |
| `lag364` | **80.5%** | 364 |

`lag7` is answerable only for rows in the first week of the block. Everything
after that anchors past the freeze point and **degrades into `_last`**, inheriting
its pathology — which is exactly why t23 (`[7, 30, 364]`) is the *worst* of the
four good history tables. Adding short offsets adds boundary-anchored columns
wearing row-anchored names.

So:

- **The declared floor decides whether an offset is legal.** (0047, unchanged.)
- **The span of the serving block decides whether it is useful.** An offset
  shorter than that is honoured for early rows and silently degrades for the
  rest. **Nothing in the harness currently says this**, and it should — the tool
  knows the fold blocks and could report the share of rows an offset was actually
  answerable for, which is the number a caller would act on.

## The design decision that paid, and the one that nearly did not happen

**`_age` measured to the observation actually read** — not to the offset requested
— is the only reason any of this was diagnosable. A column named `lag7` reported a
median age of 24 and told the truth about having failed. Had age been recorded as
the offset asked for, `lag7` would have read 7 everywhere and been silently wrong,
which is this project's characteristic failure in its purest form.

**`forecast_horizon` was nearly left undeclared.** Without it, `fit_history`
refuses offsets outright and every table above is impossible — the run's best
result would not exist. 0029's placement (the task's owner states it) is what made
the capability reachable at all.

## Against the registered predictions

Registered in the script's docstring before the run, so they cannot be revised
afterwards.

1. **Mechanical — held.** Callable, produced pairs, accepted offsets.
2. **"Will the agent reach for it?" — yes**, five times, and it ran a clean
   ablation of its own: last-only, then `[364]`, then `[7,30,364]`, then
   `[364,728]`, then `[182,364,728]`. The vocabulary did not go unnoticed.
3. **"History tables score worse" — WRONG**, and wrong because the prediction
   treated one tool as one capability. It held for `_last` (−0.264, worse than the
   −0.13 predicted) and failed completely for offsets (+0.022). The lesson is
   about the prediction, not the run: **`_last` and offsets should never have been
   registered as a single expectation.**
4. **"The run will not ship a history feature" — WRONG, for a good reason.** The
   winner carries `Store_Sales_lag364`, `lag728` and their ages. The prediction's
   *reasoning* was that the probe would report these tables as worse; four of five
   were better, so shipping them was correct. The feedback loop did its job in
   both directions: it steered the run off t7 immediately and back onto offsets
   later.

**A caution about how this was nearly mis-reported.** `summarise_history` printed
"Registered 4 is WRONG and the feedback loop is decorative" — a hardcoded verdict
that *assumed its own premise* (that the probe had reported the tables as worse)
instead of reading the deltas beside it. The run's own numbers contradicted the
sentence printed above them. It is fixed to read the deltas. Worth recording
because it is 0031's shape in the reporting layer: a check that looks like it is
checking and is only asserting.

## The sealed slice

51,290 rows, scored once, after the winner was picked.

| model | val | sealed | optimism |
|---|---|---|---|
| lgbm (best experiment) | 0.13669 | 0.18595 | +0.04926 |
| xgb | 0.15544 | 0.17757 | +0.02213 |
| catboost | 0.14162 | 0.19670 | +0.05507 |
| elasticnet | 0.23873 | 0.25304 | +0.01431 |
| **blend — what ships (0044)** | 0.14138 | **0.17773** | +0.03635 |

**The blend beat the argmax on the seal again** (0.17773 against lgbm's 0.18595),
which is 0044 holding for the second dataset.

**v7's registered "optimism ≤ 0" fails here, and this run cannot say why.** The
sealed rows are a later period than validation, so the gap mixes selection
optimism with drift — 0041's confound, unresolved. One extra candidate specific to
this run: the sealed block sits 45–90 days past the freeze, so `_last` is staler
there than anywhere in training, and the tree models that lean on it hardest
(catboost +0.055, lgbm +0.049) show more optimism than the one that barely can
(elasticnet +0.014). **That is suggestive and is not evidence** — the ordering is
also what you would expect from model capacity alone.

## What this means for 0049 and for slice 3

**0049's measurements stand; its scope was too broad.** It says "history" where it
measured `_last`, because `_last` was all it measured — offsets need a declared
horizon and the hand-driven probe declared one but only ever paired it with a
single 7-day offset, below the block span, which is now known to be the case that
degrades. Its central sentence is still right and now has a sharper form: *the
most recent observation is a sample of size one* → **a boundary-anchored
observation is a sample of size one shared by the whole batch.**

**Slice 3 (windows) is still worth building, and its argument changes.** 0049
justified it as a fix for `_last`. The better argument is the one that came from
outside: the competition's winner used *"sales for this store on a typical Monday,
averaged over the last 6 months / 1 year"*, and this run shows the agent reaching
for the same shape with the parts it had — 364 and 728 are one and two years,
chosen to keep the weekday aligned. **Offsets already deliver the recency and the
periodicity; a window would add the averaging**, which is the remaining variance
reduction. That is a smaller, better-specified gap than "windows are the fix".

**A cheaper change may be worth more than slice 3.** `_last` is emitted
unconditionally and cost this run its worst table. Whether it should be optional —
or reported alongside the share of rows each offset was answerable for — is a tool
design question this record does not settle, and it is now better evidenced than
anything slice 3 rests on.
