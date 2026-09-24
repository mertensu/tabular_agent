# 0077 — Several disagreeing rulers, rather than one better one

- **Date:** 2026-08-18
- **Status:** accepted and implemented (`panel.py`, wired into `probe_table` and
  the stop rule). **Overturns 0073's refusal of a panel**, on a cost argument that
  was measured rather than assumed.
- **Answers:** Ulf, after this session's IEEE-CIS run: *"i would like you to fix
  the ruler using multiple disagreeing validation schemes"*
- **Related:**
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (the
  instrument this widens, and whose own docstring names the gap this fills),
  [0073](0073-the-ruler-is-mispriced-on-one-dataset-and-three-mechanisms-failed-to-say-why.md)
  (which refused a two-budget panel on cost — the refusal this record overturns),
  [0066](0066-the-collapsed-column-wins-everywhere-except-where-it-is-graded.md)
  and [0065](0065-the-agent-found-the-leak-and-the-ruler-overruled-it.md) (two of
  the four failures this exists for, and the limitation it does **not** escape),
  [0075](0075-every-declared-categorical-it-could-see-it-used-and-every-one-it-could-not-it-ignored.md)
  (18 of 20 comparisons unstable, 9 accepted anyway — the run that prompted this),
  [0035](../../../docs/decisions/0035-the-search-is-over-features.md) (the argmax, deliberately untouched),
  [0022](../../../docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md) and
  [0028](0028-the-yardstick-is-fixed-the-bar-is-not.md) (the yardstick this is
  ANDed with, not a replacement for),
  [0034](0034-the-tiebreak-was-tested-and-rejected.md) (why no threshold here is
  fitted to three observations),
  `fraud_detection/winner.md`.

## The problem, stated as a count

The ruler has been measured wrong four times — 0056, 0065, 0066, 0073 — and the
failures share a shape. It was not undecided. It was **confident and pointing the
wrong way**: 0066's leaky table read `sign_stable: True` at 2.9 block sd with every
block agreeing, and lost privately.

0059's interval was the first response, and its own docstring names what it cannot
see:

> *"It samples the periods it has, and a run has one validation half. If an effect
> changes sign beyond the far edge of that half [...] every block can agree and the
> effect can still fail on the rows that matter."*

This session made that concrete. On IEEE-CIS, **18 of 20 comparisons came back
`sign_stable: False` and nine of those were accepted as improvements anyway**,
because nothing is gated on the interval. An instrument that reports and is not
consulted is not an instrument.

## What was chosen, and why it is not "a better ruler"

The instinct after four failures is to build one ruler that is finally right. The
IEEE-CIS winners, on this exact data, did the opposite and said so:

> *"We never trusted a single validation strategy so we used lots of validation
> strategies. Train on first 4 months of train, skip a month, predict last month.
> We also did train 2, skip 2, predict 2. We did train 1 skip 4 predict 1."*

Those four geometries are now `panel.ordered_schemes`, transcribed. Blocking
(0059) re-measures **the same fit** on different evaluation rows. A panel refits on
**different training periods, at different distances, across different gaps** — so
what varies is the forecasting problem itself, which is what the run is graded on.

| scheme | train | gap | evaluate |
|---|---|---|---|
| `train4_gap1_eval1` | blocks 1-4 | block 5 | block 6 |
| `train2_gap2_eval2` | blocks 1-2 | blocks 3-4 | blocks 5-6 |
| `train1_gap4_eval1` | block 1 | blocks 2-5 | block 6 |
| `train5_gap0_eval1` | blocks 1-5 | — | block 6 |

**The gap is the part no other instrument here has.** A run is graded across a real
one (IEEE-CIS's is thirty days) and a ruler that trains up to the moment it
predicts answers an easier question. Measured on `_runs/ieee_v1/tables/t18.parquet`,
531,486 x 489, identical columns:

    train5_gap0_eval1   0.92129
    train4_gap1_eval1   0.90891
    train2_gap2_eval2   0.88911
    train1_gap4_eval1   0.83411

**A 0.087 spread from geometry alone** — four times what that run's entire feature
chain was worth. What a panel buys is not precision; every scheme is noisier than
the run's own validation score because each trains on a fraction of the rows. It
buys **disagreement**, and a gain that survives four geometries is a different
object from one that does not.

## Why 0073's refusal does not apply, and this is the load-bearing correction

0073 designed a two-budget panel and refused it: *"the second arm costs ~10x on
every dataset — fit time at 3000 rounds is dominated by tree count, not rows."*

That is correct **about the panel it priced**, and this is a different one. Raising
the budget multiplies the cost of every fit. Varying the *geometry* at the same
budget trains on **fewer rows each time**. Measured, not assumed:

| | |
|---|---|
| the current ruler, one fit on all 531,486 work rows | 8.4 s |
| four schemes, total | 20.3 s |
| **multiplier** | **2.4x** |

0073's cost argument was sound and its conclusion was scoped to budget panels. The
number that made it decisive does not survive changing what varies.

## The soft gate — Ulf's call, and what it deliberately does not do

Three shapes were put to him. He chose the middle one.

> **Agreement decides what counts as *progress*. It never decides what may win.**

`_since_improvement` now requires `gain > min_gain_ratio * noise` **and** the
panel's agreement. A step the panel rejects does not reset patience and is flagged
to the agent. `best_table` is untouched: it is still the argmax of `probe_score`,
because that is 0035's, and a gate there would change what ships on every dataset
on the strength of an instrument with no track record.

Two consequences, both intended:

* **A false negative costs patience, not a table.** The strictness below is
  therefore cheap to be wrong about, which is the only reason it is affordable.
* **It would not have saved 0066.** The leaky table would still have shipped. The
  hard gate would have caught it, and was refused because it can also refuse a real
  gain and record nothing about what was lost. That trade is revisitable once the
  panel has fired on a few runs; it is not decidable now.

## Agreement is all-or-nothing

With four schemes, a majority rule lets a step through on 3-1 — and the whole
reason this exists is that the project has repeatedly accepted steps most of its
instruments liked. So `agrees` is true only when **every** scheme that ran confirms
the headline's sign. Fewer than two schemes returns `None`, not a verdict: one
scheme is not a panel, and reporting it as one would be worse than silence.

Note what is *not* here: no threshold on magnitude, no p-value, no tuned constant.
0034 refused to fit a constant to three observations and that applies with full
force. The panel votes on **sign** only, because sign is what a decision consumes.

## Unordered runs get a different question, and are told so

House-prices and Porto Seguro have no ordering column, and a time geometry cut on
row order would be a fiction. The schemes instead vary **which rows were trained
on** — four subsamples, each scored on the rows it did not see — so the question
becomes "does this gain survive a different fit". Weaker, and honest: the record in
the journal is tagged `kind: "subsample"` rather than `"horizon"`, so nobody reading
it later is in doubt about which question was answered.

## The limitation this does not escape

**The table's encoders were fitted on the whole work half before the panel sees
it.** A scheme evaluating on block 6 scores rows whose features were built with
knowledge of block 6's targets — out-of-fold per row (0032), but in-sample for the
block. Every scheme shares this, and so does the parent it is compared against, so
it does not obviously bias the *difference*.

But "every window shares the defect" is exactly how 0066's leaky column passed
three instruments in a row. **A panel is a wider net, not a proof**, and it is
written down here so that the first time it is confidently wrong, nobody is
surprised.

## Two bugs the tests found, both silent

* **Interpolated quantiles invent boundaries.** 200 rows each of 1, 2 and 3
  produced edges at 1.997 *and* 2.0, with a block between them holding nothing. A
  near-empty block is worse than a missing one: it is scored, it is wild, and it
  votes. Boundaries now use `method="lower"`, so every edge is a value that occurs.
* **An edge equal to the maximum opens a block nothing falls into.** The coarse-
  column fallback counted blocks from `len(edges)` and produced a scheme evaluating
  on **zero rows** while looking valid. Blocks are now counted from the ones that
  actually hold rows.

Both were found by tests written from the docstrings rather than from the code, and
neither would have raised anything in a run.

## Cost, stated plainly

2.4x on the probe path, per table. The test suite went from ~5 to ~12 minutes for
the same reason, which is the honest signal that this is not free. Per-table scheme
scores are cached on the journal row so a parent is never refit for its children,
and they are kept out of the agent's payload (0018) because `probe_panel` already
carries the same fact as signed gains.
