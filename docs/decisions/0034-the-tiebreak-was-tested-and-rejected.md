# 0034 — Breaking a `val_se` tie with `cv_score` was tested and rejected

- **Date:** 2026-08-13
- **Status:** accepted (the rejection is the decision)
- **Related:** [0028](0028-the-yardstick-is-fixed-the-bar-is-not.md) (which left
  `min_gain_ratio` deliberately open), [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md),
  [0023](0023-the-third-split-has-evidence-now.md) (which this record argues is
  the real conclusion)

## The problem

`finalize_run` packages `argmax(val_score)`. Nothing consults the margin. On
Porto Seguro's second run the margin between the winner and the runner-up was
**0.00020 of gini against a `val_se` of 0.00417 — one twentieth of one standard
error.** The harness recorded a coin flip as a verdict.

That would be a curiosity if the two models were interchangeable. They were not.
`scripts/rebuild_submission.py` was written to package experiments the harness
had declined, and both were scored on Kaggle:

| | our `val_score` | Kaggle private |
|---|---|---|
| e8 — the packaged winner | 0.28480 (rank 1) | **0.26522** |
| e11 — the runner-up | 0.28460 (rank 2) | **0.27724** |

**The model the harness discarded was better by 0.012 of gini**, which on that
competition is worth more than every feature the agent engineered after its
second tool call. Losing 0.012 to a 0.00020 measurement is not a tuning problem;
it is the selection rule reporting noise as a decision.

The obvious fix presents itself immediately, and it is wrong. This record exists
so that it stays rejected.

## The candidate rule

> When two experiments are within one `val_se`, break the tie with `cv_score`.

It has a good argument behind it. `cv_score` really is the better estimator of
the two the harness computes — measured across the runs we have Kaggle numbers
for:

| | error spread vs Kaggle | rank correlation with Kaggle |
|---|---|---|
| `val_score` | 0.0124 | **−0.4** |
| `cv_score` | **0.0043** | **+0.8** |

Five folds averaged beats one holdout, as theory says it should. `val_score`'s
*negative* rank correlation is the striking number: across these runs, ranking by
our headline metric was slightly worse than ranking at random.

## The measurement

Three runs where the alternative pick could be rebuilt and scored. In each, the
tie-broken choice is compared against what the harness actually shipped.

| run | cv-tiebreak picks | harness picked | outcome |
|---|---|---|---|
| seguro_v2 | e11 → 0.27724 | e8 → 0.26522 | **+0.0120 win** |
| seguro_v1 | e3 → 0.26884 | e4 → 0.27191 | **−0.0031 loss** |
| house_v8 | e14 → 0.12296 | e9 → 0.12300 | +0.00004, null |

**One win, one loss, one null.** The win is large and the loss is small, which is
exactly the shape that tempts you into shipping it on three data points.

## The decision

**Do not break ties with `cv_score`. Do not add a margin threshold.**

Two reasons, and the second is the one that generalises.

**A better estimator is not a better tiebreaker.** `cv_score`'s advantage is
lower variance around a model's true skill. At a tie, the quantity being
compared is *approximately zero* — the two experiments genuinely have almost the
same skill — so the comparison is decided by whatever noise each estimator has
left. Reducing an estimator's variance from 0.0124 to 0.0043 does not help when
the signal you are asking it to resolve is 0.00020. The rank correlation of +0.8
is a statement about ordering models that *differ*; it carries no promise about
ordering models that do not.

**And it would not have escaped the problem anyway.** `cv_score` is clean in
these three runs only because nothing selected on it. Promote it to tiebreaker
and it enters the selection rule; the agent's search then optimises against a
composite of both numbers, and `cv_score` starts accumulating the same
winner's-curse bias `val_score` already carries. The property that made it
trustworthy is destroyed by the act of trusting it.

Stated generally, and this is the sentence the record is for:

> **A number can be an estimate, or it can be an objective. Not both.** Any
> quantity a run selects on stops estimating generalisation, no matter how well
> it estimated before it was used.

## What was rejected, explicitly

**A margin threshold** (`prefer cv when the val margin is below k·val_se`). The
win/loss split invites fitting `k` to make seguro_v2 fire and seguro_v1 not. That
is fitting a constant to three points, on data that came from three runs of one
harness against two datasets. It is the failure 0022 records, one level up: a
change that improves the scores we can see and cannot be defended on its own
terms.

**Blending the two numbers** (rank on a weighted average of `val` and `cv`). Same
objection with an extra knob, and it makes both numbers objectives at once.

**The midpoint as a truth estimate.** On seguro_v1, `(cv + val)/2 = 0.27239`
against a private truth of 0.272 — nearly exact, and a coincidence. The two
numbers are biased in opposite directions for unrelated reasons: `val` upward by
selection, which grows with experiment count; `cv` downward because fold models
train on (K−1)/K of the rows while the shipped model trains on all of them, which
depends on dataset size and the steepness of the learning curve. Their difference
is not a constant, and averaging them assumes it is.

## What this actually establishes

The three-run test was designed to answer "which of our two numbers should
decide a tie". It answered a larger question instead: **both of them are
contaminated once a run selects on them, and no arrangement of the two recovers
what selection took.**

Measured optimism, `val_score` minus the truth, on every run with a Kaggle
number:

| run | experiments | optimism |
|---|---|---|
| house run 2 | 17 | +0.00206 |
| house run 3 | 24 | +0.00944 |
| house v8 | 14 | +0.0113 |
| seguro v1 | 5 | +0.0129 |
| seguro v2 | 11 | +0.0183 |

Two datasets, two task types, two metrics, one direction. **The real conclusion
of this experiment is [0023](0023-the-third-split-has-evidence-now.md)** — a
split the search never sees — and this record should be read as its second piece
of evidence rather than as a result about tiebreaking.

## What it costs

We ship a rule we know is wrong at the margin. Seguro v2 left 0.012 of gini on
the table and we are choosing not to fix it, because every available fix is
either unmeasurable from inside the run or fitted to three points.

That is deliberate, and it is bounded by an honest statement of what we do not
know: **a third split will tell us how much the selection costs, and only then
will a fix be judgeable.** Until then the diagnostic path is
`scripts/rebuild_submission.py` plus a Kaggle submission — expensive, external,
and available for two of our datasets.

## Footnote: how e11 came to exist

e11 — the best model any run of this harness has produced — was trained on a
table that a no-op `transform_features` (`df = df.copy()`, nothing else) had
produced, which reset `table_patience` and bought the run further experiments it
would otherwise not have had. The best model we have was a side effect of a
loophole in the stopping rule.

That is not an argument for keeping the loophole. It is an argument that our
stopping rule and our selection rule are both being decided by quantities
smaller than their own noise, and it is recorded here so that closing the
loophole is not mistaken for a pure improvement.
