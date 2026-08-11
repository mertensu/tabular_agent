# 0028 — `val_se` is implemented; the bar it is a quarter of is still open

- **Date:** 2026-08-11
- **Status:** accepted, implemented (the yardstick); **one question deliberately left open** (the bar)
- **Supersedes:** one claim in [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md) —
  that having `val_se` "would have told run 3 at e4 that its 0.008 gain sat inside
  the noise". It would not have. 0022's **diagnosis and its decision both stand**;
  only its statement about what the fix achieves is wrong.
- **Related:** [0014](0014-two-level-stop.md) (whose `min_gain_ratio` is the open
  question), [0019](0019-the-leaderboard-is-a-public-leaderboard.md),
  [0024](0024-the-episode-stop-is-structural.md) (the advisory-vs-structural
  pattern this record has to weigh)

## What was built

0022's decision, as specified. `_score_on_val` bootstraps the validation score
over its own rows (200 resamples, seeded from the run's seed so the run stays
deterministic — this number gates the stop rule) and returns `val_se` beside the
score. It lands on the leaderboard row, in `meta.json`, in the training payload,
in the `lb` frame `explore_journal` hands over, and beside the headline number in
the report. `stop._noise` prefers it over `fold_spread`, falling back only when it
is absent — an old row, or a validation half too small to resample.

Measured on house-prices, against 0022's independent bootstrap of ~0.0098:

| experiment | `val_score` | `val_se` | `fold_spread` | ratio |
|---|---|---|---|---|
| baseline | 0.36869 | 0.01693 | 0.01509 | 1.12x |
| lgbm | 0.13758 | 0.01223 | 0.01008 | 1.21x |
| lgbm | 0.14346 | 0.01192 | 0.00854 | 1.40x |

The implementation agrees with 0022's number. The category error is fixed: the
yardstick now measures the quantity being ranked.

## The correction

0022 says: *"Had this existed, run 3 would have been told at e4 that its 0.008
gain sat inside the noise."*

It would not have, because **the enforced bar is a quarter of the yardstick**:

    old:  0.25 x fold_spread (0.01008) = 0.00252  ->  gain 0.00808 PASSES
    new:  0.25 x val_se      (0.01223) = 0.00306  ->  gain 0.00808 PASSES

To reject that gain the ratio would have to exceed **0.66**, not 0.25.

0022 conflated two statements that are both about the same numbers:

1. *the gain is smaller than one standard error* — **true**, and the correct
   diagnosis of what went wrong;
2. *the stop rule would therefore have rejected it* — **false**, because the rule
   compares against a quarter of a standard error, not a whole one.

Swapping the yardstick was still right, and on a dataset where `val_se` and
`fold_spread` diverge sharply it will matter a great deal. On *this* dataset it
moves the bar from 0.0025 to 0.0031, which changes nothing about the run that
motivated the record.

**0022's own opening sentence is the accurate statement of the defect** — "the bar
is four times smaller than the noise in the quantity it gates" — and it identifies
a problem the prescribed fix does not solve. The factor of four is
`min_gain_ratio`, and it survived the fix untouched.

## What was decided, and what was not

**Decided: the advisory.** `stop._latest_gain_inside_one_se` reports when an
experiment beat the incumbent by less than one standard error, and `advisory`
appends a sentence to the `note` the agent already reads:

> Note that this beat the previous best by 0.0081, which is inside one standard
> error of the validation score (0.0122) — a gain that size is as likely to be the
> split as the model.

It refuses nothing, gates nothing, and stops nothing. It fires only in the band
between "no gain" and "one standard error", so it is silent on most rows.

**Not decided: `min_gain_ratio`.** Left at 0.25 pending a second dataset, which
Ulf is supplying. 0022 rejected tuning it on the grounds that doing so "leaves the
category error in place" — that objection has expired, since the category error is
now fixed, and the arithmetic question is live and separate.

Two things make it a genuinely open call rather than an oversight:

- **What a stricter bar actually costs is smaller than 0022 implied.**
  `min_gain_ratio` does not refuse experiments; it decides whether one counts as
  *progress* for the patience counters. A higher bar makes a run **stop sooner and
  retire tables sooner** — it does not prevent the agent from trying anything.
- **The one dataset with ground truth argues for strictness.** Run 2 → run 3
  gained 0.008 on validation and **0.0007** on Kaggle. A bar at one standard error
  would have ended that run earlier and lost almost nothing real.

Against which: n=1, and 0014 loosened this rule deliberately because trying eight
things that fail in order to find the ninth is the only kind of search there is.

## The tension this record has to name

**Advisory guidance has failed three times in this project**, and each failure is
recorded: the episode stop the agent may own and will not exercise (0024), the
"work in short stretches" steer that produced 25 experiments in one episode (v4),
and the brief, which handed over information for free and changed nothing (0027).
The pattern is under-compliance with advice until something structural intervenes.

So shipping this as advisory is, on the project's own evidence, the option least
likely to work. It is chosen anyway, for a reason that only holds because of what
it costs: **the advisory is free, and it doubles as the experiment.** If the agent
is told a gain is inside the noise and grinds on regardless, that is the fourth
data point and the structural version gets made on evidence. If it backs off, the
stricter bar is never needed. Either way the second dataset arrives into a run
that has already been instrumented to answer the question.

The distinction that makes this not simply a repeat of the earlier failures: those
were advice about **what to do**. This is a **fact the agent cannot compute** — it
has no bootstrap and no access to the validation rows. 0027 is the reason to hold
that distinction loosely rather than confidently.

## What it costs

**Good**

- The stop rule's yardstick now measures the quantity it gates. That was a
  category error and it is gone.
- Every validation score in the repo ships with its own uncertainty — payload,
  leaderboard, journal frame, report.
- The open question is written down with the arithmetic behind it, instead of
  being rediscovered the next time a run celebrates noise.

**Bad**

- **The motivating case still passes the bar.** This record fixes the ruler and
  leaves the threshold, so a reader who takes "0022 is implemented" to mean "the
  run-3 failure cannot recur" would be wrong.
- A bootstrap standard error assumes the validation rows are an i.i.d. sample of
  what will be served. Under an ordered split (`split_on`) they are exactly not,
  and `val_se` will understate the real uncertainty — inherited from 0022,
  unaddressed.
- **A 14th payload field.** 0018's budget guard forced it to be declared, which is
  the guard working, but the payload is one field fatter on every training call.
- `N_BOOTSTRAP = 200` is a judgement, not a measurement. Fine for a second moment;
  it would be too few for a tail quantile, if anyone ever wants a confidence
  interval instead.

## Where it lives

- `harness/tools.py` — `_bootstrap_se`, `N_BOOTSTRAP`, `_score_on_val`.
- `harness/stop.py` — `_noise`, `_latest_gain_inside_one_se`, `advisory`.
- `report.py` — the headline score, plus or minus.
- `tests/test_stop.py` — the 0022 case, the advisory band, the fallback.
