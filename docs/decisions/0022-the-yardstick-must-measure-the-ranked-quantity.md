# 0022 — The yardstick must measure the quantity being ranked

- **Date:** 2026-08-10
- **Status:** accepted, **not yet implemented**
- **Related:** [0019](0019-the-leaderboard-is-a-public-leaderboard.md) (which moved
  ranking onto `val_score` and left this behind),
  [0014](0014-two-level-stop.md) (the rule that uses the yardstick),
  [0021](0021-the-agent-picks-the-target-scale.md) (the feature whose measurement
  exposed it), [0023](0023-the-third-split-has-evidence-now.md)

## The problem, found by accident

Decision 0021 added target transforms. On house-prices the agent measured a gain
of **0.00808 rmsle** for `log1p` and the run's own stop rule agreed it was real:
the bar is `min_gain_ratio (0.25) x fold_spread (~0.0098) = ~0.0025`, and 0.008
clears it three times over.

Kaggle then scored both submissions:

| | run 2 (no transform) | run 3 (log1p) |
|---|---|---|
| experiments | 17 | 24 |
| our `val_score` | 0.12162 | 0.11354 |
| Kaggle | 0.12368 | 0.12298 |
| optimism | +0.00206 | +0.00944 |

The transform was worth **0.00070**, not 0.00808. **91% of the measured gain was
optimism**, and the optimism grew 4.6x while the experiment count grew 1.4x.

Bootstrapping `val_score` over its own 292 validation rows says why:

    run 2:  rmsle 0.12162   sd 0.01078   95% CI [0.10187, 0.14315]
    run 3:  rmsle 0.11354   sd 0.00973   95% CI [0.09488, 0.13206]

**The validation estimate's own standard error is ~0.010.** The gain the run
celebrated was 0.008 — smaller than one standard error of the instrument that
measured it. The two runs' confidence intervals overlap almost entirely. The
validation half never had the resolution to separate them, and reported a
difference anyway, because a maximum over 24 noisy draws always reports one.

## The defect, stated precisely

**0019 moved ranking onto `val_score` and left the yardstick on `fold_spread`.**

`fold_spread` is the standard deviation of `cv_score` across folds. It is a
perfectly good noise estimate — *for `cv_score`*. Since 0019, nothing is ranked
by `cv_score`. Every decision the run makes — `best`, `vs_baseline`,
`vs_source_table`, the stop rule, `table_patience` — is made on `val_score`, and
is gated by the wobble of a quantity nobody is ranking.

Numerically on this run the two happen to be close (`fold_spread` ~0.0098, val
standard error ~0.010), which is exactly why it hid. But the *bar* is a quarter
of the yardstick, so:

    bar for "this is an improvement":   0.25 x 0.0098  =  0.0025
    actual noise in the compared number:               =  0.0098

**The bar is four times smaller than the noise in the quantity it gates.** Every
sub-noise validation gain passes, and the run records it as progress. That is not
a tuning problem; the two numbers are estimates of different things and their
ratio is a coincidence of this dataset.

## The decision

**Compute the validation score's own uncertainty, and use that as the yardstick
for validation comparisons.**

Bootstrap `val_score` over the validation rows at training time, store `val_se`
on the leaderboard row, and have `stop.evaluate` compare gains against `val_se`
rather than `fold_spread`. Cost is a few hundred metric evaluations on a few
hundred rows — milliseconds, against the seconds a fit already takes.

`fold_spread` stays on the row. It is still the honest answer to "how much did
this model wobble across folds", and the `cv`-vs-`val` gap remains the run's
record of a gain that lived only in the folds. It simply stops being the ruler
held against a number it does not measure.

Had this existed, run 3 would have been told at e4 that its 0.008 gain sat inside
the noise — which is the correct thing to know *while the run is still going*,
and is not what any later split can tell it.

## What was rejected

- **Tuning `min_gain_ratio` upward.** Fixes the arithmetic on this dataset and
  leaves the category error in place. The next dataset where `fold_spread` and
  the validation standard error diverge breaks it again, silently.
- **An absolute threshold.** Already rejected in 0014 for good reasons — needs
  tuning per metric and per dataset, and fails quietly.
- **Growing the validation half.** The standard error falls only as `1/sqrt(n)`,
  so halving it costs 4x the rows — rows taken directly from the work half the
  agent builds features on. It also does nothing about *reporting* a noisy
  maximum, which is 0023's problem.
- **Doing nothing because the effect looks small.** It is 91% of a headline gain,
  and it grew steeply with experiment count. At 150 experiments it is the whole
  number.

## What it costs

**Good**

- The run gains the ability to distinguish a real gain from a lucky one *during*
  the run. Nothing else in the design can do that.
- Cheap, local, and it makes the stop rule's promise true rather than
  approximately true.

**Bad**

- **Not implemented yet.** Everything above is measurement and reasoning; the
  code still gates on `fold_spread`. Until it lands, every `val_score` comparison
  in this repo is against the wrong ruler, including the ones in the handoff.
- A bootstrap standard error assumes the validation rows are an i.i.d. sample of
  what you will serve. Under an ordered split (`split_on`) that is exactly what
  they are not, and `val_se` will understate the real uncertainty.
- More experiments get refused, some of them wrongly. That is the intended
  direction, and it is a real cost: 0014 loosened the stop rule deliberately
  because exploration is cheap, and this tightens part of it back.

## Where it will live

- `src/tabular_agent/harness/tools.py` — `_score_on_val` returns `val_se`.
- `src/tabular_agent/harness/stop.py` — `evaluate` and `table_status` compare
  against `val_se`.
- The leaderboard row, and the report.
