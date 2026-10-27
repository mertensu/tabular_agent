# 0092 — What ships is refitted on every row the search could use

- **Date:** 2026-10-27
- **Status:** accepted and implemented (`refit.refit_lineage`,
  `tools._refit_for_shipping`, both finalize profiles). Tests in
  `tests/test_refit.py`.
- **Decided by:** Ulf
- **Related:** [0023](../../research/archive/decisions/0023-the-third-split-has-evidence-now.md)
  (the holdout test rows, and the "not refitted" rule this record limits),
  [0044](0044-what-ships-is-the-average-not-the-argmax.md) (the blend, and its
  test-score rows), [0080](../../research/archive/decisions/0080-a-blend-weight-a-member-cannot-inflate-decided-before-the-seal.md)
  (the blend weights), [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md),
  `research/capacity_ship_minimal/README.md` § Restaurant r1

Vocabulary: **training** rows are what every tool fits on (the code calls them
*work*), **validation** rows rank the tables (the minimal profile's report calls
them *evaluation*), **test** rows are the optional holdout scored once at
finalize (the code calls them *holdout*, the minimal report *audit*).

## The problem

A run splits its rows into training and validation before any tool runs, and
every table, encoder and model it builds is fitted on the training rows. That is
necessary while the search runs: the validation rows rank tables, so nothing
fitted may have seen them.

Until now the bundle shipped those same training-only fits. On a time-ordered
split the validation rows are the *latest* the run has, so they sit between the
end of training and the first row the bundle will serve. Every stateful step
ships stale: a history encoder's stored past ends one validation period early, a
target encoding has never seen the newest groups' levels, and the models have
never seen the most recent regime.

Restaurant r1 measured what that costs. On 39 held-out test days after the run's
data, the shipped bundle scored 0.611 rmsle. Refitting the *same* features and
blend, with the same parameters, on training + validation rows scored 0.525:
+0.086 [0.074, 0.099] (store-level bootstrap). That is most of the distance
between the run's own validation score (0.513) and what it shipped.

## The decision

**After the search has chosen, `finalize_run` refits the selected table's
features and the shipped model(s) on training + validation rows, and ships the
refit.** Both profiles.

- **Nothing is re-chosen.** The table, every step's parameters (columns, windows,
  smoothing, leads, filter code), the blend's members and its 0080 weights are
  all fixed on the training-only fits before the refit runs. Only *fitted state*
  is redone: encoder mappings, stored history, out-of-fold training values, model
  weights.
- **The refit reuses the tools' fit code**, step by step: target, frequency and
  ordinal encoding, group statistics, history, stream aggregates, future-covariate
  windows, sandbox transforms (replayed) and train-only filters (re-evaluated on
  the new rows). Folds are rebuilt on the new rows with the run's seed and fold
  count, so out-of-fold values mean what they meant. Fitted on the training rows
  alone, the refit reproduces each tool's table and lineage exactly; that is the
  test that it is the same code.
- **Ranking, selection and every reported validation score stay on the
  training-only fits.** The refit has no validation score: the validation rows
  are in its training set.
- **The test rows stay out of the refit and score what ships, once.** The holdout
  record keeps its training-only rows unchanged, so 0023's optimism series and
  0044's blend-vs-argmax comparison continue, and gains a `shipped_refit` entry
  beside them with its test score and `vs_work_fit` (the gain over the same
  configuration fitted on training rows only).
- **A refit that fails ships the training-only fit** and records why, in the
  bundle manifest (`fitted_on`), the finalize result and the holdout record. A run
  still finalizes.

## What 0023 and 0044 said, and why this does not contradict them

`_score_on_holdout`'s docstring says the models are "not refitted": refitting on
training + validation "would produce a better model and a meaningless gap". That
sentence is about *measurement*. The optimism gap compares a model's validation
score with its test score, and a model trained on the validation rows has no
honest validation score to compare. It was never a decision about what ships.

So the measurement is unchanged: the training-only fits are still scored on the
test rows, with their optimism. What changes is what the bundle holds. The refit
gets its own test score, which is the number to expect on new rows, and no
optimism.

## What was rejected

- **Refit inside the agent's tool surface, or let the agent choose whether to
  refit.** The agent would choose on the validation rows, which the refit
  consumes. Refitting is a harness step after the search, like the blend (0044).
- **Re-run the search's choices on the larger set** (re-select the table, refit
  blend weights on new folds). That is a new selection on rows nothing is left to
  check it against.
- **Replay via a second `RunSession`**, as `restaurant_diag.py` did. It works as
  a measurement, but it re-runs every tool's probe and journal writes, needs a
  dataset-specific driver, and produces a second run directory. The product
  needs the fit, not the tool.
- **Refit the test rows in too.** Then nothing would judge what ships.
- **Make it opt-in.** The refit takes no rows from anything, changes no reported
  validation score, and the training-only test score is still printed beside the
  refit's, so a run where the refit hurts says so in its own report.

## A defect found while implementing it

The full profile computed the 0080 weights *after* scoring the blend, and scored
it, on validation and on the test rows, with equal weights. So since 0080 the
blend's reported validation score and its `shipped` test-score row described an
equal-weight average while the bundle predicted with the weighted one. The
weights are now fixed first and every blend score uses them. Single-model runs
are unaffected; blended runs' blend rows are not comparable to earlier ones.

## What it costs

- **The shipped object has no score from the rows that ranked it.** Its test
  score is the only direct measurement of it, and a run without holdout test rows
  (`ForecastingAgent`'s default) ships a model nothing in the run has scored.
- **Finalize does more work:** one more pass over each feature step on
  1/(1 − val_fraction) as many rows, plus one final fit per shipped model,
  without folds.
- **`Run.training_features()` now returns the refit table** (training +
  validation rows), because that is what the shipped models were fitted on. It
  lives in `refit/`, outside `tables/`, so no tool can reach it.
- **On a random split the gain is a larger training set only**, not recency; it
  should be small. The case for the refit is time-ordered data.

## Registered

1. On the next `ForecastingAgent` run with holdout test rows, `vs_work_fit` is
   positive and outside its noise, as Restaurant r1's offline replay was.
2. On random-split runs with holdout test rows, `vs_work_fit` is small (|Δ| below
   the training-only fit's test-score standard error) and not systematically
   negative.
