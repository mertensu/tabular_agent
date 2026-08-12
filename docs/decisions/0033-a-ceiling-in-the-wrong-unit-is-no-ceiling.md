# 0033 — A ceiling in the wrong unit is no ceiling; the search is bounded, the fit is not

- **Date:** 2026-08-12
- **Status:** accepted, implemented
- **Related:** the `elasticnet` model type and its "is the signal in the features
  or the model" argument (recorded in `models.py`, no record of its own);
  [0029](0029-the-harness-owns-the-exam.md) (modelling decisions are the agent's,
  which is why the fix bounds cost rather than choices);
  [0021](0021-the-agent-picks-the-target-scale.md) (a menu entry should be worth
  an experiment — an entry nobody can afford to run is not)
- **Does not supersede, but contradicts a premise of:** the reasoning in
  `train_cv`'s docstring that checking wall clock *between folds* is
  sufficient because "the case this exists for is a fit whose *every* fold is
  slow, so the first boundary catches it". That premise is false and the guard it
  justifies is still in place — see **What is left open**.

## What happened

The first classification run in this project (Porto Seguro, 595,212 rows, 3.6%
positive) reached experiment 6 and stopped producing anything. The process sat at
**100% of exactly one core for 63 minutes** with no journal write. A stack sample
found it inside `sklearn.linear_model._sag_fast.sag64`: the saga solver, which
is what `LogisticRegressionCV` uses and the only solver implementing the
elasticnet penalty.

The agent had done nothing wrong. `train_tabular_model`'s own description tells it
elasticnet is *"worth one experiment on any table you care about"*, and it took
that advice at the first opportunity after the GBDTs.

Measured on that table afterwards:

| | |
|---|---|
| one saga epoch, 476,170 x 52 rows | **0.23 s** |
| `max_iter` in force | **20,000** |
| therefore one fit | **~77 minutes** |
| fits in one `LogisticRegressionCV` (8 mixes x 10 strengths x 5 inner folds) | **400** |
| therefore one experiment | **~510 hours** |

The run was killed. It had 5 experiments of a 25-experiment budget and its best
model, `e4`, was packaged by hand; it scored 0.272 private on Kaggle.

## The two defects, which are independent

**1. One constant, two units of work.**

```python
_LINEAR_MAX_ITER = 20000   # served BOTH branches
```

Its comment — *"Coordinate descent needs headroom on the weakly-regularised end
of the path"* — is correct, and correct **only for `ElasticNetCV`**, where an
iteration is a sweep over the coordinates and is cheap. For saga an iteration is
a full **epoch over every row**. sklearn's own defaults say the same thing in
numbers: 1000 for the former, 100 for the latter, a 10x gap that had been
collapsed into one number and then multiplied by 20.

This is the part worth generalising. The value did not look reckless; it looked
*justified*, because a correct rationale sat directly above it. The rationale had
a scope and the constant had none.

**2. The inner search inherited the size of the table.**

`*CV` estimators select a hyper-parameter and fit at it on whatever rows they are
handed. But those two jobs do not scale alike: **choosing** a penalty ranks a
handful of candidates and stabilises early; **fitting** at the chosen penalty
wants every row. Joined together, the choosing costs `n_grid x n_rows` and nothing
bounds it. The grid was sized on house-prices' 1,460 rows.

## The decision

**Split the ceiling by solver, and split the search from the fit.**

```python
_COORDINATE_DESCENT_MAX_ITER = 20000   # a sweep over coordinates
_SAGA_MAX_ITER = 200                   # an epoch over every row
PENALTY_SEARCH_MAX_ROWS = 20_000
```

`_PenaltySearchThenRefit` wraps both branches: it draws at most
`PENALTY_SEARCH_MAX_ROWS` rows (stratified when classifying), runs the `*CV`
estimator on that sample to select the penalty, then refits a plain estimator at
the selected penalty on **every** row it was given. The sample is drawn inside
`train_cv`'s fold from that fold's training rows only, so it crosses no fold
boundary, and it is seeded from the run's seed so a leaderboard stays
reproducible.

The classification grid also shrinks — 3 mixes x 5 strengths x 3 inner folds = 45
fits, where it was 400 — on the same solver-shaped argument as the ceiling:
coordinate descent warm-starts along the alpha path, so a dense grid is nearly
free; saga refits from scratch at every point, so the grid multiplies cost
directly. Three points still find which *end* of the ridge-lasso family the data
wants, and the resolution given up was already measured as noise (house-prices:
the two endpoints differed by 18% of one standard error).

### The evidence that the cap does not cost the answer

The purpose of this model type is a **verdict**, not a fitted model. Measured on
the 476k-row table, a linear model scored:

| rows | time | gini |
|---|---|---|
| 25,000 | 0.8 s | 0.23669 |
| 100,000 | 3.9 s | 0.24308 |
| 376,170 | 18.1 s | 0.25011 |

Same verdict against the GBDT's 0.28480, in a fortieth of the time. The
coefficients keep improving with rows — which is why the **refit is not capped**,
only the search.

## What was rejected

**Lowering `FIT_TIME_BUDGET_S`.** It would not have fired. The bound is checked
*between folds*, and here a single fold exceeded the entire budget, so the check
was never reached. When fold 1 eventually returned, the check would have fired and
**discarded 510 hours of work** — the guard's failure mode here is not laxity, it
is doing its job far too late to matter.

**Capping the rows the model is fitted on.** Cheaper still, and it would corrupt
the thing the model exists for: an under-fitted linear model makes the GBDT look
better, which biases the diagnostic in the one direction it must never be biased.

**Dropping `elasticnet` from the menu at large `n`.** A diagnostic that
disappears exactly when a run is expensive enough to need it is worse than none,
and it would make the menu depend on table size in a way the agent cannot see.

**Selecting the penalty once outside the folds.** Much cheaper — one search
instead of six. It also selects a hyper-parameter using rows that later fold
scores are measured on, which is the batch dependence this project puts
preprocessing inside the estimator to avoid.

## What it costs

**Good**

- One experiment on the 476k-row table: **510 hours -> 4.6 minutes**, and it now
  returns an answer (elasticnet gini 0.25111 vs lgbm 0.28480, so the trees win by
  roughly 4 `val_se` and the signal there is not purely linear).
- Cost of the search stops growing with the table. It is now roughly the cost of
  the refits, which is bounded by the same thing every other model type is.
- **Below the cap nothing changes at all.** No sampling occurs, and the search
  estimator — which refits at its own choice anyway — is used as it stands.
  Verified against the pre-change code on house-prices v8's winning table:
  identical to 1e-12 (0.11356705 both).

**Bad**

- **One extra fit per fold** whenever sampling happens. Small against a search it
  makes finite, and zero below the cap.
- **`PENALTY_SEARCH_MAX_ROWS = 20_000` is a judgement**, defensible from the table
  above but not derived. A dataset where the penalty genuinely needs more rows to
  rank would be selected on too little; nothing detects that.
- **The classification grid is coarser than the regression one.** Two model types'
  worth of tuning behaviour under one menu name, differing by task. Defensible by
  the solver argument, invisible from the outside.
- **`params` can still restore the pathology.** An agent passing
  `{"max_iter": 20000, "Cs": 10}` gets exactly what it asks for, by design
  (0029: the modelling is the agent's) — so the cost floor here is a default, not
  a guarantee.
- Sampling is stratified on the target for classification, which means the search
  sample's class balance is *by construction* the population's. Right for the
  imbalanced case that motivated it; it is still an assumption.

## What is left open

**The wall-clock bound still cannot bound a single fold**, and that is general —
it has nothing to do with elasticnet. Any estimator whose first fold outruns the
budget is invisible to `train_cv`'s between-folds check, and this run is the
existence proof that "every fold is slow, so the first boundary catches it" is not
the only shape the problem takes. The honest fix is a hard timeout, which needs a
subprocess — explicitly rejected in `train_cv`'s docstring on the grounds that it
"would buy little". That cost/benefit was measured on runs where it had never
mattered; it has now cost a 25-experiment run. **Revisiting it deserves its own
record and its own measurement, and neither exists yet.**

## Where it lives

- `models.py` — `_COORDINATE_DESCENT_MAX_ITER`, `_SAGA_MAX_ITER`,
  `_SAGA_L1_RATIOS`, `PENALTY_SEARCH_MAX_ROWS`, `_PenaltySearchThenRefit`,
  `_make_elasticnet`.
- `tests/test_models.py` — `test_the_penalty_search_is_bounded_but_the_fit_is_not`,
  `test_the_penalty_search_sample_is_stratified_for_classification`,
  `test_the_saga_iteration_ceiling_is_in_saga_s_own_unit`.
