# 0035 — The search is over features; models are enumerated, not searched

- **Date:** 2026-08-13
- **Status:** accepted, **not yet implemented**
- **Decided by:** Ulf ("only searching for good features and no models at all is
  also appealing … expanding the search for models is probably not the best
  thing")
- **Supersedes:** [0021](0021-the-agent-picks-the-target-scale.md) in one respect
  only — *where* the target scale is chosen. Its menu-not-code decision stands.
- **Related:** [0014](0014-two-level-stop.md) (the two-level stop this collapses),
  [0017](0017-credit-by-ablation.md), [0034](0034-the-tiebreak-was-tested-and-rejected.md),
  [0023](0023-the-third-split-has-evidence-now.md) (the instrument that can
  judge this change, still unbuilt)

## The problem

A run's leaderboard ranks **experiments**: one model, with one set of
parameters, on one table. Two very different acts produce identical-looking rows.

    e9   catboost   t10   0.11171     <- a different model
    e14  lgbm       t16   0.11255     <- a different table

Asked whether t16 is a better table than t10, the leaderboard cannot answer.
Something else moved. It ranks them anyway, and `finalize_run` packages the top
row.

This is [0017](0017-credit-by-ablation.md)'s rule — *a comparison must differ by
exactly one thing* — applied to attribution but never to selection.

## The measurement

Across all 16 runs in `_runs/`, every experiment classified by whether it
introduced a new table or re-modelled a table already trained on:

| | |
|---|---|
| experiments, all runs | **222** |
| varied only the model or its parameters | **97 (44%)** |
| runs whose winner was a model variation | **10 of 16** |

**Forty-four percent of the search budget went to choosing among an enumerable
set**, and the run's winner came out of that set two times in three.

Those are also the *least reliable* comparisons available. Two feature sets
scored by the same estimator on the same rows have highly correlated errors, so
the difference between them is measured far more precisely than either score:

    Var(A − B) = Var(A) + Var(B) − 2·Cov(A, B)

Two *different* estimators on the same rows make different mistakes. `Cov` falls,
and the variance of the difference rises toward the sum. A hyper-parameter tweak
and a feature change are not comparable measurements, and we have been ranking
them in one column.

The margins confirm there was nothing to resolve. Winner over runner-up, in each
run's own `val_se`:

| run | margin | in `val_se` |
|---|---|---|
| house_v8 | 0.00036 | **0.03** |
| seguro_v2 | 0.00020 | **0.05** |
| seguro_v1 | 0.00078 | **0.18** |

Three to eighteen hundredths of a standard error. 0034 records what one of those
coin flips cost on Kaggle: 0.012 of gini.

### What the agent already does well

Lineages across all runs were reconstructed into a forest to ask whether the
agent ever recombines abandoned directions. **It does — 24 times, in 10 of the 16
runs**, always in the same shape: continue from the current table and re-apply
one step first seen on a branch that had been dropped.

    house_v8    t8  = t7 + group_stats(Neighborhood)   first seen in t6
    house_v8    t13 = t12 + filter(GrLivArea <= 4000)  first seen in t11
    rossmann_v2 t6  = t5 + target_encode(Store)        first seen in t3

The feature search is doing real, structured work unprompted. It is the half of
the loop worth spending on.

### The external comparison

`wandb_fc/tabular-nomad` (rank 1/685 agent submissions on NOMAD) fixes the model
and searches only features: LightGBM for screening, *minimal tuning*, selection by
5-fold out-of-fold metrics only, ~60 feature iterations, a feature matrix that
grew to **677 columns** against our ~30. Their held-out labels are not on the
machine the agent runs on.

Their advantage over us is accumulation volume on one axis, not a better search
over two.

## The decision

**The agent's action space is tables. Models are enumerated at the end.**

The principle, which is what to remember when this record has aged:

> **Search where the space is unbounded. Enumerate where it is small.**

Model types are a set of four. You never need to *search* a set you can simply
train all of. Feature space is unbounded and compositional, and is the only place
a search loop earns the noise it mines.

Three parts.

**1. A probe model ranks tables.** One estimator, fixed parameters, fixed fold
assignment, fixed seed, applied unchanged to every table. It is a *measuring
instrument, not a candidate* — its job is to be a constant ruler so that a
difference between two tables is attributable to the tables. Because the folds
and the estimator never move, every table-vs-table comparison is paired, which is
the cheap kind. It may be deliberately small (few folds, capped trees): it only
has to rank, never to ship.

**2. The terminal phase is enumerated, not searched.** When the table search
stops, fit xgboost, catboost and elasticnet on the winning table and ship the
best by the probe's own folds. The agent has no part in it.

**3. The target scale is derived from the metric, overridable in `TaskSpec`.**
Not an agent choice, and not searched.

## Why the target scale moves, against 0021's argument

0021 put `target_transform` on `train_tabular_model` **specifically to keep it out
of `TaskSpec`**, on the grounds that a run setting would mean "*we* decide once,
at the top, that logging helps — an assumption nobody measures". That reasoning
was right, and the run that followed measured it: lgbm/`none` 0.1366 against
lgbm/`log1p` 0.1287 on the same table.

Two things have changed since.

**The measurement was made and it agreed with a derivation.** 0022 established
that the objective the model is fitted under must be the quantity the run ranks
on. Under `rmsle` — squared error on the log scale — fitting on `log1p` *is*
fitting the metric, and fitting raw is optimising something nobody is scoring.
For that metric the answer follows from the metric rather than from a search.

**Under this record there are no model experiments to attach it to.** It cannot
stay where 0021 put it, because the argument it was an argument to is gone.

`TaskSpec` is also the right home under the project's own rule: dataset-specific
knowledge lives with the data's owner, not in `src/`. A target that should be
logged under `mae` — where no derivation is available — is exactly that kind of
knowledge.

**0021's core decision is untouched**: a validated menu of stateless invertible
transforms selected by name, never free-form code over the target.

## What was rejected

**Keeping the mixed leaderboard and adding a margin rule.** 0034 rejects the
margin rule on its own evidence. It also treats the symptom: the reason winners
are decided at 0.03 `val_se` is that half the candidates differ by things too
small to resolve.

**Equal-weight blending as the terminal step.** It is the theoretically better
answer — picking the best of three is an argmax worth about `√(2 ln 3) ≈ 1.5`
standard errors of luck, and averaging mines none. Our measurements disagree:

| | |
|---|---|
| Rossmann, OOF weights over 4 models | 0.12991 vs **0.12741** best single — worse |
| Rossmann, weights from the last fold | 0.12792 — still worse |
| house-prices, 3 GBDT families | +0.00024, i.e. nothing |

The cause is in the residuals: **GBDT-vs-GBDT correlation 0.86–0.88.** Three
boosted-tree models on one feature matrix make one set of mistakes. And on
Rossmann the OOF weight optimiser put **0.8 on the lgbm that was worst on val**,
because it had the best `cv_score` — a weight fitter is itself a selection
procedure running on a contaminated number, so it reintroduces what the terminal
phase exists to avoid.

Rejected for now, but the terminal rule is written as one named function with the
runners-up recorded, so that *pick-best vs equal-weight* becomes a one-line A/B
the moment 0023's split exists to judge it. Two datasets, mixed-to-negative, every
number contaminated: we do not actually know.

**Fixing a single model type for both search and shipping.** On v8's winning
table: catboost 0.11207, elasticnet 0.11217, **lgbm 0.12141** — a gap of about
0.8 `val_se` between families. Model choice is not free, which is exactly why the
terminal set must be enumerated rather than assumed away.

**Enumerating the target scale in the terminal phase too** (3 scales × 3 models).
It would measure rather than derive, honouring 0021. It also widens the terminal
argmax from 3 to 9 candidates — `√(2 ln 9) ≈ 2.1` standard errors of luck instead
of 1.5 — to re-derive something 0022 already derives. Revisit when the third
split can price both sides.

## What it costs

**Feature–model interaction is real, and we are choosing to ignore it during
search.** A table that suits the probe may not suit the shipped model: an ordinal
encoding a tree splits happily on is not what a linear model wants, and the probe
will never notice. The mitigation is that the probe comes from the same family as
the likely winner, and that the terminal phase gets the last word. It is a
mitigation, not an answer.

**The agent loses a lever it sometimes used well.** Ten of sixteen runs found
their winner in a model variation. Most of those margins are inside the noise —
that is the case for this record — but "most" is doing work there, and we cannot
currently separate the ones that were real.

**Every number in this record is a `val_score`**, which is the contaminated one.
The 44% is a count and is solid; the margins and the model gaps are measured with
the instrument 0023 exists to replace. This change is justified by the *structure*
of the comparisons, not by their values, and it deliberately does not claim a
score improvement.

**We cannot verify it worked.** A feature-only search will report a different
`val_score`, and that number is not evidence. 0023 first, or this change is
another thing we believe.

## What follows

The two-level stop of [0014](0014-two-level-stop.md) collapses: with no model
experiments there is one patience, over tables. The `df.copy()` no-op that reset
`table_patience` — and thereby produced e11, the best model this harness has made
(0034's footnote) — stops being a loophole in a search that has only one level to
game.
