# 0036 — The pairing asymmetry was not there; the probe is justified by coverage

- **Date:** 2026-08-13
- **Status:** accepted
- **Corrects:** [0035](0035-the-search-is-over-features.md) — its central
  *mechanism*, not its decision. Written the same day, measured a few hours
  later. 0035 stays as written.
- **Related:** [0028](0028-the-yardstick-is-fixed-the-bar-is-not.md) (whose open
  `min_gain_ratio` this record supplies a candidate for),
  [0034](0034-the-tiebreak-was-tested-and-rejected.md)

## What 0035 claimed

0035 argued that a run's leaderboard mixes two kinds of comparison and that one
of them is much noisier:

> Two feature sets scored by the same estimator on the same rows have highly
> correlated errors, so the difference between them is measured far more
> precisely than either score … Two *different* estimators on the same rows make
> different mistakes. `Cov` falls, and the variance of the difference rises
> toward the sum.

From which: model-vs-model comparisons are the least reliable thing we measure,
and 44% of the search budget was going into them.

**The second half of that is false.**

## The measurement

House-prices v8. Each candidate fitted, then predicting the *same* 292
validation rows; those rows resampled with replacement 400 times, both scores
recomputed on each resample. `sd(A−B)` is how precisely the comparison is
determined; `sqrt(varA + varB)` is what it would be if the two scores were
independent.

| comparison | sd(A−B) | if independent | ratio | corr |
|---|---|---|---|---|
| **table** t10 vs t14 (both lgbm) | 0.00413 | 0.01364 | 0.30 | 0.909 |
| **table** t10 vs t16 (both lgbm) | 0.00408 | 0.01427 | 0.29 | 0.918 |
| **table** t13 vs t14 (both lgbm) | 0.00236 | 0.01342 | **0.18** | 0.971 |
| **model** lgbm vs catboost (t10) | 0.00421 | 0.01434 | 0.29 | 0.914 |
| **model** lgbm vs catboost (t14) | 0.00409 | 0.01340 | 0.31 | 0.909 |
| **model** lgbm vs xgb (t10) | 0.00515 | 0.01395 | **0.37** | 0.866 |

**Pairing is real and large — every comparison is 3 to 5 times better determined
than independent scores would be. It is also not asymmetric.** Table comparisons
land at 0.18–0.30, model comparisons at 0.29–0.37. The ranges overlap.

## Why this should have been predictable

It was already in our own notes. The Rossmann blending work measured
**GBDT-vs-GBDT residual correlation at 0.86–0.88** and drew the conclusion that
blending is not our gap, because three boosted trees on one feature matrix make
one set of mistakes.

That number *is* the statement that swapping estimators leaves predictions highly
correlated. 0035 asserted the opposite one paragraph away from citing it.

What actually governs the covariance is **how similar the two predictors are**,
which has nothing to do with whether the thing that was varied is called a model
or a table. The tightest comparison measured here is `t13 vs t14` — two adjacent
lineages one step apart — at 0.18, and the loosest is `lgbm vs xgb`, two
different libraries, at 0.37. Similar predictors, precise comparison. That axis
is continuous and it cuts across the model/table distinction entirely.

## What still stands in 0035

Its decision, and most of its evidence:

- **222 experiments, 97 of them (44%) varying only the model or its parameters.**
  A count, unaffected.
- **Winning margins of 0.03, 0.05 and 0.18 `val_se`.** Unaffected — and
  strengthened below.
- **"Search where the space is unbounded, enumerate where it is small."** This
  never depended on the pairing argument. Spending 44% of the queries in a run
  choosing among four things you could simply train all of is wasteful whatever
  the covariance turns out to be.
- The probe, the enumerated terminal phase, the target scale moving to the
  metric with a `TaskSpec` override.

What falls is one sentence of mechanism — *model comparisons are the least
reliable comparisons we make* — and the rhetorical weight 0035 put on it.

## The probe's actual justification

Rewritten from what the run data supports.

**1. Coverage.** Scoring every table of v8 with one fixed estimator produced
numbers for **16 tables where the run itself had numbers for 8**. The agent
builds chains — t12 → t13 → t14 — and trains only the endpoint, so the run's
own record has holes exactly where the interesting comparisons are. The lineage
analysis behind 0035 hit this directly: of 24 recombined tables, **22 could not
be compared to the branch they continued, because that parent was never
trained.** A cheap ruler applied to everything closes that, and makes ablation
credit (0017) mostly free as a side effect.

**2. Removing a confound.** Without a fixed estimator, "is t16 better than t10"
is answered by whichever model the agent happened to try on each. In v8, t10's
number came from catboost and several tables' numbers came from lgbm; the
comparison was never about the tables. This is 0017's own rule — *a comparison
must differ by exactly one thing* — applied to selection rather than attribution.

**3. Cost.** The ranking number is the validation score, which comes from a model
fitted on all of a table's rows, so the fold count never enters it: **one fit per
table, not `k+1`.** That is what makes scoring every table affordable, and it is
what the next step (automatic scoring on table creation) rests on.

None of these three is a claim about covariance.

## A yardstick, which is the useful part

`val_se` for v8 is **0.01116**. Against that, nothing any run has produced
resolves, and 0028 left `min_gain_ratio` open for exactly that reason.

But `val_se` is the uncertainty of a *score*, and every decision a run makes is
about a *difference*. Measured above, the paired sd of a difference is
**0.0024–0.0041** — roughly 3x tighter. Comparing a table to its parent by a
paired bootstrap over the validation rows is a strictly better instrument than
comparing two point scores against a pooled `val_se`, and it needs no fitted
constant: it is the measurement's own scale.

This is a candidate answer to 0028's open question and should be built and tested
as one, not adopted because it sounds right.

It does not rescue the runs we have. v8's winning margin was **0.00036**, still
about six times smaller than the tightest paired sd measured here. Those
selections were noise under either yardstick, which is 0034's conclusion arrived
at from a second direction.

## What this record does not establish

**One dataset, one run, one metric.** 292 validation rows, `rmsle`, 400
bootstrap resamples, six comparisons. Porto Seguro would test it where it is most
likely to differ: a classification metric, 119,042 validation rows, and an
`elasticnet` in the mix — a genuinely different predictor, where the covariance
really might fall. The v8 model comparisons were all boosted trees, so what was
measured is mostly "GBDTs resemble each other", which we already knew.

Until that runs, the honest statement is: **the asymmetry 0035 assumed is absent
among tree models on one small regression dataset, and the probe is justified
without it.**
