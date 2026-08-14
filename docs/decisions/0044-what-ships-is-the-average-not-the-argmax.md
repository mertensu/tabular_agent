# 0044 — What ships is the average of the terminal fits, not the best of them

- **Date:** 2026-08-14
- **Status:** accepted and implemented
- **Decided by:** Ulf
- **Related:** [0042](0042-the-terminal-argmax-shipped-the-worst-model.md) (the
  measurement and the pre-registration this implements),
  [0035](0035-the-search-is-over-features.md) (whose terminal enumeration this
  changes the ending of), [0023](0023-the-third-split-has-evidence-now.md) /
  [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md) (the seal
  that judges it), [0034](0034-the-tiebreak-was-tested-and-rejected.md),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md)

## The problem

`finalize_run` fitted an enumerated set of model families on the winning table
and shipped the one with the best validation score. On Rossmann v7 that argmax
shipped the **worst of the three boosted trees** on rows it had never seen, and
0042 measured what the alternatives were worth on the sealed split:

| | sealed rmspe |
|---|---|
| xgb *(shipped)* | 0.15253 |
| catboost *(best single)* | 0.14113 |
| lgbm + catboost | **0.13432** |
| lgbm + xgb + catboost | 0.13610 |
| all four | 0.13618 |

Every average beat every single model, by about seven times what the whole
feature search bought from t8 onward. 0042 accepted the diagnosis and
**deliberately did not implement it**, because the numbers came from v7's own
holdout and choosing a rule by consulting a holdout is the failure the third
split exists to detect, one level up. It registered the claim for a later run
instead.

## What changed since 0042

**The blend was scored on Kaggle's private leaderboard**, using the submissions
`scripts/blend_submission.py` had already written:

| | private | public |
|---|---|---|
| `submission.csv` — the argmax winner (xgb) | 0.14356 | 0.13960 |
| `submission_blend_lgbm_xgb_catboost.csv` | **0.13813** | **0.12932** |

**+0.00543 on the private set.** 0042's claim 1 passes on rows that are not
merely unread by this run — they are unread by this *project*, held by a third
party, and were never consulted for any decision here before this one.

Being precise about what that is and is not: it is **not** a fresh run. Same
models, same table, same 40 minutes of agent work; only the rows judging them
are new. What it rules out is the specific failure 0042 was protecting against —
that the effect lives in v7's sealed tenth and nowhere else. It cannot rule out
that equal-weight blending happens to suit Rossmann. That needs a second dataset,
and every run from here produces one.

## The decision

**`finalize_run` ships an equal-weight average of every terminal fit that
trained.** No weights are fitted, no candidate is dropped, and nothing chooses.

The whole argument is in what the harness stops doing. An argmax over `k`
candidates costs roughly `sqrt(2*ln k)` standard errors of optimism *by
construction* — it does not pick the best candidate, it picks the luckiest one,
and luck does not repeat on the next set of rows. An average pays none of that,
while still collecting whatever the members' residual correlations of 0.86–0.88
leave on the table. The measured optimism of v7's shipped model was **+0.0227**;
most of that was the price of having chosen it.

### Every candidate, with no filtering by score

The tempting refinement is to keep the boosted trees and drop the linear model —
on Rossmann elasticnet sat 0.054 of rmspe behind the next-worst candidate.

Rejected, for two reasons that point the same way. **House v9 refutes the
category:** elasticnet was the *best* model there, beating catboost by 0.005 on
Kaggle, so a rule that drops linear models would have thrown away that run's
winner. And **any filter is a selection**, which is the cost this record exists
to stop paying; a filter chosen by looking at scores is the argmax again, wearing
a different hat.

The empirical answer is that it barely matters: on v7's seal the four-way average
scored 0.13618 against the three-way 0.13610, a difference of 0.00008. An
equal-weight average is not very sensitive to one dissenting member.

**A residual risk, named rather than hidden:** the variant with Kaggle backing is
the three-GBDT blend, and what ships is the four-way. They are separated by
0.00008 on the only set that has judged both, which is evidence but not the same
evidence.

### One survivor is not a blend

If only one family trains, that model ships as before. The bundle format already
carried a list of models and a weight vector; this is the first time the library
writes more than one into it.

## A defect found while implementing it

`predict_bundle` combined its models **on the model's own scale and inverted the
target transform once, at the end.** Under `log1p` that is a *geometric* mean of
sales. Every measurement in 0042, and both Kaggle submissions, were arithmetic
means — `blend_submission.py` inverts first and says so in a comment.

So the code as written would have shipped an estimator that had never been
scored against anything, while the record claimed a number belonging to a
different one. `predict_bundle` now inverts each model's predictions before
applying the weights.

**Nothing on disk changes.** For a single-model bundle the two orders are
identical, and until this record every bundle the library wrote had one model.
The only multi-model bundles in the repo are in `notebooks/phase1`, on a binary
task with no transform, where the orders also agree.

`_score_on_held_out` takes a list of models for the same reason: a blend's score
is comparable to its members' only if it was produced by the same code.

## What the seal now records, and why both rows are kept

The holdout entry gained a row for the blend, and the flags were split:

- **`selected`** — the candidate the *search* crowned. Unchanged meaning, because
  the optimism of the selected candidate is the series 0039 and 0040 have been
  tracking across runs, and folding it into "what shipped" would break that
  series at exactly the run where the blend arrived.
- **`shipped`** — what the bundle predicts with.

On a blended run these are different rows, and their difference is 0042's A/B
**recomputed by every run from now on** rather than by a script afterwards. This
is the point: the decision stays falsifiable. If blending is wrong for some
dataset, that run's own report says so in a line.

`selected` still picks nothing. The members are fixed before any sealed row is
read — `_blend_members` runs off the terminal fits, and `_score_on_holdout`
receives the finished blend as an argument, exactly as it receives the finished
winner (0023 rule 2).

## What was rejected

- **Letting the agent choose between argmax and blend.** 0042's reasoning holds:
  it would choose on `val`, the signal that just failed. An argmax does not become
  better by moving who performs it.
- **OOF-optimised weights.** This is what 0035 measured and rejected, and its
  measurements were right about *that* estimator: fitted weights are a selection,
  on the quantity 0022 says is inflated. On Rossmann the optimiser put 0.8 on the
  model with the best `cv_score`. Equal weights fit nothing.
- **Blending on the model's scale.** Never measured; see above.
- **Ranking terminal fits by `cv` instead.** 0042 rejected it on house v9, and it
  is moot: nothing is ranked now.
- **Making it opt-in, as the seal is (0041).** The seal was opt-in because turning
  it on takes rows away and makes the run incomparable to earlier ones. Blending
  takes nothing away — the fits already existed, and both numbers are still
  reported — so there is no incomparability to opt into.

## What it costs

- **What ships was never cross-validated.** A blend has no folds of its own, so
  the report prints no `cv_score`, no fold spread, and no per-fold table for it.
  The argmax winner's `cv_score` is still on the finalize result, because it is
  still the run's record of how far the search drifted from its folds — but the
  run no longer ends with a fold estimate of the *shipped* object.
- **The bundle is `k` times larger and `k` times slower to predict with.** Four
  boosted-tree models rather than one.
- **Attribution is unchanged and slightly less exact.** `journal.attribution`
  runs off the shipped table's lineage, which every member shares, so the feature
  story is intact — but "which columns earned the score" now describes a score no
  single fitted model produced.
- **`val_se` for the blend comes from the same bootstrap as everyone's**, which
  on an ordered split is `None`, which is the open problem 0042 named and this
  record does not touch.

## Registered for the next sealed run

1. **The blend's optimism is smaller than the selected candidate's.** This is the
   mechanism stated as a prediction: an average makes no selection, so it has no
   luck in its validation score to give back. v7 measured +0.0227 for the argmax
   winner; the blend's own gap has never been measured on a run where it shipped.
2. **The blend beats the argmax winner on the sealed rows**, on a dataset that is
   not Rossmann. This is the generalisation Kaggle cannot supply.
3. **0039's pattern holds a fourth time** — the selected candidate is the most
   optimistic row on the ballot — now with a shipped model that is not on it.
