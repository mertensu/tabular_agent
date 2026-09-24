# 0042 — The terminal argmax shipped the worst model, and the seal caught it

- **Date:** 2026-08-13
- **Status:** accepted (a measurement and a diagnosis), **not implemented**
- **Related:** [0023](0023-the-third-split-has-evidence-now.md) /
  [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md) (the split
  that made this visible), [0035](../../../docs/decisions/0035-the-search-is-over-features.md) (whose
  terminal enumeration this is about), [0034](0034-the-tiebreak-was-tested-and-rejected.md)
  (the cv-versus-val question, reopened with a fifth data point),
  [0022](../../../docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md),
  [0029](0029-the-harness-owns-the-exam.md)

## What happened

Rossmann v7, the first run with a sealed third split. Four models fitted on the
winning table, ranked on the validation half, best one shipped:

| model | val | **sealed** | optimism | |
|---|---|---|---|---|
| **xgb** | **0.12980** | **0.15253** | **+0.02273** | shipped |
| lgbm | 0.13202 | 0.14145 | +0.00943 | |
| catboost | 0.14329 | 0.14113 | −0.00217 | |
| elasticnet | 0.20556 | 0.19493 | −0.01063 | |

**The harness shipped the worst of the three boosted-tree models.** On the
validation half xgb led; on 51,290 rows nothing in the run could read it is last
of the three by 0.011 of rmspe.

This is seguro v2's failure repeated — that run shipped `e8` over `e11` and lost
0.0120 of gini — with one difference that is the entire point of 0023: **v2
needed a Kaggle submission to find out, and v7 found it itself.**

## The diagnosis, and it is not noise

The obvious reading is that the argmax got unlucky. It does not survive contact
with the numbers: the validation half here is **50,175 rows**, xgb beat lgbm by
0.0022, and a 0.011 reversal is not sampling wobble at that size.

What is actually wrong is that under `split_on` the validation half is **one
contiguous period**, and the quantity being estimated is "which model wins on the
*next* period". xgb was genuinely the best model on 2 May – 15 Jun and genuinely
the worst on 16 Jun – 31 Jul. The ranking is n = 1 in the unit that varies.

The harness already knew this. It is why `val_se` is `None` on an ordered split,
in `_score_on_val`'s own words:

> resampling within it measures how much the score wobbles *inside* that period
> while saying nothing about the thing that actually varies — which period it is.

So the library declines to compute a yardstick for `val_score` because the
replication unit is wrong, and then ranks by that number with no yardstick at
all, one function away. The defect was documented before it was committed.

## Replaying the alternatives over every run with a known truth

The 0034 method: do not decide from the run that raised the question.

| run | split | `val` picks → truth | `cv` picks → truth | better |
|---|---|---|---|---|
| **rossmann v7** | ordered, 50k val | xgb → **0.15253** | lgbm → **0.14145** | **cv by 0.0111** |
| house v9 | random, 292 val | elasticnet → **0.12037** | catboost → **0.12553** | **val by 0.0052** |
| seguro v4 | random, 119k val | lgbm → 0.27670 | lgbm → 0.27670 | tie |
| seguro v3 | random, 119k val | elasticnet → 0.24907 | elasticnet → 0.24907 | tie |
| seguro v2 *(pre-0035)* | random, 119k val | e8 → 0.26522 | e11 → **0.27724** | **cv by 0.0120** |

Neither rule wins outright, and **the dividing line is not the split type** —
seguro v2 is a random split where cv was right.

What separates them is how much cv's fold-training-size reduction distorts the
*order*. House v9 is 1168 rows, so fold models train on ~930 — the steep part of
a linear model's learning curve, which reorders families (0038 recorded exactly
this). Seguro and Rossmann are 476k–916k rows, where fold models are off that
curve; cv's *level* is badly biased there (rossmann cv 0.24 against val 0.13,
because forward-chaining folds train on a fraction of the history) but the bias
is near-parallel across families, so the ranking survives.

That is a real mechanism, and it is still not enough to justify "use cv when
there are enough rows", because *enough* is a constant, and 0034 refused to fit a
constant to three points. It has four now.

## The measurement that reframed the question

If the argmax is the problem, the fix is not a better argmax. Equal-weight
averages of the same four fits, scored on the same sealed rows:

| | sealed rmspe |
|---|---|
| xgb *(shipped)* | 0.15253 |
| catboost *(best single)* | 0.14113 |
| **lgbm + catboost** | **0.13432** |
| lgbm + xgb + catboost | 0.13610 |
| all four | 0.13618 |

**Every blend beat every single model.** The three-family equal-weight average is
0.0164 better than what shipped — about seven times what the whole feature search
bought from t8 onward (0.0024).

The mechanism is not in doubt: an argmax over k candidates costs roughly
`sqrt(2*ln k)` standard errors of optimism *by construction*, and an average
costs none, while still gaining from residual correlations being 0.86–0.88 rather
than 1.0.

**0035's stated reason for rejecting a blend does not apply to this blend.** That
docstring cites Rossmann (blend 0.12991 against 0.12741) and house-prices
(+0.00024). Both are **OOF-weighted** blends scored on **val** — weights fitted on
the quantity 0022 says is inflated (on Rossmann the optimiser put 0.8 on the model
with the best `cv_score`), and judged by the number the search had already been
steering. Equal weights against sealed rows had never been tried.

## Why this record does not change the code

**Because the numbers above came out of v7's own sealed rows.** Choosing a
selection rule by consulting a holdout is what 0023 exists to prevent, one level
up. Implementing the blend on the strength of this table would make v7's holdout
the thing the harness is fitted to, which is the failure the third split was built
to detect.

So this is a hypothesis, and the next run is its test.

### Registered for the next sealed run

1. **An equal-weight blend of the terminal candidates beats the argmax winner on
   the sealed rows.** Registered before the run, judged on a seal that has never
   been consulted for this purpose.
2. **The gap is larger than the spread among the individual candidates** — i.e.
   the blend is not merely landing between them, which is what "averaging helps"
   would look like if it were only avoiding the worst case.
3. **The selected candidate is the most optimistic one** (0039, now held three
   times, and in v7 with the confound broken — see below).

## What was rejected, and why

- **Letting the agent choose among the finalists.** It would choose on `val`, the
  signal that just failed, or on `cv`, which needs the constant we do not have.
  It also reopens the surface 0035 closed after counting 97 of 222 experiments
  spent on model variation for winner margins of 0.03–0.18 `val_se`. An argmax
  does not become better by moving who performs it.
- **Ranking terminal fits by `cv` unconditionally.** House v9 refutes it.
- **Ranking by `cv` above a row threshold.** The mechanism is sound; the constant
  is fitted to four points.
- **Always shipping the probe's fit.** Zero selection over models, and it would
  have helped here (lgbm 0.14145 against xgb 0.15253) — but it discards real
  family gains, house v9's elasticnet beat catboost by 0.005 on Kaggle, and it
  makes every run's outcome hostage to one fixed estimator.
- **Selecting on the sealed rows.** Then it is the optimised number and we have
  three splits with two splits' worth of protection (0023).

## Two things this run also settled

**0039's hypothesis held a third time, and this time without its confound.** Both
earlier pairs had the selected model being lgbm — the probe's own family — so
"selection inflates" and "lgbm inflates more than a linear model" predicted the
same thing. Here the probe is lgbm and **the selected model is xgb**, and the
ordering is still by selection: shipped xgb +0.0227, then the probe that chose the
table +0.0094, then two models that selected nothing at −0.0022 and −0.0106.

**The drift confound 0041 named is bounded, and smaller than feared.** All four
models are scored on the same sealed rows, so period drift is a common additive
shift. If drift dominated, all four gaps would be positive; **two are negative**.
The model-to-model spread is therefore selection, not the calendar. 0041's warning
stands for the *level* of any single gap and not for comparisons within the
ballot.

## Open, and named rather than fixed

- **The agent reads the sealed number.** `finalize_run` returns `holdout_score`
  and `optimism`, and v7's agent reasoned about them in its closing summary
  ("the optimism gap of ~0.023 ... I considered that a fair trade"). The run is
  over, so it cannot act — but that is a convention about *when* the tool is
  called, not a structural bar, and under episodes or any post-finalize work it
  would stop being true. 0023's rule 1 deserves a stronger guarantee than call
  ordering.
- **A hang has no timeout anywhere in the library.** v7's first attempt sat 105
  minutes on one gateway request: socket open, 0% CPU, no bytes. The drivers
  retry provider *errors*; a hang is not an error. Fixed in
  `run_rossmann_v7_sealed.py` with a `timeout` in `MODEL_SETTINGS`, which leaves
  every other driver exposed and the library itself unguarded. Same shape as
  0033's still-open single-fold timeout.
