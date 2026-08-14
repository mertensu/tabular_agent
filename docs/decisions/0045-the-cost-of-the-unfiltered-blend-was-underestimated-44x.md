# 0045 — The cost of the unfiltered blend was underestimated 44x

- **Date:** 2026-08-14
- **Status:** accepted (a measurement), **not implemented** — deliberately
- **Related:** [0044](0044-what-ships-is-the-average-not-the-argmax.md) (whose cost
  section this corrects), [0042](0042-the-terminal-argmax-shipped-the-worst-model.md),
  [0023](0023-the-third-split-has-evidence-now.md) (the rule that says why this
  changes no code today)

## What happened

0044 was committed on the strength of one Kaggle submission and a sealed-rows
estimate. A second submission was scored hours later, and it is the four-way
blend — the one the harness now actually ships.

| submission | private | public | vs argmax |
|---|---|---|---|
| `submission.csv` — argmax winner (xgb) | 0.14356 | 0.13960 | — |
| `submission_blend_all.csv` — **what ships** | **0.14172** | 0.13169 | **+0.00184** |
| `submission_blend_lgbm_xgb_catboost.csv` | **0.13813** | 0.12932 | **+0.00543** |

**0044's decision survives.** An equal-weight average beats the argmax on an
external set: +0.00184. That is the claim the record was written on and it holds.

**0044's cost estimate does not.** It said this, and the number was wrong:

> The empirical answer is that it barely matters: on v7's seal the four-way
> average scored 0.13618 against the three-way 0.13610, a difference of 0.00008.

On Kaggle the same comparison is **0.00359** — about **44x** larger, and it is
two thirds of the entire benefit of blending. Including `elasticnet` costs more
than the argmax-versus-average decision gains.

## Why the seal got it wrong

The sealed tenth said 0.00008; Kaggle said 0.00359. Both are held-out sets, both
were scored honestly, and they disagree by a factor of 44 on the *same four
models*. Two candidate mechanisms, and this run cannot separate them:

- **Horizon.** 0043 already recorded that v7's sealed rows are 16 Jun – 31 Jul
  while Kaggle's test set is 1 Aug – 17 Sep. A linear model extrapolating a
  trend degrades with distance from its training data in a way boosted trees —
  which predict a constant outside their splits — do not. If elasticnet's error
  grows with horizon, the seal measures it at six weeks and Kaggle at four
  months, and the seal will understate it every time.
- **Noise.** 0.00008 is a difference between two numbers on 51,290 rows of one
  contiguous period. 0042 already established that this quantity is n=1 in the
  unit that varies. A difference this small on such a measurement was never
  distinguishable from zero, and treating it as an estimate of anything was the
  error.

The second is enough on its own, and it is the more useful reading: **0.00008 was
not a small effect, it was an unresolved one**, and 0044 quoted it as though the
smallness were the finding.

## What this does not license

**Filtering `elasticnet` out of the blend.** That is the obvious response and it
is refused, for the reasons 0044 already gave and one it did not:

- **House v9 still refutes the category rule.** Elasticnet was the best model on
  that board, beating catboost by 0.005 on Kaggle. "Drop linear models" deletes
  that run's winner. One dataset saying "drop it" and another saying "keep it" is
  not evidence for a rule; it is evidence that the rule is dataset-specific, which
  is the thing that may not enter `src/`.
- **Acting on this number is choosing a rule by consulting a holdout.** It is the
  same move 0042 refused six days ago, and the fact that the holdout is Kaggle's
  rather than ours does not change what it is — the submissions have now been
  scored, so the private leaderboard is no longer an unconsulted set for this
  question. Spending it to pick the blend membership leaves nothing to judge the
  result.

The quiz question that produced this record asked which of 0044's two arguments
would survive a skeptic. The answer was house v9, and that **the 0.00008 justified
nothing — it only reported that on one occasion the rule looked cheap**. That
answer was given before this submission was scored, and it is the reason this
record changes no code.

## What it does license: a hypothesis with a mechanism

Not "drop linear models" but something narrower and testable, which is the real
content of the horizon reading above:

> **An equal-weight member contributes `1/k` of every prediction with no bound on
> how wrong it can be.** The blend has no way to notice a member that fails badly
> rather than poorly, and nothing in 0044 looks.

0044 named this as its open edge and had no measurement of it. It has one now:
elasticnet at 0.195 rmspe against 0.141 on the seal is exactly the "mildly bad"
case, and it cost 0.0036 out of a 0.0054 benefit. A genuinely broken member — a
fit that collapses to a constant, a scale error — is unmeasured and would cost
more.

What a defensible response would need, and none of it is a category rule:

- A **bound stated in advance** on how far a member may sit from the others before
  it is excluded, expressed in the run's own yardstick rather than in model
  families, and fixed before any held-out set is read.
- Or **weights that are not equal but are also not fitted** — inverse-variance,
  say — which reintroduces the question of what estimates the variance, and 0022
  says it must not be `val`.
- Or accepting the cost, on the grounds that a rule with a constant in it is the
  failure 0034 refused twice.

## Registered

1. **On the next sealed run, the gap between the full blend and the blend without
   its worst-scoring member is larger than the seal suggests.** Stated before that
   run exists. If it is small there too, the horizon reading is wrong and the
   noise reading is right.
2. **0044's claim 1 — the blend's optimism is smaller than the selected
   candidate's** — is unaffected and still open.

## What was corrected, and what was not

0044 is not edited. Its decision stands: the harness ships the average, and that
average beat the argmax on an external set. Only its cost section is wrong, and
this record is where a reader finds that out.
