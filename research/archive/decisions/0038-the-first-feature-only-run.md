# 0038 — The first feature-only run: less optimism, and a prediction that failed

- **Date:** 2026-08-13
- **Status:** accepted (a measurement)
- **Related:** [0035](../../../docs/decisions/0035-the-search-is-over-features.md) (the change this
  tests), [0034](0034-the-tiebreak-was-tested-and-rejected.md),
  [0023](0023-the-third-split-has-evidence-now.md) (still unbuilt, and still the
  thing that would settle this properly)

## The design

`scripts/run_house_v9_feature_only.py` against run 8 as control. Same data file,
same two-sentence prompt, same model and route
(`fireworks/deepseek-v4-flash-0731`, thinking at high effort), same
`data_description.txt` through `column_notes`. **The only difference is the
harness.**

The claims were written into the script's docstring before the submission was
scored, so they could not be adjusted afterwards:

1. Kaggle comparable to or better than run 8's 0.12298.
2. Less optimism than run 8's +0.0113.
3. Fewer, more informative comparisons.

## The result

| | val | Kaggle | optimism | rank /3765 |
|---|---|---|---|---|
| **run 8** — catboost | 0.11171 | 0.12298 | **+0.01127** | 756, top 20% |
| **run 9** — elasticnet | 0.11579 | **0.12037** | **+0.00458** | **336, top 9%** |
| run 9's runner-up — catboost | 0.12078 | 0.12553 | +0.00475 | 1149, top 31% |

**All three claims passed.** Kaggle better by 0.00261; optimism less than half.

The shape of it matters more than the size: **run 9's validation score was worse
than run 8's and its Kaggle score was better.** A harness that reports a less
flattering number and delivers a better model is doing the thing 0019 named and
0023 exists to measure.

Cost went the same way. Run 9 took **2.3 minutes and 17 tool calls** to build 10
tables, every one of them scored. Run 8 took 14 experiments, five of which
changed only the model.

## The prediction that failed, and why it failed

Mid-run, before the submission was scored, the runner-up was flagged as the safer
pick on this reasoning: elasticnet's `val_score` of 0.11579 sat 0.014 away from
its `cv_score` of 0.13015, while catboost's two numbers agreed to 0.001. That
divergence was read as a winner's-curse signature — the selected candidate being
the one whose validation draw was luckiest.

**It was not luck. It was cross-validation's own downward bias**, and 0034
already describes the mechanism: fold models train on (K−1)/K of the rows while
the shipped model trains on all of them. On 1168 rows, a linear model fitted on
~930 of them is on a steep part of its learning curve, so `cv_score` understates
it badly. A boosted tree at that size is much flatter, which is exactly why
catboost's two numbers agreed and elasticnet's did not.

The consequences, in order of how much they should change what we do:

**Val ranked the two models correctly and cv ranked them backwards.** Val said
elasticnet then catboost; Kaggle said elasticnet then catboost. Cv said the
reverse. This is a fourth data point against "prefer `cv_score`", after 0034's
one win, one loss and one null — and the first one where the margin was not a
tie (0.005 of val, about 0.45 `val_se`).

**A val–cv gap is not a diagnostic for selection noise.** The two numbers differ
for a structural reason whose size varies with dataset size, model family and the
steepness of the learning curve. Reading their disagreement as evidence about
luck confuses a bias with a variance, and it produced a confident and wrong
recommendation here.

**The one selection event 0035 kept made the right call the one time it has been
tested.** That is not evidence it is safe — an argmax over three candidates is
still worth about 1.5 standard errors of luck — but it is the only evidence there
is, and it points the other way from the concern.

## What this does not establish

**n = 1.** The optimism improvement is 0.0067 against a `val_se` of about 0.011.
The direction is right; the magnitude is inside the noise of the very number
being compared, which is the joke 0023 exists to stop being funny.

**Two things changed, not one — in the headline.** Run 8 shipped catboost, run 9
shipped elasticnet, and run 8 never fitted a linear model at all. The ablation
below separates them, and the separation is clean; without it this record would
have claimed the harness caused something the model family might have.

### The ablation: the features did it

The enumerated terminal set was fitted on **run 8's own winning table** (`t11`),
with run 9's procedure and target scale, and the linear model was submitted:

| on run 8's `t11` | val | cv | Kaggle |
|---|---|---|---|
| xgb | 0.13131 | 0.13423 | — |
| catboost | **0.11207** | 0.11829 | 0.12298 (what run 8 shipped) |
| elasticnet | 0.11217 | **0.11357** | **0.12934** |

**Elasticnet on run 8's features scores 0.12934. The same model family on run 9's
features scores 0.12037** — 0.009 apart, four times the gap between the two runs'
headlines.

So the gain was **the feature search**, not the terminal enumeration. Run 9 built
a table on which a linear model generalises; run 8 did not. That is the claim
0035 makes, tested against the confound that could most easily have explained it
away, and it survives.

Also worth noting: the terminal argmax on run 8's table preferred catboost by
**0.0001 of val — one hundredth of a standard error** — and catboost went on to
beat elasticnet by 0.006 on Kaggle. That is the second time pick-best-by-val has
made the right call, and this one was decided by a margin indistinguishable from
zero. Two for two is not evidence of safety; it is evidence that we have not yet
seen it fail.

### The thing nobody was looking for: optimism belongs to the table

Four models now have both a validation score and a Kaggle score, two on each
run's winning table:

| table | model | optimism |
|---|---|---|
| run 8's `t11` | catboost | +0.01127 |
| run 8's `t11` | elasticnet | **+0.01717** |
| run 9's `t10` | elasticnet | +0.00458 |
| run 9's `t10` | catboost | **+0.00475** |

**They cluster by table, not by model.** Both models on run 8's table overstate
by 0.011–0.017; both on run 9's table overstate by 0.0046, and the two pairs do
not overlap. Swapping the model family changes the optimism by less than 0.006;
swapping the table changes it by a factor of three.

That is a sharper statement of what 0019 and 0034 describe. The winner's curse
was discussed as if it attached to the *model* the run selects — argmax over a
leaderboard of fits. It attaches to the **table**: a feature set chosen by
repeated querying of the validation half carries that selection with it, and
hands it to whatever is fitted on it afterwards. Run 8 reached `t11` through 14
experiments ranked on val; run 9 reached `t10` through 10 tables compared under
one fixed ruler.

Four points on two tables from two runs, so this is a hypothesis with a
measurement attached rather than a result. It is also the most testable thing to
come out of this run, and it predicts something specific: **the third split
(0023) should show optimism varying with how hard the table search worked, and
barely at all with which model is fitted at the end.**

## What would settle it

The third split (0023). Every number above is measured either on rows the search
selected against or on a Kaggle submission, and Kaggle is an oracle we can only
consult a few times per dataset — which is why five runs' worth of optimism
readings took two months to accumulate. A held-out split reports this for free on
every run, including on datasets with no competition attached.

Run 9 is evidence that the harness change was worth making. It is not evidence
about how large the effect is, and no run of this design can be.
