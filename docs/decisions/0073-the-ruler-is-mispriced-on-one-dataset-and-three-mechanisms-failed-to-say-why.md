# 0073 — The ruler is mispriced on one dataset, and three mechanisms failed to say why

- **Date:** 2026-08-17
- **Status:** accepted (a measurement; the fix it was expected to justify is
  **deliberately not built**)
- **Answers:** the plan at the top of `HANDOFF.md` — *"every decision a run makes
  goes through a ruler that has now been measured wrong four times, and a run makes
  15-25 of them"* — which asked for one script and named the falsifier.
- **Related:**
  [0035](0035-the-search-is-over-features.md) (what the probe is for, and
  why it must stay untuned),
  [0036](0036-the-pairing-asymmetry-was-not-there.md) (why a fixed estimator was
  thought to buy the pairing, and the family correlations that bear on the panel),
  [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md) (the sign reversal that started this,
  and its open item 2),
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (the block
  interval that this argues is the wrong instrument for this failure),
  [0066](0066-the-collapsed-column-wins-everywhere-except-where-it-is-graded.md) and
  [0067](0067-capacity-is-worth-a-tenth-of-that-on-a-table-that-has-features.md)
  (both records of extrapolating an uncontrolled comparison, the error this record
  refuses to repeat at the end),
  [0069](0069-feature-selection-pays-on-columns-that-mislead-not-on-columns-that-are-idle.md)
  (sizing a lever from a description, the freshest instance),
  [0017](0017-credit-by-ablation.md) (the rule this record's own first grid broke),
  [0055](0055-a-rename-makes-every-earlier-artifact-unreplayable.md) and
  [0030](0030-a-model-that-cannot-be-served-is-not-trained.md) (why parts of the corpus could not be re-ruled),
  `scripts/probe_ruler_budget_rank.py`, `scripts/probe_ruler_deflation_across_runs.py`,
  `_runs/ruler_budget_rank.csv`, `_runs/ruler_deflation_across_runs.csv`.

## The question

Every table decision a run makes goes through one number: the probe's
`probe_vs_source_table`, produced by a single lgbm at `n_estimators=200` scored on
the run's validation half. What ships is not that model — it is a blend of three
families, and 0054 measured the *sign* of one thing's value reversing between the
two regimes.

If the ruler shifts like that in general, then every table decision in this
project's history was taken on an instrument measuring a different model than the
one that ships. That is not a problem an interval fixes: **an interval measures
variance, and a shift is bias.** 0059 built a block interval and nothing is gated
on it; making it bind was the obvious next move and would not have caught either
documented failure (0066's leaky table read `sign_stable: True` at 2.9 sd; 0059's
own history arm had 5/5 blocks in favour and Kaggle read -0.0301).

So: re-rule the tables at a higher budget and see whether the ranking survives.

## What was measured

**Pass 1 — `scripts/probe_ruler_budget_rank.py`.** All 14 scored tables of Rossmann
v7's chain, at five budgets. v7's split is *copied, not recut*: probe scores are
measured on the validation half, and the same rows for every table is the whole
basis of the comparison. The 200-tree arm reproduces v7's journal to six decimals
on all 14 tables, which is what makes the rest readable.

**Pass 2 — `scripts/probe_ruler_deflation_across_runs.py`.** Every other run whose
journal carries probe scores, at `200 @ 0.10` against `3000 @ 0.03`: four datasets,
six runs, 55 comparable steps. The probe arrived with 0035, so the fourteen runs
before it have no scores to compare against and cannot enter. v7 is excluded from
pass 2 on purpose — it is the chain the hypothesis was formed on.

## Finding 1 — the probe inflates most of v7's chain

Not two steps of it. The chain's own worth, measured by the ruler and at capacity:

| | bare table `t2` | best table | chain is worth |
|---|---|---|---|
| 200 @ 0.10 (the incumbent) | 0.23635 | `t13` 0.13202 | **0.104** |
| 5000 @ 0.03 | 0.15617 | `t12` 0.12409 | **0.032** |

> **Two thirds of what this project's feature search appears to have bought is
> capacity the ruler did not have.**

Sign flips are not a separate phenomenon; they are where the inflation exceeds a
step's real value. The argmax moves off the table v7 shipped (`t13` -> `t12`) at
both well-behaved high budgets.

## Finding 2 — there is no budget at which the ruler is simply correct

Two beliefs died here, and the second is the important one.

**Rounds and learning rate are not interchangeable.** `probe_capacity.BUDGETS`
says the grid "trades learning rate against rounds along the one axis that matters:
total shrinkage". At matched product, `1000 @ 0.10` is *worse* than both
`3000 @ 0.03` and `5000 @ 0.03` on six of fourteen tables — the ordinary result
that a lower rate over more rounds generalises better.

**And more capacity is not simply truer.** Along the fixed-rate ladder
1000 -> 3000 -> 5000 @ 0.03, `t3` goes 0.19774 -> 0.15726 -> **0.16520** and `t5`
goes 0.17288 -> 0.15515 -> **0.15984**. Genuine overfitting. Eleven tables prefer
the largest budget; three do not.

> **Each table has its own capacity optimum, and the optimum depends on the table —
> which is the thing being ranked.** A fixed-budget ruler necessarily favours tables
> whose own optimum sits near that budget.

That circularity is the honest answer to "why is it so hard to get a proper signal
for the agent". There is no better single ruler waiting to be found: the ruler and
the shipped model are the same *kind* of object, every property that decides which
table is better is a property the ruler must share, and cheapness — the only reason
to have a separate ruler at all — is bought by dropping exactly those properties.

## Finding 3 — two clean sub-mechanisms, both nameable without a dataset

**A row filter's cost is training data, and a ruler too small to use that data
cannot see the cost.** v7's `t2->t3` discards 155,823 of 915,744 rows (17%) and
decays monotonically in *both* rate families: +0.01398 -> +0.00225 at lr 0.10, and
+0.01168 -> +0.00830 -> -0.00903 at lr 0.03. A monotone dose-response, the
strongest evidence shape available here.

**v7's largest fork reverses.** From `t3` the agent built two branches; the ruler
priced the target branch at +0.046 and the frequency branch at +0.018, so it took
the target branch:

| | 20 | 30 | 90 | 100 | 150 |
|---|---|---|---|---|---|
| `t4` frequency | 0.20446 | 0.18035 | **0.14487** | **0.14937** | **0.13898** |
| `t5` target | **0.17601** | **0.17288** | 0.15515 | 0.16591 | 0.15984 |

`t4` wins at all three high budgets, by 0.010 to 0.021, across two learning rates.
**Every table v7 shipped — including the `t13` that scored 0.13813 private —
descends from `t5`.**

## Finding 4 — it replicates on a second run, and on no other dataset

`rossmann_v8` reproduces both mechanisms independently: `t5->t6`, a target encoding
of the 1,115-way store identity, reads **+0.02590 at 200 trees and -0.02747 at
3000**; `t13->t15`, a row filter, reads +0.00970 -> -0.00155; `t15->t16` and
`t16->t18` are further store-identity encodings that both go positive to negative.
**Five of eleven materially-accepted Rossmann steps do not help at capacity.**

Elsewhere, nothing:

| run | material steps (gain at or above 0.002) | flips |
|---|---|---|
| house_prices_v9 (1,168 rows) | 7 | **0** |
| restaurant_v1 (227k rows, ordered) | 8 | **0** |
| restaurant_v2 | 2 | **0** |
| seguro_v3 / v4 (476k rows, auc) | 7 | 4, all noise-scale |
| **rossmann_v8 (916k rows)** | **15** | **7** |

## What was predicted, and what the predictions cost

Registered in the scripts' docstrings before each pass, and left there unedited.

**Pass 1.** Predicted a global Spearman above 0.9 (0.877 and 0.943 — half right),
at least two of the last six steps flipping (**refuted**: one), and the argmax
moving off `t13` (**confirmed** at two of four higher budgets). The reasoning was
wrong in a way the prediction concealed: it assumed **large steps are safe and
close calls are fragile**, and the truth is the reverse — the flips are the largest
steps in the chain, swinging 0.023 and 0.056 against tail flips of 0.0027.

**Pass 2.** Four parts, one confirmed, and it is the null one. house-prices showing
almost nothing was **confirmed** (0 flips, deflation 1.25) and it is the reading
that makes the effect about capacity rather than about probing in general. "Porto
Seguro deflates" was **refuted** — its steps are tiny and mostly negative, with
nothing to deflate. "15-25% of steps flip" landed numerically (28% at a material
threshold) **for the wrong reason**, and that is not a hit.

**Three mechanisms were proposed and none survived.**

1. *Target-summary tools are overvalued.* Refuted within pass 1: `t9->t10` and
   `t10->t11` are target encodings whose value **rises** with capacity.
2. *Deflation tracks key cardinality.* Refuted by pass 2, wrong sign
   (Spearman -0.412). Restaurant's 8,242-way `store_weekday` encodings are the most
   stable steps in the corpus (+0.0338 -> +0.0318), and its 829-way store encoding
   *appreciates* (+0.1902 -> +0.1938).
3. *Deflation tracks rows-per-group.* Unsupported: Spearman +0.358 against
   cardinality's +0.295. Two weak numbers, not a result.

> **The mispricing is real, large, and reproducible on Rossmann across two
> independent runs and two identified mechanisms. It is absent on three other
> datasets, and nothing proposed here explains which datasets it visits.**

## What was decided

**The two-budget panel is not built.** The design was concrete — report
`probe_vs_source_table` at both budgets, never reconcile them, let the disagreement
itself be the signal, leave `best_table` untouched in the same change (0017) — and
it fits Ulf's standing principle that the harness should guide the agent rather
than decide for it. It is refused anyway, on its own cost argument: the second arm
costs about **10x the current probe on every dataset** (fit time at 3000 rounds is
dominated by tree count, not row count — 12s on house-prices' 1,168 rows), to catch
an effect visible in one dataset of four. A flat cost against a benefit that does
not appear on three quarters of the corpus is not a trade this evidence supports.

**The formal falsifier was not met, and that is a flaw in the falsifier.** It
required under 5% of steps to flip; the corpus gave 35%, but 9 of 19 flips are
Porto Seguro steps worth under 0.005 auc on a 3.6%-positive target — a ruler and a
second opinion disagreeing about noise. Flipped steps are *smaller* than agreeing
steps (median |gain| 0.00217 against 0.00369), which is the noise signature and the
opposite of v7. Screening by magnitude was missing from the falsifier when it was
written. **Not established** is a fairer verdict than confirmed.

**And the claim that would have made this urgent is not made.** It is very tempting
to say the misread fork cost v7 its score. That extrapolates a step-level reading to
a chain-level outcome, which is precisely the uncontrolled comparison 0066 and 0067
both record against this author. The `t4` branch might have found nothing.

## What it costs

Two things are now known and unfixed, which is worse than not knowing them.

The 200-tree ruler is **not** a neutral instrument on Rossmann-shaped data, and
every Rossmann table decision in `_runs/` was taken with it. Nothing is being done
about that, and no run can detect it from inside itself.

And the chain-value collapse (0.104 -> 0.032) means the *reported* worth of feature
work in this project's journals is inflated wherever the effect operates. Those
numbers are not being restated — they are inputs to runs that already happened, per
0061's rule — so a reader has to bring this record with them.

## What was rejected

**Making 0059's block interval bind.** The obvious fix, and it addresses the wrong
quantity: every failure documented here is a shift, and an interval measures spread.

**Raising the probe's budget.** Cost is not the objection — 5s to 45s per table,
about 13 minutes on a run that takes 8-22. The objections are that no budget is
correct (Finding 2), and that picking the one that looked good on Rossmann is
0022 one level up.

**A multi-*family* panel.** Argued for on the grounds that CatBoost's ordered target
statistics would make it structurally immune to a target encoding, then withdrawn
on reading the code: `_feature_frame` keeps only numeric dtypes and `_make_model`
never passes `cat_features`, so **no model in this project has ever been told a
column is categorical.** CatBoost receives a plain numeric matrix and its target
statistics never engage. 0036's measured family correlation of 0.86-0.88 therefore
stands unopposed, and the panel — had it been built — would have varied capacity.

That withdrawal left something larger behind, recorded here because it is nobody's
decision and therefore nobody's to find: **the harness made agent-chosen encoding
the only route from a categorical column to a model, and then measured that route
with a ruler that overpays for it.** It is general in the way `CLAUDE.md` requires —
"models are never told which columns are categorical" names no dataset. It is *not*
settled that declaring them would be an improvement, and this record's whole subject
is why the probe cannot be the instrument that judges it.

## The next measurement, and it is one script

The rows-per-group idea is post-hoc and unsupported, but it makes a sharp
prediction. Rossmann carries ~821 rows per store; Restaurant ~274. If capacity
substitutes for an encoding only once it has enough rows per group to learn the
effect by splitting, then **Restaurant's steps should flip too, at a budget higher
than `3000 @ 0.03`.**

That is `probe_ruler_budget_rank.py` pointed at `restaurant_v1` — no new machinery.
Decisive both ways: flips appear and the mechanism is capacity relative to group
size, which is general and nameable without a dataset; nothing moves at any budget
and the effect is a Rossmann property, the panel stays unbuilt, and 0070's
architecture answer returns to the front.

Whoever runs it should register the prediction first and screen flips by magnitude,
which this record's own falsifier failed to do.
