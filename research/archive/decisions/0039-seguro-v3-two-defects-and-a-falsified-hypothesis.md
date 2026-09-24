# 0039 — Seguro v3: two defects, and 0038's hypothesis falsified

- **Date:** 2026-08-13
- **Status:** accepted (a measurement)
- **Refines:** [0038](0038-the-first-feature-only-run.md) — its "optimism belongs
  to the table" hypothesis, which this record's registered test rejects. 0038
  stands as written; it labelled the claim a hypothesis and named the test that
  would break it, and the test broke it.
- **Related:** [0035](../../../docs/decisions/0035-the-search-is-over-features.md),
  [0030](../../../docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md),
  [0034](0034-the-tiebreak-was-tested-and-rejected.md)

## The run

`scripts/run_seguro_v3.py` against seguro v2 as control: same file, prompt,
model, route and metric, harness only. 595,212 rows, 3.6% positive, Normalized
Gini. Eight tables built, all scored, one conversation.

**The feature search did what it should.** The probe found `t3` at gini 0.29066
against 0.28192 for the raw work table, and reported honestly that seven of the
eight tables were *worse* than what they were built from. On a dataset where
feature engineering is famously near-useless — v2's own best model was trained on
the untouched work table — that is the right shape of answer.

Then the run gave it all back.

## Defect 1: the ruler was not on the terminal ballot

`TERMINAL_MODEL_TYPES` was `("xgb", "catboost", "elasticnet")`. The probe is
`lgbm`. So the run measured every table with lgbm, chose the table lgbm liked
best, and packaged the best of a set that excluded lgbm.

| on `t3` | val gini | Kaggle private |
|---|---|---|
| probe / lgbm — **not a candidate** | **0.29066** | **0.27470** |
| elasticnet — shipped | 0.25389 | **0.24907** |

**The omission cost 0.02563 of gini on the private leaderboard** — twice what
0034 records the tiebreak costing, and it was a design oversight rather than
noise. I wrote that tuple in 0035 and never justified excluding the ruler.

The fix is to put `PROBE_MODEL_TYPE` in the set, which turns an invariant that
was merely hoped for into one that holds by construction: **the shipped model is
never worse than the number the search was ranked on**, because the probe's own
fit is on the ballot.

### What this is not evidence for

The tempting reading — lgbm is simply the better family here — is wrong, and
checking it is what produced the useful part. Refitted on `t3` with comparable
capacity:

| | auc | gini | sec |
|---|---|---|---|
| lgbm, `n_estimators=200` | 0.64533 | 0.29066 | 2 |
| xgb, `n_estimators=200` | 0.60536 | 0.21072 | 1 |
| catboost, `n_estimators=200` | 0.61541 | 0.23082 | 4 |
| xgb, 1000 rounds, lr 0.05, depth 5 | 0.63705 | 0.27409 | 5 |
| catboost, 1500 rounds, lr 0.05 | 0.64175 | 0.28351 | 28 |
| lgbm, 1000 rounds, lr 0.05 | 0.63621 | 0.27242 | 7 |

The 0.08 spread at library defaults collapses to under 0.02 once the families get
comparable budgets. `n_estimators=200` means very different things to three
libraries — xgb's default learning rate is 0.3, catboost's schedule wants far
more iterations — and lgbm's defaults happen to suit this data. Note lgbm is
*worse* at 1000 rounds than at 200: it is not that more is better.

**So the terminal phase enumerates library defaults, and therefore measures the
libraries rather than the families.** Fixing that needs a common budget or early
stopping, which is a design and a record of its own — not a learning rate copied
off this table, which is the constant-fitting 0034 exists to refuse. Left open
deliberately.

## Defect 2: the servability guard was bypassed

Adding lgbm to the set changed which model won overall and broke an existing
test, which is how this surfaced — it had been latent since 0035 landed, with a
green suite over it.

`_refuse_if_unservable` sits behind `if enforce_stop:`, and the terminal phase
calls `train_tabular_model(enforce_stop=False)`. So [0030](../../../docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md)'s
guarantee was not being enforced where it now matters most. Worse, 0035 had
removed the only place the agent ever *heard* about it, because that refusal
lived in `train_tabular_model`. A `group_stats` table whose value column is later
dropped could win `best_table` and be packaged into a bundle that raises on the
first serving row.

The check moves into the probe, which since 0035 is the first thing to touch any
table and therefore the only place left to say so. An unservable table is left
**unscored** rather than scored: it cannot win `best_table`, it is not charged
against patience, and the agent is told immediately with the columns named.

The general lesson is about 0035 rather than about serving. Removing a tool
removes every guarantee that tool was carrying, and those guarantees do not
announce themselves — this one was found by a test about a different dataset,
three commits later.

## The registered claims

Written into the runner's docstring before anything was submitted.

**1. Kaggle at or above v2's shipped 0.26522.**

- **As the run actually shipped: failed.** 0.24907, below v2 by 0.016.
- **With the one-line fix: passed.** 0.27470, above v2 by 0.00948.

Both are recorded because only the first is what the harness did. The second is a
post-hoc repair and does not get to be the headline.

It also does not beat what v2 *had*: v2's discarded `e11` scored **0.27724**, so
the corrected v3 is still 0.0025 short of the best model the old harness produced
and threw away (0034).

**2. Optimism below v2's +0.0183.** Passed, narrowly: **+0.01596** for the
corrected model (val 0.29066 → 0.27470) against v2's e8 at **+0.01828** (val
0.28350 → 0.26522). A 0.002 improvement, which is well inside the noise of the
numbers being compared and should not be read as an effect.

**3. The terminal models' optimism within ~0.002 of each other.** **Failed, and
decisively.**

## 0038's hypothesis is rejected

0038 observed that four models across two house-prices tables clustered by table
rather than by model, and proposed that optimism attaches to the feature set a
search selected. This run tested it on one table, and:

| on `t3` | val gini | private gini | optimism |
|---|---|---|---|
| lgbm | 0.29066 | 0.27470 | **+0.01596** |
| elasticnet | 0.25389 | 0.24907 | **+0.00482** |

Same table, same validation rows, same 476,170 training rows. **A factor of 3.3,
against a predicted 0.002.** The hypothesis as stated is dead.

### What replaces it, tentatively

A pattern that fits three of the four pairs we now have, and is a better story
than either:

> Optimism attaches to the **selected measurement** — the model-and-table
> combination that won an argmax — not to the table, and not to the model.

| run | measurement | selected? | optimism |
|---|---|---|---|
| seguro v2 | e8 lgbm — the run's pick | **yes** | +0.01828 |
| seguro v2 | e11 lgbm — the runner-up | no | +0.00587 |
| seguro v3 | lgbm on `t3` — the probe, which chose `t3` | **yes** | +0.01596 |
| seguro v3 | elasticnet on `t3` | no | +0.00482 |

Both seguro runs show the selected measurement at **about 3x** the optimism of an
unselected one on the same data. It also explains house v9, where *neither*
shipped model was the probe — the probe chose `t10`, elasticnet and catboost were
fitted onto it afterwards, and both came in low (+0.00458, +0.00475).

**And there is a point it does not explain.** On house v8's `t11`, the table was
selected through a catboost-heavy search, yet the unselected elasticnet was
*more* optimistic (+0.01717) than the selected catboost (+0.01127). Recorded
rather than explained away.

So: five points, one contradiction, and a hypothesis that is sharper than 0038's
and has survived one test it could have failed. Not a finding.

## Smaller things this run established

**0033's elasticnet fix carried its first real run.** 3.0 minutes for the
terminal fit on 476,170 rows, against v1 dying at 63 minutes with ~510 hours
projected. Confirmed at scale, two records after it was written.

**The probe's fixed seed is load-bearing at scale.** lgbm at identical parameters
on `t3` scores 0.645331 at seed 0 and 0.642306 at seed 42 — exactly reproducible
per seed, threads irrelevant, but **0.003 of auc apart between seeds**. The whole
feature search gained 0.0044 of auc over the raw table, so the ruler's arbitrary
seed is worth a third of everything the agent achieved. On house-prices' 1168
rows the same check found bit-identical results, which is why
`notebooks/phase4_probe_and_feature_search.ipynb` says seed variation measures
nothing: true there, false here.

## What would settle any of it

0023, still unbuilt. Every optimism figure above cost a Kaggle submission, this
run spent four of them, and the competition is one of two datasets where that
oracle exists at all.
