# 0071 — The gap was not the vocabulary; it was a column pair nobody asked for

- **Date:** 2026-08-17
- **Status:** accepted (a measurement; the fix it implies is **not built**)
- **Answers:** [0070](0070-what-the-winner-had-that-we-cannot-express.md)'s closing
  recommendation, which refused to rank the missing capabilities and asked for this
  measurement instead
- **Related:** [0049](0049-the-last-observation-is-a-sample-of-size-one.md) and
  [0050](0050-the-anchor-is-what-matters-not-the-history.md) (which measured the
  direction of this effect and never its price on a table worth having),
  [0030](0030-serving-columns.md) (the servability rule behind the `Customers`
  finding), [0011](0011-stability-check.md) (the guard that did not fire),
  [0044](0044-what-ships-is-the-average-not-the-argmax.md),
  [0068](0068-capacity-reaches-the-shipped-blend-at-half-strength-and-validation-picks-the-wrong-budget.md)
  (the budget these arms are fitted at, and the probe→sealed halving seen again
  here), [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (the
  interval that makes this the first solid reading in three sessions),
  `scripts/probe_winner_salesmodel.py`, `_runs/winner_salesmodel/`.

## The question, in Ulf's words

> *"ich will nur vermeiden eine echte Lücke zu haben in unserer harness/agent. wenn
> der agent alles könnte aber es nicht schafft ist es eine andere sache die man
> klären muss aber bislang habe ich das gefühl iwas fehlt"*

Two problems wear the same symptom of being mid-table, and they need opposite fixes:

* a **capability gap** — the harness cannot express a feature, so no run could ever
  have built it;
* an **agent gap** — it can, and the agent did not.

This project had been guessing which one it had. 0069 and 0070 are both records of
a lever sized from a description and then measured much smaller. So this measured.

## What was done

`scripts/probe_winner_salesmodel.py` rebuilds the Rossmann winner's **salesmodel**
— the 24 features carrying bit 0 of his feature dict, parsed from the writeup
rather than transcribed — using **real tool calls**, on v7's own copied split, at
0068's `1000 @ lr 0.10`, scored on v7's sealed holdout.

Classified before anything was fitted: **19 of 24 buildable with existing tools**,
3 a genuine capability gap (`holidays_last/this/nextweek`), 2 excluded by decision
(weather, 0001).

## The answer, and it is neither of the two

| | value |
|---|---|
| **capability gap** — the three calendar windows, probe, same step both arms | **+0.00122** |
| **agent gap** — his salesmodel against v7's `t13`, sealed | **+0.00220** |
| **`fit_history`'s unrequested `_latest` pair** — probe | **+0.01428** |

> **The tool defect is six times the agent gap and twelve times the capability gap.**

`fit_history` emits four columns where the winner's feature is one: the window
statistic, its count, **and an unrequested `_latest` / `_latest_ago` pair** — the
last known value and its age. There is no argument to decline them. Dropping only
those four columns from the finished table:

| | probe |
|---|---|
| with the pair | 0.14098 |
| *v7's `t13`, what the agent built* | *0.13202* |
| **without the pair** | **0.12670** |

and 0059's interval makes this the first solid reading in three sessions:

    block gains  [+0.0150, +0.0142, +0.0128, +0.0072, +0.0249]
    block sd 0.00576   sign_stable True   5/5 blocks favour it

**2.5 block standard deviations, every block agreeing.** For contrast, 0069's
feature-selection readings were both `sign_stable: False` and inside their noise.
Sealed, the effect is roughly +0.008 — the same probe→sealed halving 0068 measured
for capacity.

This is not new *information*: 0049 measured the last observation as a sample of
size one, 0050 found the anchor mattered more than the history, and `HANDOFF.md`
has carried *"`_latest` still emitted unconditionally"* as an open item for weeks.
What is new is the **price, on a table that is worth something**. Every earlier
reading was on tables where the pair was one nuisance among many.

## Two findings nobody was looking for

**1. The stability check does not catch a whole class of batch-dependent column.**

The naive implementation of the holiday windows — roll a per-date flag over seven
days — was **accepted**, and it should not have been. Its value for one row is
computed from other rows' dates, so a served batch of one date produces different
numbers than training did.

The mechanism is exact. `transform.py` tests stability by re-running the code on
`df.sample(n // 4)` — a random sample of **rows**. Rossmann has 1,115 stores per
date, so the probability that a sample of 25% drops any date entirely is
`0.75 ** 1115`, i.e. zero. The set of distinct dates is preserved, `groupby(Date)`
returns the same index, the rolling window returns the same numbers, and the check
sees perfect agreement.

> **The check tests row-locality, not batch-independence.** A column that is a
> function of the *set of distinct values* of a key column survives row sampling
> whenever there are many rows per key — which for panel data is the normal case,
> not the corner case.

The cases 0011 was built for (`pd.qcut`, batch normalisation) depend on the row
multiset and are still caught. This class is not, and the failure is silent.

**2. `Customers` is not thrown away out of inability. The harness rewards throwing
it away.**

`fit_group_stats`'s docstring says the value column need not exist at predict time,
only the `by` columns — so a per-store aggregate of `Customers` is perfectly
serveable, and 0070 flagged that no run has ever built one. The reason is order.
0030 refuses to score *any* table still carrying a column absent from `test.csv`,
so every step before the drop comes back `unscored`. Dropping `Customers` is
therefore the agent's rational first move — and doing it first destroys the only
thing it was good for.

> The agent is optimising correctly against what it can see. The penalty for
> keeping the column is immediate and visible; the reward for aggregating it first
> arrives two steps later.

## What this says about the distance to the winner

His single un-ensembled model scored ~0.105 private, rank ~3 of 3,300. A faithful
rebuild of that model's feature set, with our tools, at a real budget, lands at
**0.13024 sealed** — 0.0022 better than what our agent found by itself.

> **The distance to him is not in the feature vocabulary.** It is in what 0070
> called architecture: four model families, segment models trained on the test
> months, `colsample_bytree=0.3`, 5,000 rounds, an ensemble over feature subsets.

That was the outcome this script's docstring registered as expected and as "the
most useful negative result available", and it is what came back.

## What is decided, and what is not

**Decided: nothing in the library, yet.** The obvious fix — let `fit_history` omit
the pair — is a few lines, and it is not made here. This is one dataset, one
ablation, n=1, and the record this project keeps breaking is the one about acting
on a single reading. Restaurant's `t7` is on disk and can carry the same ablation
in an hour.

**Not decided: the stability-check hole.** Widening the check to resample *keys*
rather than rows would catch this class, and would also change what every existing
transform is allowed to do. That needs its own record and probably its own
measurement of what it would newly refuse.

**Not decided: whether `Customers` is worth having.** The rebuild built the
aggregate and never isolated its contribution. What is established is only that
the harness's incentives, not its capabilities, are why no run has one.

## What this cost

The rebuild was wrong twice before it was right, and both mistakes are worth
keeping. `fit_encoding(["Store", "Promo"], method="target")` encodes the two
columns *separately*; his `shopsales_promo` is the interaction, and building the
key first turned +0.01663 into **+0.03013**. v7's agent had got that right and the
hand rebuild had not — the error made the rebuild weaker and so flattered the
agent, which is the direction that would have gone unnoticed.

And the first run printed a "capability gap" of −0.00786 that is not one:
`finalize_run` ships the argmax of the probe scores, which in arm A is the ablated
table and in arm B is not, so the sealed difference between the arms is mostly the
`_latest` pair. The clean comparison is the probe at the same step of both arms.
Corrected in the script's output rather than left to be misread.
