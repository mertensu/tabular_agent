# 0040 — Seguro v4: a near-copy of the ruler is not the ruler

- **Date:** 2026-08-13
- **Status:** accepted (a measurement)
- **Related:** [0039](0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md)
  (the run this repeats, and the hypothesis this tests a second time),
  [0035](../../../docs/decisions/0035-the-search-is-over-features.md), [0034](0034-the-tiebreak-was-tested-and-rejected.md)

## Why there is a v4

v3 was not a clean test of anything: it shipped a model 0.026 of gini below what
its own ruler had already achieved on the chosen table, and its corrected number
came from a repair script rather than from a run. Same dataset, same prompt,
same model, same control (v2), with 0039's two fixes in.

## The result

| | val gini | Kaggle private | optimism | rank /5157 |
|---|---|---|---|---|
| **v4 shipped — lgbm** | 0.28508 | **0.27670** | **+0.00838** | 2963, top 57% |
| v4 — elasticnet on the same table | 0.26016 | 0.25639 | +0.00377 | — |
| v2 shipped — e8 | 0.28350 | 0.26522 | +0.01828 | 3704, top 72% |
| v2 discarded — e11 | 0.28311 | 0.27724 | +0.00587 | 2918, top 57% |

**Claim 1 — Kaggle at or above v2's shipped 0.26522, unaided: passed**, by
0.01148, with no hand repair. v3 failed this.

**Claim 3 — optimism below v2's +0.01828: passed**, at **+0.00838**, less than
half. Combined with house v9 (+0.01127 → +0.00458) that is two datasets, two task
types and two metrics all moving the same way.

**Claim 4 — the probe's model is the more optimistic one: passed.** Below.

**Claim 2 — the shipped model scores at least the winning table's `probe_score`:
failed**, by 0.00279 of gini, and it is the most instructive part of the run.

**And the fix that prompted v4 is doing real work.** Under v3's terminal set this
run would have shipped elasticnet at 0.26026; it shipped lgbm at 0.28508 —
**0.0248 of gini recovered on val** by putting the ruler on the ballot.

Worth noting where that lands: v4's 0.27670 is within 0.0005 of v2's *discarded*
e11 at 0.27724. The old harness had that model and threw it away on a 0.00020 val
margin (0034). The new harness now delivers, by construction, what the old one
could only have delivered by luck.

## The defect: model type is not a fit

0039 claimed the invariant — *the shipped model is never worse than the number the
search was ranked on* — would hold **by construction** once `PROBE_MODEL_TYPE`
was on the terminal ballot. That claim was wrong, and v4 falsified it in the one
place it could.

`train_tabular_model` fits with the run's seed (42). The probe fits with its own
module constant (0). Same library, same parameters, different seed — and 0039
itself records that lgbm moves **0.003 of auc between seeds** on this dataset.
The shortfall was 0.00279 of gini, which is exactly that size.

So the terminal candidate was a *near-copy* of the ruler, and a near-copy does not
carry the guarantee. The invariant needs the probe's model type, **its parameters
and its seed** — the fit itself, not the family it belongs to.

`_terminal_fits` now passes `PROBE_PARAMS` and `PROBE_SEED` for that candidate,
which required `train_tabular_model` to take an explicit model seed. The folds are
untouched and still come from `spec.seed`: which rows are held out is the
harness's exam ([0029](0029-the-harness-owns-the-exam.md)) and must not move
between fits.

The test asserts **equality** with the probe's recorded score rather than
membership in a list, because membership is what was already true and false.

This is the same error twice in two days, at two grains: first the family was
missing from the ballot, then the configuration was. Both times the reasoning was
"the right kind of model is available", and both times the number that had been
*measured* was not the number that got *shipped*. The general form worth carrying:
**an invariant about a specific measurement cannot be enforced by reproducing the
category that measurement belonged to.**

A footnote on process: writing the equality test surfaced that `probe_score` is
stored rounded to six decimals, so the assertion is at storage precision. The fit
is bit-identical.

## 0039's hypothesis, tested again

0039 rejected 0038's "optimism belongs to the table" and proposed instead:

> Optimism attaches to the **selected measurement** — the model-and-table
> combination that won an argmax.

v4 was registered to test it: the probe chose `t8`, so the probe's own model
should overstate more there than a model that took no part in choosing it.

| on `t8` | optimism |
|---|---|
| lgbm — the probe's family, which selected `t8` | **+0.00838** |
| elasticnet — no part in the selection | **+0.00377** |

**2.2x, in the predicted direction.** The tally across everything we have:

| pair | selected | unselected | ratio |
|---|---|---|---|
| seguro v2 | +0.01828 | +0.00587 | 3.1x |
| seguro v3 | +0.01596 | +0.00482 | 3.3x |
| seguro v4 | +0.00838 | +0.00377 | 2.2x |
| house v9 | *neither model was the probe* | +0.00458, +0.00475 | consistent |
| house v8 | catboost +0.01127 | elasticnet **+0.01717** | **contradicts** |

Three direct pairs, all in the same direction, at 2.2–3.3x; one run consistent by
construction; one contradiction still unexplained.

**The confound is not resolved and was named in the claim before the run.** In
every direct pair the selected model is lgbm and the unselected one is
elasticnet, so "selection inflates a score" and "lgbm's score inflates more than a
linear model's" predict the same thing here. Separating them needs a run where the
probe is not the family that ends up selected, or a probe of a different family.
Until then this is a pattern with a rival explanation, not a finding.

## What it cost to learn

Six Kaggle submissions across v3 and v4, on one of the two datasets where that
oracle exists at all, to measure four optimism figures. That is the argument for
[0023](0023-the-third-split-has-evidence-now.md) restated as a bill.

There is a second, quieter argument in this run. v3 and v4 are the same harness on
the same data with the same prompt, and the agent searched differently: 8 tables
against 11, no `fit_group_stats` in v4's first six calls, best table +0.0087 of
gini in v3 against +0.0060 in v4. Run-to-run variation in the search is a
confound on every A/B in this project, and a free honest number per run would let
us run a configuration three times and report a spread instead of treating one
run as the measurement.
