# 0081 — On 18 historical Restaurant steps, a quarter of incumbent block spread changes no decision

- **Date:** 2026-08-21
- **Status:** accepted (a measurement). Resolves [the preregistration](PREREGISTRATION.md): the registered prediction is **refuted**
- **Measured by:** `analyze.py`, `contrasts.csv`, and `summary.json`
- **Related:** [0078](../../../../research/archive/decisions/0078-the-panel-was-a-stop-rule-and-the-gain-threshold-is-the-real-lever.md)
  (the zero ordered threshold), [0059](../../../../research/archive/decisions/0059-the-interval-belongs-to-the-difference-not-the-score.md)
  (the block instrument), [0079](../../../../research/archive/decisions/0079-the-work-table-stays-on-the-terminal-ballot.md)
  (work-aware versus historical derived incumbents).

## The registered question

The preregistration fixed the geometry and decision rule before reading the reconstructed
contrasts. For each historical step, compare its stored predictions with the best
earlier **derived** table — the incumbent used by the historical stop rule — and
ask whether

```text
gain > 0.25 × incumbent block_sd
```

predicts a candidate remaining best-so-far under an independently recomputed
high-capacity incumbent better than the ordered run's current `gain > 0` rule.
The decision question is the same; the pair is not always the same. The
prediction required both higher precision and at least 0.50 recall. `k = 0.5`
and `1.0` were registered as sensitivity readings that could not replace a
failed primary value.

High-capacity reruling outcomes exist for 18 steps: 16 from `restaurant_v1`, two from
`restaurant_v2`. IEEE-CIS has prediction sidecars but no high-budget reruling, so
its 23 incumbent contrasts are descriptive only.

## Controls passed

The archive contained journal, validation rows, and one prediction sidecar per
journalised table for all three runs.

| run | parent intervals reproduced | derived-incumbent contrasts | work-aware contrasts |
|---|---:|---:|---:|
| `ieee_v1` | 24/24 | 23 | 24 |
| `restaurant_v1` | 17/17 | 16 | 17 |
| `restaurant_v2` | 13/13 | 12 | 13 |
| **total** | **54/54** | **51** | **54** |

Every stored parent `gain`, `block_gains`, `block_sd`, sign flag, row count and
mode reproduced from the sidecars. Prediction lengths matched validation rows.
Score differences matched prediction differences up to the independently stored
six-decimal rounding. All 18 high-capacity lineage edges resolved to one relative
utility per table without conflicts. Those utilities are reconstructed from the
CSV's six-decimal edge gains, not from unrounded model scores.

The geometry correction mattered: parent and historical global incumbent differ
on **20 of 51** reconstructed steps, and on **7 of the 18** capacity-labelled
steps. Reusing candidate-vs-parent `block_sd` in the stop rule would therefore
have answered another question on a substantial share of the run.

The incumbent identity itself changes between low and high capacity on **6 of
18** labelled steps. The outcome therefore preserves the best-so-far decision
question, not one fixed pairwise contrast.

## The registered result is negative

Seven candidates remain best-so-far under the high-capacity reruling.

| rule | accepts | capacity-positive accepted | precision | recall |
|---|---:|---:|---:|---:|
| current: `gain > 0` | 11 | 7 | **0.636** | **1.000** |
| registered: `gain > 0.25 × block_sd` | 11 | 7 | **0.636** | **1.000** |
| sensitivity: `gain > 0.5 × block_sd` | 7 | 6 | 0.857 | 0.857 |
| sensitivity: `gain > 1.0 × block_sd` | 7 | 6 | 0.857 | 0.857 |

> **A quarter of block spread changes not one decision.** Precision does not
> increase, so 0080's prediction is refuted even though recall remains perfect.
> `0.25 × block_sd` must not replace the zero floor.

The four positive low-budget incumbent gains that are negative under the
high-capacity reruling have
ratios `gain / block_sd` of approximately **0.30, 1.18, 0.31 and 0.33**. The one
small capacity-positive outcome has a ratio of approximately **0.25**. There is
no clean separation in these 18 readings: the sensitivity value `0.5` removes
three capacity-negative cases but also removes that capacity-positive case,
while the capacity-negative case at `1.18` survives even `k = 1.0`.

Selecting `0.5` now would repeat 0034's failure — fitting a constant to the same
small set used to notice it — and violate 0080's explicit rule that sensitivity
values cannot win after the primary value fails.

## IEEE says the contrast is useful as a report, not a gate

Of IEEE's 23 derived-incumbent comparisons, the parent differs from the incumbent
on eight. Nine incumbent gains are positive; the quarter rule reports seven and
the half rule four. **Twenty of 23 have blockwise sign disagreement.** With no
high-capacity outcome these are not successes or failures, but they show that the
exact incumbent contrast carries information the parent result does not.

The safe conclusion is therefore asymmetric:

- persist and report candidate-vs-incumbent gain and its full block sequence;
- do not gate stopping on a `block_sd` multiple.

## The known block defect is maximal on Restaurant

The historical instrument cuts five equal-row blocks without moving boundaries
off tied ordering values. Both Restaurant validation halves contain only 18
distinct dates, and **all four of four internal block boundaries split a date**.
IEEE has 28,816 distinct ordering values among 29,527 rows and zero of four split
boundaries.

This means the Restaurant `block_sd` readings are not five clean period
contrasts. It weakens every positive interpretation, especially the tempting
`k = 0.5` sensitivity result. It does not rescue the registered quarter rule:
that rule already failed by making exactly the same decisions as zero.

## What this settles and what it does not

**Settled for now:** no tested `block_sd` multiple has prospective evidence
sufficient to gate stopping. `k = 0.25` failed its registered test; `k = 0.5`
and `1.0` remain exploratory sensitivity readings. This does not disprove every
possible block-based rule. The exact incumbent card is worth exposing; none of
these constants is evidenced for enforcement.

**Not settled:** what should replace the zero ordered threshold. Eighteen adaptive
steps from two runs of one dataset family are not independent replications, and
capacity is itself a second ruler rather than truth. This measurement does not
show that ruler defects cause the below-median external scores.

**Next before any repeated block experiment:** fix block boundaries so tied
ordering values cannot straddle blocks. A future dataset may then use the exact
incumbent contrast in shadow mode, with a new registered rule rather than a
constant selected from this result.
