# 0080 — The incumbent block threshold gets one prospective test

- **Date:** 2026-08-21
- **Status:** registered prediction, unresolved — written before computing any
  candidate-vs-incumbent block contrast from `artifacts.tar.gz`
- **Related:** [0078](../../../../docs/decisions/0078-the-panel-was-a-stop-rule-and-the-gain-threshold-is-the-real-lever.md)
  (the zero threshold on ordered runs), [0059](../../../../docs/decisions/0059-the-interval-belongs-to-the-difference-not-the-score.md)
  (the block instrument), [0079](../../../../docs/decisions/0079-the-work-table-stays-on-the-terminal-ballot.md)
  (why raw and derived incumbents are different questions),
  [`ruler_deflation_across_runs.csv`](../../readings/ruler_deflation_across_runs.csv).

## The question

On ordered runs `probe_se` is deliberately absent, so the stop threshold is zero:
every positive gain resets patience. `blockwise_difference()` already measures how
a difference varies over consecutive periods, but the stored interval is
candidate versus its parent while the stop rule compares a candidate with the
best earlier derived table. On a branch those are not the same contrast.

The supplied archive makes the exact low-budget incumbent contrast recoverable
without refits for `restaurant_v1`, `restaurant_v2`, and `ieee_v1`: journal order,
validation targets/order, and one prediction sidecar per table all survive.
Capacity outcomes survive only for 18 Restaurant steps in
`../../readings/ruler_deflation_across_runs.csv`. IEEE is therefore descriptive and
cannot decide the threshold.

## Geometry fixed before looking

For this test, **incumbent means the best earlier derived table**, matching
`stop._since_improvement` as it behaved in these historical runs. The work table
is reported as a second, terminal-consistent contrast but cannot be used for the
capacity verdict because the high-budget CSV has no raw-work anchor. `t1` has no
earlier derived incumbent and is excluded from the threshold comparison.

At high capacity, each edge's `gain_high` is accumulated through the feature
lineage, with `t1 = 0`, to recover every table's relative utility. The high-budget
outcome for a step asks the same best-so-far question as at low budget: did the
candidate beat the independently recomputed best earlier derived table? The
incumbent itself may therefore differ between budgets; the decision question is
held fixed, not necessarily the pair of tables.

## Registered rule and prediction

The primary candidate is fixed by the library's existing constant:

```text
low incumbent gain > 0.25 × incumbent block_sd
```

It is compared with the current ordered-run rule:

```text
low incumbent gain > 0
```

**Prediction:** the quarter-block rule increases precision for candidates that
remain best-so-far under the high-capacity reruling and retains at least half of
those capacity-positive outcomes.
Both conditions are required. If precision does not increase, or recall falls
below 0.50, the rule is refuted and must not gate stopping.

`k = 0.5` and `k = 1.0` are reported only as sensitivity readings. They cannot
win this test if `k = 0.25` fails. No value of `k` is selected from the result.

## Controls required before the verdict

1. Recomputed candidate-vs-parent `gain`, `block_gains`, and `block_sd` match the
   nested journal record for every covered step.
2. Recomputed candidate-vs-incumbent gain matches the scalar score difference in
   the metric's correct direction.
3. Prediction vectors have exactly one row per validation target.
4. Every capacity edge resolves to one high-budget relative utility without a
   cycle or conflicting duplicate.

A failed control makes the test unavailable, not negative.

## Limits registered in advance

Eighteen adaptive steps from two runs of one dataset family are not eighteen
independent replications. This test can reject the proposed gate; a pass is only
permission for a prospective shadow run on a new forecasting dataset, not for
wiring the threshold into `stop.py`. `block_sd` is temporal heterogeneity, not a
standard error and not proof of transfer beyond the measured horizon.

The historical implementation also cuts equal-sized row blocks without moving a
boundary off tied ordering values. Restaurant has many stores per date, so one
date can straddle two nominal blocks. This test deliberately reproduces the
stored instrument; even a positive result cannot license a gate until the known
tie-boundary defect is fixed and the reading is repeated.
