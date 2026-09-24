# 0068 — Capacity reaches the shipped blend at half strength, and validation picks the wrong budget

- **Date:** 2026-08-16
- **Status:** accepted (a measurement)
- **Extends:** [0067](0067-capacity-is-worth-a-tenth-of-that-on-a-table-that-has-features.md) —
  whose reading stands; this measures the same knob on what actually ships
- **Related:** [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
  (where the budget question was opened and its item 2 left the design open),
  [0039](0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md) (which first
  named the missing common budget),
  [0044](../../../docs/decisions/0044-what-ships-is-the-average-not-the-argmax.md) (the blend this
  measures), [0023](0023-the-third-split-has-evidence-now.md) /
  [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md) (the sealed
  rows that produced the finding),
  [0056](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
  and [0060](0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md) (rulers
  that ranked confidently and wrongly), `scripts/probe_capacity_terminal_blend.py`,
  `_runs/capacity_terminal_blend.csv`.

## Why this was run

0067 measured capacity on Rossmann's `t13` — the chain that shipped — at
**+0.0068**, and the `HANDOFF.md` plan built its arithmetic on that number: if it
transferred, v7's private 0.13813 would become 0.13133 against a private median of
0.12967, closing about 80% of the gap.

But +0.0068 is a **probe** reading: one lgbm, scored on the validation half. What
ships is `finalize_run`'s output — four families fitted on the winning table and
averaged equal-weight (0044), scored on the sealed holdout. And every capacity
number this project has ever produced is a probe reading. `models.py` sets
`n_estimators=200` for lgbm, xgb *and* catboost; `PROBE_PARAMS` does the same. **The
models that actually serve have never been run at a real budget on any dataset.**

## What was run

`scripts/probe_capacity_terminal_blend.py`. Each arm is a fresh run directory
holding a byte-for-byte copy of the source run's split plus the one shipped table,
driven through the real `finalize_run`. The budget is applied identically to all
three boosted families — 0039's "common budget" — and layered *on top of* the
params `_terminal_fits` passes, because the lgbm candidate is pinned to
`n_estimators=200` by `PROBE_PARAMS` and patching only the library default would
have moved xgb and catboost while leaving the blend's lgbm member behind.

Two budgets, both reused from 0067's grid rather than freshly chosen, so the
readings are comparable to the single-lgbm ones they are read against.

**Four controls, all passed.** Both 200-tree arms reproduce their source journals
to six decimals (Rossmann v7's four models, Restaurant v2's four plus its blend);
`elasticnet` has no budget and did not move in any arm; and the single-lgbm sweep
replayed `_runs/capacity_v7_chain.csv` digit-for-digit, which is also a standing
check on 0055's replayability problem.

## The reading

| Rossmann `t13` | blend val | gain | blend sealed | gain |
|---|---|---|---|---|
| current 200 @ lr 0.10 | 0.13086 | — | 0.13618 | — |
| 1000 @ lr 0.10 | 0.12926 | +0.00160 | **0.13245** | **+0.00373** |
| 5000 @ lr 0.03 | **0.12779** | **+0.00307** | 0.13429 | +0.00189 |

| Restaurant `t7` | blend val | gain | blend sealed | gain |
|---|---|---|---|---|
| current 200 @ lr 0.10 | 0.51971 | — | 0.50571 | — |
| 1000 @ lr 0.10 | 0.52088 | −0.00117 | 0.50566 | +0.00005 |
| 5000 @ lr 0.03 | 0.52023 | −0.00053 | 0.50375 | +0.00197 |

## 1. The blend inherits part of the gain, not all of it

Rossmann: the single lgbm gains **+0.00682** on the probe, the blend **+0.00373**
sealed. Restaurant: +0.00186 against +0.00197, both of them nothing.

The prediction registered before the run said the blend would gain *less* than the
single fit, at +0.003 to +0.006, because an equal-weight average of four models
already buys some of what capacity buys — variance reduction. That held on
Rossmann. It was **refuted on Restaurant**, where the prediction was that a
string-keyed table with 829 shop effects would gain *more* than Rossmann; it gains
nothing.

> **Capacity closes about 44% of Rossmann's gap to the median, not 80%.** Carrying
> the sealed gain across, v7's private 0.13813 becomes roughly **0.13440** against
> a median of 0.12967 — still short by 0.0047.

## 2. Validation picks the wrong budget, on both datasets

This is the finding that outlives the numbers.

- **Rossmann:** `val` ranks `5000@0.03` best (0.12779). The sealed rows rank
  `1000@0.10` best, and `5000@0.03` delivers **less than half** the sealed gain.
- **Restaurant:** every budget makes the blend's `val` score *worse* and its sealed
  score *better*. The two rulers disagree in **sign** about whether to spend
  capacity at all.

0054 left open where the budget comes from and observed that early stopping "wins
or ties on all five readings". The obvious designs that follow — early-stop on the
validation block, or grid-search the budget on `val` — are now refuted before
being built, because the ruler they would use is measurably wrong here. Two further
readings agree: 0067's inner-tail early stop chose 1195 rounds on `t13` for
+0.0028 against a fixed 5000@0.03's +0.0068, and the same arm on Restaurant's `t7`
chose 1498 rounds for **−0.0002** — worse than doing nothing.

> **The budget cannot be chosen by the run's own ruler.** Whatever answers 0054's
> open item 2, it is not "let validation decide", and it is not early stopping in
> the form measured so far.

## 3. A common budget is the wrong shape, because the families disagree

Sealed scores, 200 trees → `5000@0.03`:

| | Rossmann | Restaurant |
|---|---|---|
| catboost | 0.14113 → 0.13370 → **0.13214** | 0.50520 → 0.50055 → **0.50007** |
| lgbm | 0.14145 → **0.13689** → 0.14181 | **0.50670** → 0.51057 → 0.50732 |
| xgb | 0.15253 → **0.14011** → 0.14483 | 0.51987 → 0.51520 → **0.51381** |

**catboost improves monotonically on both datasets; lgbm and xgb both turn over and
end worse at the largest budget on both.** 0039 said enumerating library defaults
measures the libraries rather than the families; this says the repair is not one
common number either. Note also that lgbm is the family the *probe* is made of,
which is why the probe reads Restaurant backwards: its best single-lgbm budget is
`1000@0.03` at +0.00186, and `5000@0.03` costs it **−0.00879**.

## What is settled, and what is not

**Settled:** capacity is real on the serving path but half the size the probe
suggested, and it is not sufficient. 0067's conclusion — that features and capacity
together do not close Rossmann's gap — is strengthened.

**Settled:** the design space for 0054 item 2 is smaller than it was. Anything
that selects a budget using the validation half is refuted by measurement, not by
argument.

**Not settled:** where the budget does come from. Nothing here proposes an answer,
and the honest position is that the three cheap candidates (a constant, early
stopping on an inner tail, selection on `val`) are all now measured and all
inadequate.

**Not decided, deliberately: nothing was changed in the library.** These are two
tables and two budgets. Raising `n_estimators` on the strength of them would be
0054's rule broken with better arithmetic.

## What this cost

About three hours of fits and no submissions — both boards are spent (0045, 0066).
The prediction that Restaurant would gain more than Rossmann was argued from 0054's
per-entity-identity mechanism and was simply wrong; the mechanism holds for a *bare*
table with a store code added, and `t7` already carries per-store structure through
the encoders its agent built. That is 0067's substitution finding arriving a second
time, and it was not anticipated even though 0067 was written the same day.
