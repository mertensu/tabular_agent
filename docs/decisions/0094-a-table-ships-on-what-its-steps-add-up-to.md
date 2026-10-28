# 0094 — A table ships on what its steps add up to, not on its last step

- **Date:** 2026-10-28
- **Status:** accepted and implemented (`stop.eligible_ancestor`,
  `stop.best_table`, `tools._vs_eligible_ancestor`). Tests in `tests/test_stop.py`
  and `tests/test_harness_tools.py`.
- **Decided by:** Ulf
- **Supersedes:** [0090](../../research/archive/decisions/0090-the-terminal-table-must-have-a-demonstrated-gain.md)
  — its noise bar and its fallback stand; what a table is compared with changes.
- **Evidence:** `research/capacity_ship_minimal/` (README, "Rossmann r11" and the
  Kaggle sections; `selection_check.py`).

## The problem

A run builds tables step by step: each new table is an earlier one plus a few
columns. At the end, one table is chosen for the final models to train on. 0090
made that choice safe against a ratchet: the agent keeps adding steps that each
move the validation score a hair in the right direction, and the best-scoring
table drifts into the noise. 0090's rule is that a table may only be chosen if
its gain over **the table it was built from** is larger than that gain's spread
across consecutive validation periods (`block_sd`; on a random split, the paired
bootstrap's standard error).

Comparing with the immediate source only has the opposite failure. Take a chain
t8 → t9 → t15 → t20 where each step improves the validation score by less than
its own spread, while the whole chain improves it by far more. Under 0090 none of
t9, t15 or t20 can ship, because no single step was demonstrated, and the run
ships t8. In Rossmann r11 the probe score went from 0.1426 (t8) to 0.1264 (t20) in
steps that each failed 0090's bar, and t8 shipped.

Lengthening validation does not help: the bar is the spread of the gain *between
periods*, which more periods estimate better but do not shrink (r11 validated on
96 days instead of 48; the spreads were the same size).

## The decision

**A table is eligible to ship when its gain over its nearest eligible ancestor is
larger than that comparison's noise bar.** The ancestor is found by walking the
table's lineage back from its source to the first table that is itself eligible.
While no table on the path is eligible, it is the immediate source, so the first
step out of the raw rows is judged exactly as before. The terminal table is the
best probe score among eligible tables; if there is none, the raw work table
ships, as under 0090.

What 0090 protects is kept: a table one noise-level step past an eligible table
is compared with that eligible table, which is its own source, and fails. What
changes is that steps which are each inside the noise can ship once, together,
they are not.

- The comparison is computed when a table is scored, from the stored validation
  predictions of both tables (no extra fit), and recorded on the table's journal
  row beside the comparison with its source. Eligibility depends only on a
  table's own row and its ancestors are scored before it, so the choice does not
  depend on the order the agent built things in or on what it builds later.
- The agent is not shown it. Search guidance stays the ungated probe score, as
  0090 kept it; this is the terminal choice only.
- Journal rows written before this record have no ancestor comparison and are
  judged as 0090 judged them.
- The minimal profile, which has no noise bar, is unaffected.

## The evidence

Applied offline to the stored validation predictions of seven Rossmann runs, and
both picks scored by the run's ruler on the experiment's own test days, then on
Kaggle with the same recipe (7-model blend, same weights, refitted on all rows):

| run | 0090 ships | 0094 ships | Kaggle private, 0090 → 0094 |
|---|---|---|---|
| r4 | t5 | t16 | 0.1410 → **0.1290** |
| r7 | t10 | t18 | 0.1364 → **0.1321** |
| r10 | t14 | t21 | 0.1310 → **0.1293** |
| r11 | t8 | t20 | 0.1316 → **0.1270** |

In r1 it picks an *earlier* table than 0090 (tied on the test days); in r3 and r8
the picks agree. It is not "ship the longest chain": in r11 the best probe score
on validation (t24) scored 0.1318 on Kaggle, worse than both picks.

## What it costs

- **One dataset.** All of the evidence is Rossmann. The rule is argued on its own
  terms above; the evidence says it did not hurt there and helped in every pair.
- **Longer chains are trusted more easily than under 0090.** A chain picks up
  selection optimism on the validation rows the agent searched on, and the bar
  does not grow with the number of steps since the last eligible table. If a
  later run shows a long chain shipping and losing on held-out rows, that is the
  first thing to tighten.
- **One more comparison per table**, on stored predictions: a few milliseconds.

## Rejected

- **Ship the best probe score, no bar.** That is the ratchet 0090 was written
  against; r11's t24 is a current example.
- **Compare every table with the raw work table.** Once one early step has a large
  gain, every descendant clears the bar against the raw rows, including a final
  noise step. That brings the ratchet back.
- **A longer validation period.** Measured in r11: it does not shrink a spread
  that comes from period-to-period variation.
