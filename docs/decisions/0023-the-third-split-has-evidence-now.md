# 0023 — The third split has evidence now

- **Date:** 2026-08-10
- **Status:** accepted, **not yet implemented**
- **Revises:** [0019](0019-the-leaderboard-is-a-public-leaderboard.md) — not its
  decision, which stands, but the deferral at the end of it
- **Related:** [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md)

## What 0019 left open

0019 chose two splits over three, deliberately, and named the price: the number
the run reports is the number the agent optimised against, and **nothing in a run
measures how much the search gamed it**. The third split was held in reserve
against evidence that the effect was large enough to pay for it.

Two Kaggle submissions are that evidence.

## The measurement

Same dataset, same model, same prompt. Only the run differs.

| | run 2 | run 3 |
|---|---|---|
| experiments ranked on the validation half | 17 | 24 |
| reported `val_score` | 0.12162 | 0.11354 |
| Kaggle (1459 unseen rows) | 0.12368 | 0.12298 |
| **optimism** | **+0.00206** | **+0.00944** |

**1.4x the experiments, 4.6x the optimism.** At 17 experiments the reported
number was honest to within its own noise; at 24 it overstated by nearly a
percentage point of rmsle — enough to invert the ranking between the two runs.
Our leaderboard said run 3 beat run 2 by 0.008; the truth was 0.0007.

The planned overnight run is **150 experiments**. Whatever the exact shape of
that curve, extrapolating from 17→24 to 150 says the reported number stops being
an estimate of generalisation and becomes a report of the luckiest draw — which
is precisely the failure 0019 described and priced as hypothetical.

## The decision

**Add a third split: a sealed holdout no tool can reach, scored exactly once, at
`finalize_run`, and reported beside — not instead of — `val_score`.**

Three rules that make it worth having, all of which follow from why the second
split stopped working:

1. **Nothing reads it during the run.** Not the leaderboard, not the stop rule,
   not `explain_table`. The moment a number is consulted repeatedly it becomes
   something to optimise against, which is the entire lesson here.
2. **`finalize_run` reads it once.** 0019's "finalize is genuinely once" work was
   run integrity when it was written; with a sealed split it becomes an actual
   leakage guard, exactly as that entry predicted.
3. **The report prints both, and the gap.** The gap *is* the measurement of how
   much the search gamed the validation half — the number no run has ever had.
   Reporting only the sealed score would throw away the most interesting thing it
   produces.

This is a distinct fix from 0022, and both are needed. 0022 lets a run tell a
real gain from a lucky one **while it is running**. The third split tells the
*reader afterwards* how much to trust the headline. Neither substitutes for the
other, and 0022 is much cheaper.

## What was rejected

- **Continuing to defer.** The reason for deferring was the absence of evidence.
  There is evidence.
- **Replacing `val_score` ranking with the sealed split.** Then it is the
  optimised number and we have three splits with two splits' worth of protection.
- **A bigger validation half instead.** 0022 covers why: `1/sqrt(n)`, paid for
  out of the work half, and it does not address reporting a maximum at all.
- **Reporting only the sealed score.** Loses the gap, which is the diagnostic.

## What it costs

**Good**

- The report can finally answer "how much of this is real", with a number.
- The cost falls entirely on rows, not on complexity: the machinery
  (`_ensure_split`, `_score_on_val`, lineage replay) all exists and generalises.

**Bad**

- **A third slice of a small dataset.** On house-prices, 1460 rows already split
  1168/292; a third split takes more from the half the agent builds on. On the
  120-row datasets `MIN_SPLIT_ROWS` was lowered for, three splits are not viable
  at all — this has to be optional, and off by default below some size.
- **Not implemented.** As with 0022, this record is a decision and a measurement,
  not a change. Runs made before it lands still report an optimised number.
- It does not make the *search* better, only the *reporting* honest. The agent
  will still chase noise on the validation half; 0022 is what addresses that.
- A single sealed score on a few hundred rows has its own standard error — the
  same objection 0022 raises. It is a better number, not a precise one.

## Where it will live

- `src/tabular_agent/harness/session.py` — a three-way `_ensure_split`, opt-in.
- `src/tabular_agent/harness/tools.py` — `finalize_run` scores it once;
  `resolve_table`'s allowlist must keep excluding it.
- `src/tabular_agent/report.py` — `build_report` already takes a `holdout_score`
  parameter that no caller has ever passed. This is what it was for.
