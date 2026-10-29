# 0097 — A run is bounded by its budget, not by a streak without improvement

- **Date:** 2026-10-28
- **Status:** accepted and implemented (`stop.StopPolicy`: `patience=None`,
  `max_tables=60`, `max_minutes=30`; `stop.budget_line`; `journal.mark_started`).
  Tests in `tests/test_stop.py`.
- **Decided by:** Ulf
- **Revises:** [0035](0035-the-search-is-over-features.md) — its "one patience, over
  tables". The rest of 0035, and what counts as an improvement
  ([0012](../../research/archive/decisions/0012-harness-owns-the-stop.md),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md)), stand.
- **Evidence:** `research/capacity_ship_minimal/` (README, "Restaurant r9" and
  "Restaurant r10").

## The problem

Since 0035 a run ended after six tables in a row that did not beat the best table
so far. Six was set when a unit of work was an experiment of several fits. A table
is now one fit of about ten seconds, and the agent builds several per turn, so six
tables are about a minute of search. Restaurant r9 and r10 both ended exactly
there, at 12 and 17 tables and 3–4 minutes, with no table refused in r10: the agent
finished at the moment the rule would have stopped it.

"Beat the best so far" also counts a new branch built on an earlier table as a
failure until it overtakes the leader, so a run that explores is the one that runs
out of patience. And the agent never saw the count: it was attached to training
results, which the current agent does not produce.

## The decision

**A run is bounded by a budget of tables and minutes. When it is done within that
budget is the agent's call.**

- `max_tables=60` (was 30) and `max_minutes=30`, counted from the start of the
  agent's search, recorded once in the journal so a resumed process keeps the same
  clock. Past either, building is refused and the run is finalized, as before.
- The streak rule stays available (`patience=N`) and is off by default.
- Every table result carries the budget as a fact: tables built, tables left,
  minutes left. No advice comes with it.

## What it costs

- **More search against the same validation rows.** The shipped table's score
  becomes more optimistic the more tables compete for it. The shipping noise bar
  (0094) and an experiment's own test days are what measure that.
- **Runs take longer and cost more**, up to 30 minutes of model calls.
- **The agent may still stop early.** Budget is permission, not instruction; if
  runs keep finishing in a few minutes, the stop rule was not what limited them.

## Rejected

- **A longer patience** (twelve instead of six). Same rule in the same unit,
  ending a run on a count that says nothing about whether the agent has ideas left.
- **Patience against each table's parent** rather than the best so far. Fairer to
  branches, but still a rule ending runs the agent has not finished.
- **Only a table cap.** A cap alone cannot bound cost when a table can take minutes,
  as r9's two slow stream builds did.
