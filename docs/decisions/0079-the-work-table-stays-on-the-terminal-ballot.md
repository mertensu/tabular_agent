# 0079 — Building a table does not remove the data as given from the terminal ballot

- **Date:** 2026-08-21
- **Status:** accepted and implemented (bug fix)
- **Related:** [0035](0035-the-search-is-over-features.md) (the terminal ballot),
  [0017](0017-credit-by-ablation.md) (one changed thing per comparison),
  [0044](0044-what-ships-is-the-average-not-the-argmax.md) (the blend that ships),
  `src/tabular_agent/harness/tools.py:_terminal_fits`.

## The defect

The ruler scores the data as given before it scores the first derived table. That
number is stored as `journal.work_probe`, separately from `journal.tables`,
because no tool call produced the work table.

Terminal selection ignored that separation. `_terminal_fits` called
`best_table(journal["tables"], metric)`, so as soon as one derived table existed,
the work table was absent from the ballot. If every feature step made the ruler's
score worse, finalization still had to choose the least-bad derived table.

A minimal reproduction pinned work-table logloss at `0.10` and the only derived
table at `0.20`. Every terminal family was fitted on the derived table. The
harness therefore contradicted its own question — which table scored best under
the fixed ruler — without reporting that it had done so.

## The decision

**The terminal ballot contains the work table followed by every scored derived
table.** The work candidate uses `work_probe.probe_score` and the existing
`data/work.parquet` path; derived candidates remain unchanged.

Putting the work candidate first is load-bearing. `best_table` resolves ties in
favour of the earlier candidate because it has the shorter lineage and therefore
fewer replay steps that can fail. The work table has an empty lineage, so it is
the earliest possible candidate and wins an exact tie.

Nothing about the terminal model ballot changes. The same fixed set of families
is fitted on whichever table wins, and the same equal-weight blend ships.

## What this does not fix

The stop rule still counts only derived table attempts. In particular, the first
derived table is treated as progress relative to an empty derived history even
when it is worse than `work_probe`. The agent-facing result also still reports a
parent comparison rather than the exact global decision card. Those are separate
signal defects; changing them here would make this bug fix silently alter search
length as well as terminal selection.

Old journals without `work_probe` retain the historical fallback: if no scored
work candidate exists, finalization chooses among derived tables as before. A
normal current run writes `work_probe` when its first child is scored.

## Verification

`test_finalize_ships_the_work_table_when_every_derived_table_is_worse` drives the
real `finalize_run` path. It builds a derived table, pins its score below the
stored work score, then asserts that every terminal experiment was fitted on
`data/work.parquet`. The test failed before the change with every family on
`tables/t1.parquet` and passes after it.

The existing no-derived-table test remains the degenerate case: a run that built
nothing already shipped the work table and continues to do so.

## Cost

One synthetic candidate dictionary at finalization and one comparison. No fit,
no new artifact, no change to the ruler, and no dataset-specific rule.
