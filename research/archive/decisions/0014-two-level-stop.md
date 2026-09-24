# 0014 — Exploration is cheap; tuning one table is not

- **Date:** 2026-07-30
- **Status:** accepted, with an open question (see Consequences)
- **Decided by:** Ulf
- **Revises:** [0012](0012-harness-owns-the-stop.md)
- **Commit:** `84dd8bb`

## The problem

Decision 0012 shipped with `patience=4`: four experiments with no real
improvement and the run ends.

Ulf's objection, immediately:

> we need to let it iterate and search (of course, in ML, you try a lot of things
> without success until you finally hit one)

He is right, and the rule as built punished exactly the behaviour we want.
Consider:

| exp | table | score |
|---|---|---|
| e2 | t1 | 0.40 |
| e3–e8 | t1 | 0.41–0.44 — six hyper-parameter settings, none better |
| e9 | **t2** | 0.31 |

With `patience=4` the harness refuses at e6 and never sees e9. Trying eight
things that fail in order to find the ninth that works is not a failing search —
it is the only kind of search there is.

## The insight: two runs, identical to the old rule, that are not alike

Now compare that run with a different one whose e3–e8 are **six different feature
tables** — a date parse, an encoding, a ratio, an interaction — all scoring
around 0.41.

Both have six experiments since the last improvement. `since_improvement` cannot
tell them apart. But:

- The second run tried **six ideas**. That is search.
- The first trained on **one table six times**. Experiments sharing a source
  table can only differ by model type and hyper-parameters — it is not six
  experiments, it is one experiment costed six times.

The distinguishing information was in the leaderboard all along: `source_table`.

This also matches the practitioner consensus on tabular work: feature engineering
and a trustworthy validation setup are where the movement is; hyper-parameter
tuning is worth a small fraction, does not compound, and is best done **once at
the end** on the feature set you have settled on rather than interleaved with the
search.

## The decision

**Two budgets, at different granularities and with opposite temperaments.**

- **`patience = 10`, run level, generous.** Ten experiments with no improvement
  ends the run. Deliberately loose: ending a run is expensive and irreversible.
- **`table_patience = 3`, per feature table, tight.** Three experiments on one
  table with no improvement and the harness **refuses that table** — not the run —
  and names `transform_features` / `fit_encoding` as the next move.

A refused table is a **redirect**. The run continues the moment the agent builds
new features, and costs nothing (0013 made refusals free). A refused run is over.
Only one of those deserves a hair trigger.

**This is also how "prefer feature work over hyper-parameter tuning" got
answered.** It had been an open question whether to add that sentence to the
system prompt. It does not need to be there: it is now a property of the tool,
where it cannot be ignored, forgotten, or argued with.

## Consequences

**Good**

- Long searches over genuinely different ideas are protected.
- Live behaviour changed shape: where the earlier run went straight to tuning,
  the run after this built six feature tables in sequence, climbing
  `0.688 → 0.645 → 0.474 → 0.472 → 0.464 → 0.403 → 0.316 → 0.274`.
- The structural refusal was **never needed** — the advisory alone moved the
  agent off an exhausted table (`refusals: 0`).

**Bad — and this is a real open question, not a footnote**

- **It is evadable.** Alternating between two dead tables (`t2, t4, t2, t4, …`)
  keeps both per-table counters under the limit while doing precisely the thing
  the rule targets. The original 11-experiment run did exactly this, so **the new
  rule would not have caught the run that motivated it.** Only the generous
  run-level patience catches it, several experiments later. Closing the loophole
  (counting across already-exhausted tables) would let a promising new table arrive
  already condemned by the history of others — every tightening buys one failure
  and sells another.
- **On the one dataset measured, the loosening bought nothing.** Three runs, same
  prompt and model:

  | rule | experiments | best |
  |---|---|---|
  | uncapped | 11 | 0.2718 |
  | `patience=4` | 8 | 0.2743 |
  | `patience=10` + table redirect | 19 | 0.2719 |

  All three found the same score (differences far below the 0.02 fold spread).
  The loose rule cost roughly double for no better answer. That is not an argument
  to revert — this dataset's signal is nearly all in one column, so there was never
  a ninth idea waiting to pay off, and it is exactly a dataset with more depth
  where the tight rule would have cost the answer. But it is unresolved, and it
  needs a **second dataset** before anyone claims the trade was right.
- A related tension is logged in `HANDOFF.md`: measuring a link costs a unit of
  patience, so an agent that measures carefully accumulates no-improvement rows by
  construction. Same trigger for revisiting.

## Where it lives in the code

- `src/tabular_agent/harness/stop.py` — `StopPolicy.table_patience`,
  `table_status`.
- `src/tabular_agent/harness/tools.py` — `_refuse_if_table_exhausted`.
- `src/tabular_agent/harness/instructions.py` — the advisory wording.
- `tests/test_stop.py` — a long search over new tables is not stopped; tuning one
  table exhausts that table and not the run.
