# 0003 — Tools take file paths, not DataFrames

- **Date:** 2026-07-28
- **Status:** accepted
- **Decided by:** Ulf (from three options put to him)
- **Commit:** `c6da944`

## The problem

The plan going into this session said: "register the existing library functions
as the agent's tools". That turned out not to be literally possible.

The library was built for humans. Every function takes a `DataFrame` and returns
a `DataFrame` — `fit_encoding(df, columns, method)`, `train_cv(X, y, folds, ...)`.
An LLM cannot pass a DataFrame. It emits JSON. So something has to sit between
the model's JSON arguments and the library's in-memory objects, and the shape of
that something is a real decision.

## The decision

**A file-based adapter layer.** Every tool:

- takes **paths and simple typed parameters** (strings, numbers, lists),
- reads what it needs from disk,
- writes its artifacts into the run directory,
- returns a **compact JSON summary plus paths** — never data.

So `fit_encoding` as a tool reads a table from disk, writes
`tables/t3.parquet`, and returns `{"table": "tables/t3.parquet", "new_columns":
["city_te"], ...}`. The agent never sees a row of data.

The run directory is the state. The agent's context is a view of it.

## Alternatives considered

**Keep the DataFrames in memory, hold them in the agent's dependency object.**
Much cheaper — a dict of `{name: df}` passed to each tool, no serialisation, no
parquet round-trips.

Rejected because in-memory state dies with the process. Two things we already
knew we wanted become impossible:

- **Resumption.** A long run will exhaust its context window. The fix is to start
  a fresh agent pointed at the same run directory and let it read what happened.
  If the tables only existed in the dead process's memory, there is nothing to
  point at.
- **Crash survival.** An agent run costs real money in tokens. Losing eight
  completed experiments to a crash in the ninth is not acceptable, and with
  in-memory state it is unavoidable.

Choosing the cheap option would have meant rewriting the tool layer later, after
every tool had been built against the wrong contract.

**Return data in the tool result.** Rejected immediately: a tool that returns a
DataFrame returns it *into the model's context window*, which is both expensive
and the thing that makes long runs impossible. There is a test asserting every
tool's return annotation is `dict`.

## Consequences

**Good**

- Context grows slowly. A tool result is a few hundred tokens regardless of
  whether the table has 200 rows or 2 million, so a run can span dozens of cycles.
- The tools have **no LLM dependency at all**. They are plain functions taking a
  session object and some arguments, which means they are unit-testable and — more
  useful in practice — a human can drive the entire pipeline by hand before letting
  a model near it. Every notebook in `notebooks/` does exactly that.
- Artifacts are inspectable. When a run does something strange, the tables,
  models, out-of-fold predictions and journal are all sitting on disk.

**Bad**

- Real serialisation cost. Every step round-trips through parquet even when the
  data is small.
- The adapter layer is code we own and did not originally plan for
  (`harness/tools.py`, ~500 lines with the later additions).
- Paths become part of the model's working vocabulary, so it can invent one. It
  does. `RunSession.resolve_table` therefore answers an unknown path by listing the
  tables that exist rather than raising a bare `FileNotFoundError` — showing the
  model the menu is more useful than a stack trace.

## Where it lives in the code

- `src/tabular_agent/harness/tools.py` — the adapter layer.
- `src/tabular_agent/harness/session.py` — `RunSession` holds *pointers only*
  (run dir, data path, task spec), so a fresh process handed the same run
  directory can continue.
- `tests/test_harness_tools.py` — the whole file runs without `pydantic_ai`
  installed, which is the check that the tools really are LLM-free.
