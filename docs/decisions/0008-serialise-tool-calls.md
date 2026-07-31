# 0008 — One tool call at a time

- **Date:** 2026-07-30
- **Status:** accepted (half a fix; see Consequences)
- **Commit:** `17b981e`

## The problem

Tables and experiments get sequential ids, and the id is claimed like this:

```python
def next_table_id(run_dir):
    return f"t{len(load(run_dir)['tables']) + 1}"
```

Read the journal, count, add one. Then the caller writes
`tables/<id>.parquet` and appends the row.

That is a read-then-write with no lock, and it is only safe if one thing runs at
a time. Two concurrent calls both read a journal with two tables, both decide
they are `t3`, and both write `tables/t3.parquet`.

## Background: models can request several tools at once

A model does not have to emit one tool call per message. It can put several in a
single message — "encode `city`, and also encode `size`" — and PydanticAI will
run them concurrently, dispatching synchronous tool functions to worker threads.

So the race is reachable through normal model behaviour, not just through
deliberate parallelism on our side.

The two collisions differ in severity:

- **Two tables claiming `t3`** — both write the same parquet file. One corrupt
  parquet crash was observed, consistent with this, though never reproduced on
  demand.
- **Two experiments claiming `e7`** — much worse, because it is **silent**. Both
  write into `experiments/e7/`, one set of artifacts overwrites the other, and the
  leaderboard ends up with two rows pointing at one directory. Nothing errors and
  the run reports a model that does not exist as trained.

## The decision

`Agent(..., max_concurrency=1)`. Tool calls are serialised, which removes the
entire class of collision within one agent.

The cost is parallelism we do not benefit from: each tool call here is seconds of
local CPU (fitting a model, writing a parquet), not a network wait, so overlapping
them saves little and the sequencing is not on the critical path — the wall clock
of a run is dominated by LLM round trips.

## Alternatives considered

**Exclusive-create id claim plus write-then-rename.** The real fix: claim an id
by creating a file with `O_EXCL` (which fails atomically if it exists), write
artifacts to a temporary name, rename into place when complete. Rename is atomic
on a POSIX filesystem, so a reader never sees a half-written table.

Not rejected — **still outstanding**, and it is what should eventually land.
`max_concurrency` was taken first because it is one line and closes the common
case immediately.

**Do nothing until it is reproduced.** Rejected. The `e7` variant is silent, and
a bug that corrupts results without any signal is not one to wait for evidence
on. The one crash we did see was consistent with it.

## Consequences

**Good**

- One line, no new machinery, and it eliminates the failure mode within a run.
- Asserted in a test, so removing it fails the build.

**Bad**

- **It is only half a fix, and the missing half is the more important one.**
  `max_concurrency` is a property of one `Agent` object. It does nothing about two
  *processes* touching the same run directory — which is precisely the resumption
  scenario decision 0003 exists to support: a resumed run started while the
  original is still alive would collide exactly as before.
- The test asserts on `agent._concurrency_limiter.max_running`, a private
  attribute, so it is exposed to library churn.
- It has never been *verified* to prevent an observed collision, because the
  collision was never reproduced on demand. We are reasoning from the code, not
  from a failing test that now passes.

## Where it lives in the code

- `src/tabular_agent/harness/agent.py` — `max_concurrency=1` on the `Agent`.
- `src/tabular_agent/harness/journal.py` — `next_table_id`,
  `next_experiment_id`, the unguarded claims.
- `tests/test_harness_agent.py` — `test_tool_calls_are_serialised`.
