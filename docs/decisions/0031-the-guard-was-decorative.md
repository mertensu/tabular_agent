# 0031 — The serialisation guard was decorative; correctness moves into the primitives

- **Date:** 2026-08-11
- **Status:** accepted, implemented
- **Supersedes:** the *mechanism* in [0008](0008-serialise-tool-calls.md). Its
  analysis of the danger was exactly right and is why this was findable at all;
  the setting it chose does something else. **0008's decision — that concurrent
  tool calls must not be able to corrupt a run — stands and is now actually
  enforced.**
- **Related:** [0003](0003-file-based-tools.md) (the run dir is the state),
  [0004](0004-tools-write-the-journal.md)

## What happened

Rossmann v3, at experiment 8, holding the best model any run had produced
(`val_score` 0.12344, well past v1's honest 0.13173 and closing on the Kaggle
median). It died:

    OSError: Corrupt snappy compressed data

The journal had **four duplicate table ids** — `t3`, `t7`, `t10`, `t13` each
claimed twice, with the id sequence then skipping (`t7, t7, t9`). Two
`fit_encoding` calls had each computed the same id, each written the same
`tables/tN.parquet`, and one file had been read while another was writing it.

## Why the guard did not guard

0008 set `max_concurrency=1` on the `Agent` constructor, with a comment reading
*"One tool call at a time... two parallel calls can claim the same `t1` and race
on the same parquet"*. The parameter's own documentation:

    max_concurrency: Optional limit on concurrent agent runs.

**Concurrent agent runs, not concurrent tool calls.** There is one run. The
setting has never had anything to do with the race it was written to prevent.

It survived months of review because it was never falsified: models emitted one
tool call per message, so the property held for a reason unrelated to the code
claiming to cause it. And there was a test — `test_tool_calls_are_serialised` —
which asserted `agent._concurrency_limiter.max_running == 1`. It checked that the
argument had been passed. **A test named for a property, verifying a mechanism,
green throughout.**

## The decision

**Correctness moves out of the framework setting and into the primitives.**

- `journal.next_table_id` claims by `os.open(..., O_CREAT | O_EXCL)` on the
  table's own path; `next_experiment_id` claims by `os.mkdir` on the experiment's
  directory. Both make the claim and the check one operation, so a loser takes the
  next id instead of sharing one.
- `io.save_parquet` writes to a temporary file, `fsync`s, and `os.replace`s — so a
  reader sees the old table or the new one and never a half-written one. This is
  what `journal.save` has done since 2026-08-06; it should have been done to the
  data at the same time, and the gap is exactly what corrupted the run.

Ordering is now a *nicety* rather than a correctness requirement. A driver that
wants it can wrap its run in `agent.parallel_tool_call_execution_mode("sequential")`
— the real knob, which does serialise tool calls — and nothing breaks if it
forgets.

## What was rejected

**Setting the correct framework option and stopping there.** It would fix this
instance and rebuild the same fragility: a run's integrity would again depend on
one keyword argument being right, in a library that renamed the concept once
already. It also does nothing for two *processes* on one run dir — which 0008
itself flagged as unsolved, and which the exclusive-create claim now covers.

**Reaching into `_parallel_execution_mode_ctx_var`.** It is the private variable
behind the public context manager, and setting it durably would work today.
Depending on a private name for correctness is the same bet 0008 lost.

**A lock around the claim.** Works in-process, does nothing across processes, and
adds a thing to forget.

## What it costs

**Good**

- The invariant holds under any interleaving, in one process or several, whatever
  the framework does next.
- The test now exercises the property from threads rather than asserting a
  constructor argument.
- Atomic table writes make *every* overwrite safe, not only the racing kind.

**Bad**

- **An empty placeholder file exists between claiming an id and writing the
  table.** Harmless — `save_parquet` replaces it atomically — but a crash in
  between leaves a zero-byte parquet that will fail to read rather than being
  absent. Nothing cleans it up.
- **A claimed id is never released.** A tool that claims and then raises has
  burned that number; ids stay unique and stop being contiguous.
- **`v3`'s numbers are still trustworthy but its journal is not.** Each losing
  write was fully replaced, parquet *and* lineage sidecar, so what was scored was
  internally consistent. The journal still records a column living in a file that
  does not contain it.
- The dangerous ordering — train on a table, have it overwritten, then bundle from
  the replacement's lineage — was reachable in v3 and simply did not occur. There
  is no test for it, because with unique ids it can no longer happen.

## Where it lives

- `harness/journal.py` — `next_table_id`, `next_experiment_id`.
- `io.py` — `save_parquet`.
- `harness/agent.py` — the note replacing `max_concurrency=1`.
- `tests/test_harness_agent.py` — `test_concurrent_id_claims_never_collide`,
  `test_a_table_write_is_all_or_nothing`.
