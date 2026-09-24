# 0020 — The read-only sandbox has no kernel; the run dir is the state

- **Date:** 2026-08-10
- **Status:** accepted
- **Decided by:** Ulf (the call to build `explore_experiment` first, and to
  capture stdout)
- **Related:** [0003](../../research/archive/decisions/0003-file-based-tools.md) (tools speak in paths),
  [0009](0009-target-hidden-from-transforms.md),
  [0010](../../research/archive/decisions/0010-store-code-not-columns.md),
  [0017](../../research/archive/decisions/0017-credit-by-ablation.md) (the other free diagnostic)

## The problem

The agent could inspect data, build tables, train, and score. It could not look
at **where a trained model was wrong**. The loop's arc from `score(OOF)` back to
`hypothesise` was missing: `train_tabular_model` writes `oof.parquet` and no tool
ever showed it to anyone, so the agent picked its next feature from column names
and a single scalar per experiment. That is why runs reached for `price_squared`.

A second, larger idea had been parked alongside it. Reading
`PrimeIntellect-ai/prime-agent`, its "RLM" abstraction gives the model a
**persistent Python kernel**: data lives in kernel variables across calls, and
only what the model prints enters the context. Their framing — the store is the
state, the context is a view of it — is this project's framing, and the parked
note proposed adopting the mechanism for the run's *records* (leaderboard,
profiles, lineage) as well as its rows.

So there were two questions, and they had been bundled: what do we point a
read-only sandbox at, and does that sandbox need to be a live kernel.

## The decision

**Two decisions, and the second is the one worth the record.**

### 1. `explore_experiment(exp_id, code)` — read-only code over one experiment's rows

The frame is the experiment's **whole source table** plus `oof_pred`,
`residual`, and `fold`. Not just the numeric columns the model trained on: the
string columns a GBDT cannot consume are exactly the ones worth grouping
residuals by, and *"the error is concentrated in one city"* is how an agent
discovers that a column is worth encoding at all. They are free here because
nothing here becomes a feature.

`fold` is recomputed from `make_folds` rather than stored — it is deterministic
in `(y, task_type, n_splits, seed)`, all fixed for the run. Two lines, and
without it "a genuinely weak slice" and "one fold fit badly" look identical.

Free like `explain_table` (0017): no leaderboard row, no budget, allowed after a
refusal. The moment the agent most needs to ask what went wrong is when it has
just been told to stop.

### 2. The sandbox is stateless per call. There is no kernel.

**Rejected: the persistent kernel.** The context economics that make RLM
attractive do not come from the kernel being alive — they come from the
**sandbox boundary**. `run_readonly` already pickles frames into a subprocess and
returns only a capped summary; the rows never touch the context. A long-lived
process adds exactly one thing on top of that: names that survive between calls.

Three reasons that is the wrong thing to add here, in increasing order of weight:

1. **It buys nothing measurable.** What persistence saves is re-pickling the
   frames each call. At 150 experiments the journal is a ~100KB DataFrame.
2. **Episodes delete it anyway.** The planned driver loop runs a *fresh agent per
   episode* against the same `run_dir`. Kernel variables do not survive that
   seam, so the agent could never rely on them and would have to rebuild from
   the journal every episode regardless. We would be paying for a live process
   to get state with a fifteen-experiment shelf life.
3. **It is the one class of state this design has spent every decision
   removing.** 0003 put the tables on disk, 0004 put the record on disk, 0010 put
   the transform's *code* on disk, and `RunSession` promises that a fresh process
   handed the same `run_dir` continues the run. A kernel variable is run state
   that is not in `run_dir`, cannot be resumed, and cannot be inspected after a
   crash. The rule that survives all four is: **the state is the run directory,
   and everything else is a view of it that may be thrown away at any time.**

So: the namespace is rebuilt from `run_dir` on every call, and
`test_nothing_survives_between_calls` holds the line.

### The mechanics that had to change

- `run_readonly(df, code)` became `run_readonly(frames, code)` — a `{name:
  object}` namespace. The child reads the agent's signature and passes only the
  frames it named, so `def explore(lb):` works without declaring the ones it
  ignores, and without the agent having to get an argument *order* right. A name
  it asks for that we do not have is a typo or an invented frame, so the error
  lists what exists.
- **stdout is captured** and returned alongside the result, capped at 2000
  characters. Two reasons, and the first is a bug fix: a model writes `print(...)`
  in exploration code by reflex, and discarding it silently is the same failure
  as a profile that omitted the categories an ordinal encoding needs — the agent
  behaves reasonably, gets nothing, and is told nothing. The second is that
  prints are attached to the **error** path too, so code that raised on line 8
  still reports what lines 1–7 computed, which is most of what a fix is written
  from.

## What was rejected

- **A persistent kernel** — above. If it ever returns, it needs a new record
  saying what changed, because the resumability argument does not weaken on its
  own.
- **`explore_journal` first.** It is the other half of the parked idea and has no
  leakage surface at all, which made it the tempting first build. But its entire
  justification is a 150-experiment run — at 8 experiments the leaderboard is
  cheaper to read as text than to query — and 150 experiments needs episodes,
  which do not exist. It would have shipped unexercised. Still worth building,
  and the namespace plumbing above is exactly what it needs.
- **Omitting the target from the frame.** `residual` is `y - oof_pred`, so for a
  regression the target is one subtraction away; leaving the column out would
  have been decoration presented as a guard. It is in the frame, named.
- **A single `explore(what, code)` tool.** PydanticAI ships each docstring as the
  tool description and the agent picks by name — one tool covering both
  namespaces would have to describe both on every call.
- **Charging it against `max_experiments`.** Pricing a look at a failure the same
  as causing one. Same reasoning as 0017.
- **Deleting `explore_dataframe`.** It became a two-line adapter, but the phase-2
  notebook teaches the sandbox through it and there is no gain worth breaking
  that for.

## What it costs

**Good**

- The `score → hypothesise` arc exists, on runs we can do today. It does not wait
  for episodes, unlike everything else parked next to it.
- Verified on a fixture whose error is concentrated in a string column the model
  cannot train on: grouping `residual` by `segment` recovers a >3× difference
  that no field in any tool result reports.
- Row alignment is positional and provable — recomputing the metric per `fold`
  from the frame reproduces the experiment's recorded `fold_metrics` exactly.
- The sandbox now serves any number of named frames, which is `explore_journal`'s
  whole prerequisite.

**Bad**

- **The target is in the frame and the agent can look at it per segment, freely
  and repeatedly.** The guard is the absence of a persistence path, not the
  absence of the number — the same guard `explore.py` has always documented. A
  200-row cap across unlimited free calls means the target is *reconstructable*
  in principle. Judged acceptable: there is no route back in (0010 stores code,
  so a hardcoded literal dies at predict on a length mismatch), and the threat
  model is careless, not adversarial.
- **The leak that remains is through the agent's reasoning, not through data.**
  An agent that picks features by staring at out-of-fold errors is doing feature
  selection informed by held-out predictions. Every data-path guard holds and no
  guard catches this. It is the same hole named for residual feedback on
  2026-08-05, accepted for the same reason — and it is now *live* rather than
  hypothetical, which is the real price of this record.
- Two output contracts where there was one: `run_readonly` returns
  `{"result", "stdout"}`, `explore_dataframe` still returns the summary alone.
- `SYSTEM_PROMPT` could no longer say "no free-form code". It was already loose —
  `transform_features` has run pandas since 2026-07-30 — and is now accurate
  instead: two tools run agent code, and what happens to the output is the
  difference between them.

## Where it lives in the code

- `src/tabular_agent/sandbox.py` — `run_readonly` (namespace), `_bind`,
  `MAX_STDOUT_CHARS`, stdout on the error path.
- `src/tabular_agent/harness/tools.py` — `explore_experiment`,
  `_residual_frame`, `_experiment_row`.
- `src/tabular_agent/harness/agent.py` — the seventh tool.
- `tests/test_sandbox.py` — `test_only_the_declared_frames_are_passed`,
  `test_frames_are_not_shared_between_calls`,
  `test_prints_before_an_error_survive_in_the_message`.
- `tests/test_harness_tools.py` —
  `test_rows_line_up_with_the_folds_they_were_held_out_in`,
  `test_the_frame_carries_the_error_and_the_columns_the_model_could_not_use`.
</content>
</invoke>
