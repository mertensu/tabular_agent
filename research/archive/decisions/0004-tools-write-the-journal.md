# 0004 — The tools write the journal, not the agent

- **Date:** 2026-07-28
- **Status:** accepted
- **Commit:** `c6da944`

## The problem

A run needs a record: what was tried, what it scored, what the best model is so
far. Call it the **journal** (`run_journal.json`). It exists for two audiences —
the human reading the run afterwards, and a *fresh agent* resuming after a
context reset, for whom the journal is the only memory of what already happened.

The obvious way to produce it is to give the agent a `log_experiment` tool and
ask it to record each result. This is how a lot of agent systems do it.

It is also unfalsifiable, which is the problem.

## Background: why "unfalsifiable" is the objection

If the agent is responsible for logging, then a missing entry has two possible
causes: the experiment never happened, or it happened and was not logged. From
outside — reading the journal afterwards — **those two look identical.** There is
no observation that distinguishes them.

So "the agent logs its experiments" is not a property of the system. It is a hope
about the model's behaviour, and hopes degrade quietly: a model under context
pressure drops the bookkeeping call first, precisely when the run is long enough
that the record matters most.

## The decision

**`train_tabular_model` appends the leaderboard row itself**, as a side effect of
training. The experiment record is a by-product of the experiment, not a chore
performed afterwards. It cannot be skipped, forgotten, or misremembered, because
nothing the model does or fails to do is involved.

The journal has two layers, split by *who has the information*:

- **Facts** — written by the tools. Leaderboard rows, derived tables, the task
  spec. Mechanical, and complete by construction.
- **Interpretation** — the thin layer the agent could write: a hypothesis, why it
  tried something, what it plans next. Genuinely needs a model; genuinely cannot
  be automated. (`append_note` exists for this; no tool exposes it yet.)

The `best` pointer is recomputed from the entire leaderboard on every append,
rather than compared against the previous best, so it stays correct even if a row
is edited or the journal is rebuilt by hand.

## Alternatives considered

**A `log_experiment` tool for the agent.** Rejected as above.

**Reconstruct the journal from the artifacts on disk.** Workable — every
experiment writes a directory — but it makes the record a derived thing that has
to be re-derived correctly, and it loses ordering.

## Consequences

**Good**

- The record and reality cannot drift. This turns out to matter more than
  expected: models routinely describe their own runs inaccurately. One reported
  ordinal codes as "1, 2, 3" when they were 0, 1, 2; another claimed "city one-hot
  encoding" when it had used frequency encoding. **The journal is the record; the
  model's summary is a claim.** When they disagree the journal is right, and it
  exists so that comparison is possible at all.
- Resumption works: a fresh agent reads one file and knows what has been tried.
- The pattern generalised cleanly. `fit_encoding` and `transform_features` record
  their tables the same way, which is what later made feature lineage (0006) and
  block-level credit (0016) possible without new bookkeeping.

**Bad**

- The journal grows without bound. A long run will eventually produce a journal
  too big for the context it is read into. Compaction is designed (spec §19) and
  not yet built.
- Each leaderboard row carries some redundancy — notably `source_table` — so a
  resuming agent can read the leaderboard alone without opening each experiment's
  metadata. Deliberate, but it is duplication.

## Where it lives in the code

- `src/tabular_agent/harness/journal.py` — the store. Deliberately dumb: load,
  save, append, and pure functions that interpret the result.
- `src/tabular_agent/harness/tools.py` — `train_tabular_model` calls
  `journal.append_experiment` in the same breath as writing the model.
