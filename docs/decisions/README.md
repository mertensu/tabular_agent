# Decision records

Why this project is built the way it is.

Each file records **one decision**: the problem it solved, what was chosen, what
was rejected, and what it cost. They are written for someone who knows data
science but has never seen this codebase — project-specific vocabulary is
explained where it first matters.

## How to read these

A decision record is **history, not documentation**. It was true on the day it
was written and it is never edited to match later reality. If a decision is
later reversed, a new record supersedes it and the old one stays put, because
the reasoning that turned out to be wrong is usually the most useful part.

That means: **do not treat these as a description of how the system currently
works.** For that, read the code — the rationale for individual lines lives in
docstrings and comments next to the thing it explains, which is the only place
it cannot silently drift out of date. These records explain the *shape* the code
has, not its current contents.

## What this project is

`tabular_agent` is a **leakage-safe tabular machine-learning harness**. An LLM
agent is given a dataset and a target column, and works towards a model that
generalises. The central claim is that it is *leakage-safe by construction*: the
agent physically cannot commit the standard sins (training on the target,
fitting an encoder on data the model will be scored against) because the tools
it is given are the only way it can touch the data, and none of them can express
those mistakes.

Vocabulary used throughout:

| term | meaning |
|---|---|
| **run** | one agent session against one dataset, all of it inside a `run_dir/` |
| **table** | a file: the dataset plus columns added by one tool call. Named `t1`, `t2`, … |
| **experiment** | one model trained with cross-validation on one table. Named `e1`, `e2`, … |
| **journal** | `run_journal.json` — the run's record: task, leaderboard, tables |
| **lineage** | the steps needed to rebuild a table's columns from raw input, stored beside it |
| **bundle** | the deployable artifact: lineage + fitted model, predicts from raw rows |

## Index

### Architecture

- [0001 — Own a constrained harness rather than serve other agents](0001-own-the-harness.md)
- [0002 — PydanticAI as the harness framework](0002-pydantic-ai.md)
- [0003 — Tools take file paths, not DataFrames](0003-file-based-tools.md)
- [0004 — The tools write the journal, not the agent](0004-tools-write-the-journal.md)

### Keeping features honest

- [0005 — Ordinal level order is a required argument, never inferred](0005-ordinal-order-is-required.md)
- [0006 — Feature lineage per table, not per run](0006-lineage-per-table.md)
- [0009 — The target column is invisible to agent-written code](0009-target-hidden-from-transforms.md)
- [0010 — Store the transform code, not the columns it produced](0010-store-code-not-columns.md)
- [0011 — Statelessness is tested as a property, not blacklisted](0011-stability-check.md)

### Surviving a real agent

- [0007 — A bad argument is a correction, not the end of the run](0007-retryable-arguments.md)
- [0008 — One tool call at a time](0008-serialise-tool-calls.md)
- [0013 — A refusal is a return value, not an exception](0013-refusal-as-return-value.md)

### Knowing when to stop, and what worked

- [0012 — The harness decides when the run is over](0012-harness-owns-the-stop.md)
- [0014 — Exploration is cheap; tuning one table is not](0014-two-level-stop.md)
- [0015 — Attribution gaps are reported, not prevented](0015-report-attribution-gaps.md)
- [0016 — Feature credit per tool call, not per column](0016-block-level-credit.md)
  *(superseded by 0017)*
- [0017 — Credit is measured by refitting without the columns, not by importance](0017-credit-by-ablation.md)

### Keeping the agent's surface small

- [0018 — The result payload is a budget, and so is the vocabulary](0018-the-payload-is-a-budget.md)

## Related reading

Longer-form notes that are not decisions:

- `docs/oof_vs_final_encoder.md` — why the encoder shipped in the bundle is
  fitted on all rows while the values trained on are out-of-fold, and why that
  is not leakage. This is the foundational idea the whole project rests on.
- `docs/target_encoding_leakage_insight.md` — who gets lied to by leakage, the
  model or the metric.
- `docs/feature_attribution_and_agent_iteration.md` — the long version behind
  decisions 0015 and 0016. Predates 0017: its importance-share reasoning was
  superseded, and it has not been rewritten.
- `docs/superpowers/specs/2026-07-21-tabular-agent-mcp-design.md` — the original
  design spec. Its title is stale (see 0001); revisions are dated in place.
- `notebooks/` — hands-on walkthroughs, one per slice of the system, each
  driving the machinery by hand before letting an LLM near it.
