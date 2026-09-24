# Decision records

Why this project is built the way it is.

Each file records **one decision**: the problem it solved, what was chosen, what
was rejected, and what it cost. They are written for someone who knows data
science but has never seen this codebase — project-specific vocabulary is
explained where it first matters.

This directory holds the twelve that still carry the product: its architecture,
its safety invariants, how it measures, what it ships, and its public contracts.
The other eighty — measurements, refuted mechanisms, dataset findings, plans that
were superseded — are in [`research/archive/decisions/`](../../research/archive/decisions/),
with the full index and the table of where each thread stands.

## How to read these

A decision record is **history, not documentation**. It was true on the day it
was written and it is never edited to match later reality. If a decision is
later reversed, a new record supersedes it and the old one stays put, because
the reasoning that turned out to be wrong is usually the most useful part.

So **do not treat these as a description of how the system currently works.**
For that, read the code — the rationale for individual lines lives in docstrings
and comments next to the thing it explains. These records explain the *shape*
the code has, not its current contents.

Numbers are one sequence across both directories, so gaps here are records that
live in the archive, and a record here may cite one there. New records continue
the sequence.

## What this project is

`tabular_agent` is a **leakage-safe tabular machine-learning harness**. An LLM
agent is given a dataset and a target column, and works towards a model that
generalises. The central claim is that it is *leakage-safe by construction*: the
agent cannot commit the standard sins (training on the target, fitting an encoder
on data the model will be scored against) because the tools it is given are the
only sanctioned way it can touch the data, and none of them can express those
mistakes.

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
- [0018 — The result payload is a budget, and so is the vocabulary](0018-the-payload-is-a-budget.md)
- [0020 — The read-only sandbox has no kernel; the run dir is the state](0020-the-sandbox-has-no-kernel.md)

### Safety invariants

- [0009 — The target column is invisible to agent-written code](0009-target-hidden-from-transforms.md)
  *(enforced by what the tools hand over; the sandbox is an isolation boundary,
  not a security one — see `sandbox.py`)*
- [0011 — Statelessness is tested as a property, not blacklisted](0011-stability-check.md)
- [0030 — A model that could never be served is refused, not scored](0030-a-model-that-cannot-be-served-is-not-trained.md)

### Measuring and shipping

- [0022 — The yardstick must measure the quantity being ranked](0022-the-yardstick-must-measure-the-ranked-quantity.md)
- [0035 — The search is over features; models are enumerated, not searched](0035-the-search-is-over-features.md)
- [0044 — What ships is the average of the terminal fits, not the best of them](0044-what-ships-is-the-average-not-the-argmax.md)
  *(still standing; its equal weighting was superseded by
  [0080](../../research/archive/decisions/0080-a-blend-weight-a-member-cannot-inflate-decided-before-the-seal.md))*

### Public contracts

- [0083 — Learned features are a reusable artifact](0083-learned-features-are-a-reusable-artifact.md)
- [0088 — A run may propose a capability gap, but not change the product](0088-a-run-may-propose-a-capability-gap-but-not-change-the-product.md)
- [0089 — Forecasting is an explicit public contract, not an inferred mode](0089-forecasting-is-an-explicit-public-contract.md)

## Related reading

- `docs/oof_vs_final_encoder.md` — why the encoder shipped in the bundle is
  fitted on all rows while the values trained on are out-of-fold, and why that
  is not leakage. The foundational idea the whole project rests on.
- `docs/target_encoding_leakage_insight.md` — who gets lied to by leakage, the
  model or the metric.
