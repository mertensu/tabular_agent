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
- [0020 — The read-only sandbox has no kernel; the run dir is the state](0020-the-sandbox-has-no-kernel.md)
- [0024 — The episode stop is structural, and the brief is not optional](0024-the-episode-stop-is-structural.md)
  *(accepted, not yet implemented; two supporting claims superseded by 0025)*
- [0025 — Episodes buy feasibility, not cost](0025-episodes-buy-feasibility-not-cost.md)
  *(accepted, not yet implemented)*
- [0026 — The brief is rendered, never accumulated](0026-the-brief-is-rendered-not-remembered.md)
  *(accepted and implemented; its central claim is superseded by 0027)*
- [0027 — The brief did not cut the orientation tax](0027-the-brief-did-not-cut-the-orientation-tax.md)
  *(a measurement: orientation is a fraction of the episode, not an information deficit)*

### Keeping features honest

- [0005 — Ordinal level order is a required argument, never inferred](0005-ordinal-order-is-required.md)
- [0006 — Feature lineage per table, not per run](0006-lineage-per-table.md)
- [0009 — The target column is invisible to agent-written code](0009-target-hidden-from-transforms.md)
- [0010 — Store the transform code, not the columns it produced](0010-store-code-not-columns.md)
- [0011 — Statelessness is tested as a property, not blacklisted](0011-stability-check.md)
- [0021 — The agent picks the target scale, from a fixed menu](0021-the-agent-picks-the-target-scale.md)
  *(the menu stands; where the choice is made is superseded by 0035)*
- [0032 — Fitted group aggregates, and the first step that is not replayed](0032-fitted-aggregates-and-a-step-that-is-not-replayed.md)
  *(what 0011 rejects becomes legal once the value is stored; and why a row
  filter is train-time-only)*

### Surviving a real agent

- [0007 — A bad argument is a correction, not the end of the run](0007-retryable-arguments.md)
- [0008 — One tool call at a time](0008-serialise-tool-calls.md)
  *(the danger is real; its mechanism was decorative — see 0031)*
- [0031 — The serialisation guard was decorative; correctness moves into the primitives](0031-the-guard-was-decorative.md)
- [0013 — A refusal is a return value, not an exception](0013-refusal-as-return-value.md)
- [0033 — A ceiling in the wrong unit is no ceiling; the search is bounded, the fit is not](0033-a-ceiling-in-the-wrong-unit-is-no-ceiling.md)
  *(what a self-tuning model may cost at scale; and a wall-clock bound that still
  cannot bound a single fold)*

### Knowing when to stop, and what worked

- [0012 — The harness decides when the run is over](0012-harness-owns-the-stop.md)
- [0014 — Exploration is cheap; tuning one table is not](0014-two-level-stop.md)
  *(its two levels collapse into one under 0035: there are no model experiments
  left for the per-table budget to refuse)*
- [0015 — Attribution gaps are reported, not prevented](0015-report-attribution-gaps.md)
- [0016 — Feature credit per tool call, not per column](0016-block-level-credit.md)
  *(superseded by 0017)*
- [0017 — Credit is measured by refitting without the columns, not by importance](0017-credit-by-ablation.md)
- [0037 — Residuals need folds; the ranking does not](0037-residuals-need-folds-the-ranking-does-not.md)
  *(why `explore_table` refits across folds while the ranking path deliberately
  does not, and why in-sample residuals are not the cheap version)*
- [0019 — The leaderboard is a public leaderboard](0019-the-leaderboard-is-a-public-leaderboard.md)
- [0022 — The yardstick must measure the quantity being ranked](0022-the-yardstick-must-measure-the-ranked-quantity.md)
  *(accepted and now implemented; one claim about what the fix achieves is corrected by 0028)*
- [0028 — `val_se` is implemented; the bar it is a quarter of is still open](0028-the-yardstick-is-fixed-the-bar-is-not.md)
  *(accepted; `min_gain_ratio` deliberately left open pending a second dataset)*
- [0023 — The third split has evidence now](0023-the-third-split-has-evidence-now.md)
  *(accepted, not yet implemented)*
- [0034 — Breaking a `val_se` tie with `cv_score` was tested and rejected](0034-the-tiebreak-was-tested-and-rejected.md)
  *(a measurement: 1 win, 1 loss, 1 null — and why the real conclusion is 0023)*
- [0035 — The search is over features; models are enumerated, not searched](0035-the-search-is-over-features.md)
  *(accepted and implemented; supersedes 0021 on where the target scale is
  chosen, and collapses 0014's two-level stop — its central mechanism is
  corrected by 0036, and 0038 is the first run under it)*
- [0036 — The pairing asymmetry was not there; the probe is justified by coverage](0036-the-pairing-asymmetry-was-not-there.md)
  *(a measurement: every comparison is paired, model or table alike — and a
  candidate yardstick for 0028's open `min_gain_ratio`)*
- [0038 — The first feature-only run: less optimism, and a prediction that failed](0038-the-first-feature-only-run.md)
  *(a measurement: optimism +0.01127 -> +0.00458 on a pre-registered A/B; why a
  val-cv gap is not a diagnostic for selection noise — its "optimism belongs to
  the table" hypothesis is rejected by 0039)*
- [0039 — Seguro v3: two defects, and 0038's hypothesis falsified](0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md)
  *(the ruler must be on the terminal ballot; removing a tool removes the
  guarantees it carried; optimism attaches to the selected measurement)*

### Keeping the agent's surface small

- [0018 — The result payload is a budget, and so is the vocabulary](0018-the-payload-is-a-budget.md)

### Who decides what

- [0029 — The harness owns the exam; folds respect the split](0029-the-harness-owns-the-exam.md)
  *(the evaluation protocol is ours, the modelling is the agent's — and what
  follows for time-ordered data)*
- [0030 — A model that could never be served is refused, not scored](0030-a-model-that-cannot-be-served-is-not-trained.md)
  *(the serving schema, and the Rossmann run that argued for it)*

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
