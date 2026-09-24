# 0018 — The result payload is a budget, and so is the vocabulary

- **Date:** 2026-08-06
- **Status:** accepted
- **Decided by:** Ulf (noticed the drift, and asked whether the two new concepts
  were needed — they were not)
- **Related:** [0017](../../research/archive/decisions/0017-credit-by-ablation.md), [0003](../../research/archive/decisions/0003-file-based-tools.md)

## The problem

`train_tabular_model` had grown to **21 fields**, returned on every training call.
The system prompt was 924 words. Ulf:

> i have the feeling it starts to be bloated by info, tools etc

There was a specific mechanism behind it, visible in the session that produced
0017: **every time a number read confusingly, a caption was added next to it**
instead of the number being fixed or explained once. `how_to_read_blocks`,
`how_to_read_gains`, `how_to_read_block_credit`, `vs_parent_basis` — four fields
whose only job was describing other fields.

The worst case was self-inflicted within a single session: `how_to_read_gains` was
added to the payload *and* the same sentence written into the system prompt. It
shipped on every call to say something that never changed.

## The first decision: static prose belongs in the prompt

The rule, and it is checkable:

> If the wording is fixed, it goes in the system prompt, where it is read once.
> If it varies with the run, it goes in the payload, where it carries information.

`note` (where the run stands) and `vs_source_basis` (what would produce a missing
gain) are the entire second category. Everything else that was explaining rather
than reporting moved into the prompt.

This is **not** an argument for less explanation — Ulf was explicit that
explanations help. It is an argument about where they are read. A caption shipped
beside a field is re-read on every call and says the same thing every time; the
same sentence in the prompt is read once and costs nothing thereafter.

Removed on the same rule:

- `metric`, `higher_is_better` — constant for a whole run. Stated once by
  `inspect_data`.
- `fold_metrics` — the raw per-fold list under `fold_spread`, which is the summary
  that changes a decision. Still written to `meta.json` for the report.
- `since_improvement`, `since_improvement_on_table`, `experiments_remaining` — all
  three already restated in `note`'s prose. The prose survived.
- `vs_source_basis` when the gain exists — then it describes the obvious.

21 fields → 13, and prompt 924 → 731 words on this rule alone.

## Then two more rules, because the prompt kept regrowing

Trimming duplication was not enough — the `explain_table` section grew back to
five paragraphs within a day, each added in response to something that read
unclearly, each individually defensible. Ulf again:

> dont you think its kinda overengineered, too constructed?

and then the diagnosis:

> the agent can reason on its own. its subtle ofc, but we have to tell him only
> what he cannot reason about on this own

**Rule two: say only what cannot be worked out.** The test is one question — *could
a competent data scientist derive this from the tool signatures and the numbers?*
If yes, it is not worth tokens on every run. That deleted "a strong mean with a
large spread is an unstable model", "a model that cannot beat the trivial baseline
is a bug, not a model", "change one thing at a time so you can attribute", and the
two bullets spelling out what leave-one-out implies about interactions and
redundancy — all true, all derivable from `contribution = score(all) − score(all
minus these)` once that definition is given.

The pattern this exposes: **justifications go, the facts they justified stay.** A
model that grasps the fact reconstructs the reason for free, so the reason is the
part that was never earning its place.

**Rule three: say only what no single tool can say.** Ulf:

> we do not have to explain the tools right because the agent sees the doc-strings
> of those

Verified against the built agent: PydanticAI sends each tool's docstring as its
description, and `explain_table`'s already said "costs one fit per contributing
table plus one, records no experiment, and counts against no budget" — which the
prompt was repeating almost verbatim. Per-tool facts belong in the docstring,
where they arrive with the tool; the prompt is for what spans tools and has no
other home. `vs_baseline`/`vs_source_table` semantics moved into
`train_tabular_model`'s docstring, leave-one-out into `explain_table`'s.

**And: no third-person self-reference.** The prompt opened "You are a tabular
machine-learning agent". Naming the reader in the third person spends words
describing it to itself. There is now no occurrence of "agent" in the prompt, and
a check in the test suite.

924 → **272 words**, with every harness fact still present.

## The second decision: no vocabulary that duplicates a name already in use

Ulf, on the concepts 0016 had introduced:

> we introduce quite some concepts, e.g. parents and blocks. is this needed?

It was not. Both were second names for things the agent already had:

- **"Parent"** was `source_table` — a field already present in every training,
  transform and encoding result. `_vs_parent` computed it as
  `{t["path"]: t["source_table"]}`. The name `vs_parent_table` sat in the same dict
  as the `source_table` field it referred to.
- **"Block"** was a table. `fit_encoding` and `transform_features` are the only two
  callers of `append_table`, so one tool call writes exactly one table with exactly
  one `new_columns` list. Block ↔ table is 1:1 *by construction* — which is why
  `journal.blocks()` was already returning table paths as block identities.

So the count was not four concepts but **two concepts wearing four names**. The
agent has no choice about learning "table" and "source_table"; they are in every
payload and every tool signature. The other two were additions on top.

| was | is |
|---|---|
| `vs_parent_table` | `vs_source_table` |
| `feature_blocks` | `columns_by_table` |
| `block_credit` | `credit_by_table` |
| `"original columns"` | `"the raw input table"` |
| `journal.blocks` | `journal.columns_by_table` |

What "block" was carrying — that credit is per tool call, never per column (0016) —
is carried just as well by "the columns `t1` added", using a noun the agent cannot
avoid learning anyway.

## Both guarded, because neither guarded itself

`tests/test_harness_agent.py` has pinned the exact tool set since the beginning, and
it works: adding `explain_table` in 0017 failed that test and forced a deliberate
decision with a comment justifying it. The payload had no equivalent, which is how
it reached 21 fields without anyone choosing that.

`TRAIN_RESULT_FIELDS` in `tests/test_harness_tools.py` now pins the result surface
the same way. A 14th field fails a test whose comment says where static explanation
is supposed to go instead.

## What was rejected

- **Cutting explanations to save space.** Explicitly rejected by Ulf, and correctly.
  The prompt got shorter by removing duplication and derivable content, never by
  leaving a harness fact unstated.
- **Keeping derivable content "just in case".** Every deleted line was true and
  mildly reassuring, which is exactly why the prompt tripled. "It might help" is
  not a reason; "it cannot be derived" is.
- **A generic `metadata` sub-dict.** Moves fields one level down and calls it
  progress; the tokens still ship.
- **Keeping "block" as the credit unit's name.** Defensible — it does emphasise the
  tool call over the column. Rejected because the emphasis is available without the
  extra noun, and a word that duplicates an existing one costs the agent something
  on every result it reads.
- **Dropping a tool.** Six tools, each answering a question no other one can, was
  never the problem. The bloat was in the payload and the prose.

## What it costs

**Good**

- Every training result is ~40% smaller, on the surface the agent reads dozens of
  times per run.
- Two fewer concepts, and every noun in the prompt is now one the tool signatures
  already force.
- Both surfaces have a guard, so regrowth is a decision rather than a reflex.

**Bad**

- The prompt no longer warns that a lone half of an interaction scores ~0 and is
  not disproof. That reassurance was derivable and was cut; if a live run shows an
  idea being abandoned one call too early, it is the first thing to restore.
- Per-tool facts now live in `agent.py` docstrings, one file away from the prompt
  they used to sit beside. Cheaper to send, easier to leave inconsistent.
- Field renames break any saved journal from before this record. No migration was
  written; runs in flight are abandoned. Acceptable at this stage and not later.
- `fold_metrics` is genuinely gone from the payload. An agent that wanted to spot a
  single catastrophic fold now sees only the spread, and would have to read
  `meta.json` — which it has no tool to do.
- `TRAIN_RESULT_FIELDS` will be annoying exactly when someone is right to add a
  field. That is the intended cost; the tool-set guard has the same property.

## Where it lives in the code

- `src/tabular_agent/harness/tools.py` — the `train_tabular_model` return dict and
  the comment above it.
- `src/tabular_agent/harness/instructions.py` — the module docstring states the
  static/dynamic rule and the no-new-nouns rule.
- `src/tabular_agent/harness/journal.py` — `columns_by_table`, whose docstring
  records why it is not called `blocks`.
- `tests/test_harness_tools.py` — `TRAIN_RESULT_FIELDS`,
  `test_a_training_result_carries_nothing_beyond_its_agreed_fields`.
