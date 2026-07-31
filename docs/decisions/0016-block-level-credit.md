# 0016 — Feature credit per tool call, not per column

- **Date:** 2026-07-30
- **Status:** accepted
- **Decided by:** Ulf (the reframe that set the direction, and the objection that
  shaped the answer)
- **Commit:** `0f332a8`
- **Long version:** `docs/feature_attribution_and_agent_iteration.md`

## The problem

Decision 0015 is about explaining a finished run to a human. Ulf then reframed
the question:

> it's less about me knowing in the end how much a feature contributed, but
> rather the agent iterating in the right way

Different problem, and the expensive tool (Shapley) is the wrong one for it. Fair
credit is for the human at the end. **Search guidance only has to beat nothing**
— and nothing was what the agent got.

Look at what a training result told it: a `cv_score`, a `fold_spread`, an
experiment id. It had just added sixteen columns and received one number. There
is no channel through which it can learn that `price_sq` did nothing. It is
generating candidates blind, which is precisely why it writes kitchen-sink
transforms: throw twenty features at the wall and hope.

## Why the obvious fix is a trap

Per-column feature importance is free — the fitted booster already computes it.
Report it and the agent can prune.

Ulf's objection:

> it could build a feature on top of parent which gives no improvement, but the
> two together help

That is fatal to per-column feedback. If C alone does nothing but C+B is the win,
telling the agent "C contributed 0.3%" teaches it to throw away whichever half of
an interaction arrived first — and B never gets its chance. XOR is the textbook
case; `price` and `area` are the mild everyday version.

**This reversed an earlier criticism.** The kitchen-sink transform had been
called noise. It is partly the opposite: twenty columns in one call guarantees
interacting pairs arrive in the same fit, and a tree can only find an interaction
if both halves are present. One-feature-at-a-time is cleanly attributable and
*structurally unable* to discover anything joint.

| | interactions | attribution |
|---|---|---|
| one feature at a time | invisible | clean |
| big batch | discoverable | unrecoverable |

## The decision

**Report credit per block** — the set of columns added by one tool call, already
recorded as `new_columns`, so this is a regrouping rather than new bookkeeping.

Why the block is the right unit:

- **It is the unit of decision.** The agent cannot add half a tool call. Its next
  move is "extend this table or go elsewhere", so credit per call matches an
  action it can take. Per-column numbers are finer than any available move and
  invite conclusions the measurement cannot support.
- **It is big enough to hold an interaction** — members arrive and are fitted
  together, so joint effects inside a block survive.
- **It is small enough to be one hypothesis** — if the agent makes it so.

Two fields ship with every training result:

```
cv: 0.5134 | vs_parent_table: 0.0
  original columns   raw        n=2 share=0.702 unused=0
  tables/t1.parquet  transform  n=2 share=0.298 unused=0   <- the date idea
  tables/t2.parquet  transform  n=2 share=0.000 unused=2   <- the squares idea
```

**`vs_parent_table` is `None` when the parent was never trained.** That is
deliberate, and it is how 0015's rejected option came back as an incentive: train
the table before extending it and you get a marginal number; skip it and you get
nothing. Rewarding the good path rather than mandating it — the same shape as the
advisory that moved the agent off an exhausted table without ever refusing it.

### The wording is load-bearing

Shipped with every result:

> a low or zero share means no evidence those columns helped *in this feature
> set* — not that they are useless: a feature that only pays off next to another
> scores nothing until that other one exists. Nothing is dropped for you.

Getting that sentence wrong would actively train the agent into the greedy
failure. Three rules follow:

- Zero importance is **"no evidence it helped here"**, never "useless". The
  strong claim is only safe in the other direction: a column that *was* split on
  did something.
- **Nothing auto-prunes.** No tool drops columns.
- Abandoning a branch is free and the agent is told so — tables are immutable
  (0006), so a "failed" table stays on disk and can be extended later.

### One call, one idea

Blocks are only useful if each block is one hypothesis. The live run's `t5` was
not: sixteen columns spanning dates, ratios, logs, squares and one-hots. So the
system prompt steers towards one idea per call, while keeping plausibly
interacting features together.

A **steer, never enforced.** Refusing a mixed transform would be wrong —
sometimes the kitchen sink is the right opening move precisely because it catches
interactions nobody thought of.

## Consequences

**Good**

- Verified: a date block scores `share=0.298`, a squares block `share=0.000`,
  `unused=2`, `vs_parent_table=0.0`. That is a legible next move, which one scalar
  never was.
- Costs nothing at runtime — the importances already exist on the fitted model.

**Bad**

- Split-based importance is **biased**: high-cardinality columns attract splits
  regardless of usefulness. Acceptable only because the alternative on offer is no
  feature-level feedback at all — and unacceptable to phrase as a verdict, hence
  the wording rules.
- A block is only meaningful if the agent keeps it coherent, and nothing enforces
  that. A mixed block yields a number nobody can act on.
- Still says nothing about interactions *across* blocks.

## Where it lives in the code

- `src/tabular_agent/harness/journal.py` — `blocks`.
- `src/tabular_agent/harness/tools.py` — `_block_report`, `_vs_parent`,
  `how_to_read_blocks`.
- `src/tabular_agent/harness/instructions.py` — the one-idea-per-call steer.
- `notebooks/phase3_harness_step4_stopping.ipynb` — §10.
