# 0017 — Credit is measured by refitting without the columns, not by importance

- **Date:** 2026-08-06
- **Status:** accepted
- **Supersedes:** [0016](0016-block-level-credit.md) (the mechanism; its unit of
  credit survives)
- **Decided by:** Ulf (the scenario that exposed the flaw, and the instinct that
  importance was the wrong instrument)
- **Related:** [0015](0015-report-attribution-gaps.md)

## The problem

Decision 0016 gave the agent per-tool-call feedback using the fitted model's
`feature_importances_`, normalised into an `importance_share` per block. It shipped
with a paragraph of hedging, because split-count importance is biased toward
high-cardinality columns and everyone knew it.

A run made the gap concrete. Four original columns (`price`, `area`, `rooms`,
`year_built`) held **77%** of the splitting and contributed nothing — they tied the
baseline. Three date columns held **23%** and carried the entire 0.19 improvement.
`price` is continuous and offers hundreds of thresholds, so a tree will always find
somewhere to spend a split; `is_weekend` offers exactly one.

Then Ulf put the real question:

> imagine the model creates features K but does not fit a model. now it creates
> another one on top, say feature l, and fits a model and its better than the best
> one before. could be feature l contribution or the one from the other K features
> before

## Why no amount of tuning saves importance

The two quantities answer different questions, and the difference is not a matter
of picking a better importance variant:

- **Importance** decomposes *one fitted model's attention*. It is descriptive.
- **The question** is a counterfactual: what would the score be if these columns
  had never been built.

They come apart routinely and in both directions. A column the trees split on
constantly can be worth nothing once removed, because a correlated column carries
the same signal. A binary column offering a single split point can carry the whole
gain. A perfect, unbiased importance measure would still not answer K-vs-L, because
"the model looks here" and "the model would be worse without this" are simply not
the same claim.

The hedging in `how_to_read_blocks` was the tell. A number that needs a paragraph
explaining what it does not mean is a number reporting the wrong thing.

## The decision

**Measure the counterfactual directly.** Fit once on everything, then once more per
contributing table with that table's columns withheld — same folds, same model,
same params. `contribution` is the full score minus the score without them, signed
as a gain.

Exposed as a separate tool, `explain_table`, not as a field on every training
result. It costs a fit per contributing table, so it is asked for when the answer
changes the next move — and paying that cost on every training call would be a tax
on the common case to serve the occasional question.

Two consequences the importance version had to hedge about in prose disappear:

- **Interaction partners both get credit.** If K is worthless alone but K+L is
  strong, withholding either one costs the score, so both measure positive. This is
  exactly the objection that shaped 0016, and ablation answers it structurally
  rather than with a warning label.

  Verified on a pure XOR whose halves were added in *separate* tool calls: each
  scores `+0.82`. This **retires 0016's advice to keep interacting features in the
  same call**. That advice existed because per-column importance would score a
  lone half at zero and invite the agent to abandon it before its partner arrived;
  ablation cannot make that mistake, so the guidance is now simply one idea per
  call, plus a warning not to read a lone half's zero as disproof.
- **Redundancy reads as ~0 honestly.** Two tables encoding the same signal each
  measure as free to remove. That is the truth about each one *given the other*; the
  pair still has to be tested as a pair.

### The same rule applied to `vs_parent_table`

The identical flaw was sitting one field over. `vs_parent_table` compared against
the *best* score on the source table, whatever model produced it — so training lgbm
on the source and catboost on its child credited a model swap to the columns.

Renamed `vs_source_table` (see [0018](../../../docs/decisions/0018-the-payload-is-a-budget.md)) and
restricted to an identical `(model_type, params)` fit. One rule now covers both
numbers: **a comparison must differ by exactly one thing, or it is not evidence
about that thing.**

It is `null` more often as a result. That is the trade, and it is affordable only
because `explain_table` now covers the gap — it refits everything itself, so it
works even when the agent changed model and features at once, and even on columns
that were never trained separately. Which is to say: it answers Ulf's K-then-L
scenario without requiring the intermediate fit that 0015 could only incentivise.

## What was rejected

- **A better importance variant** (gain-based, permutation). Permutation importance
  is a counterfactual about *the fitted model*, not about *the feature set* — it
  cannot tell you that a retrained model would recover via a correlated column.
  Gain-based importance is differently biased, not unbiased. Neither answers K-vs-L.
- **Shapley over blocks.** Correct, and exponential. 0016 already rejected exact
  fair credit as the wrong tool for search guidance; that reasoning stands.
- **Ablation on every training call.** `n_blocks + 1` fits per experiment, to answer
  a question the agent asks occasionally.
- **Reusing a stored leaderboard score as the full-fit reference.** Saves one fit,
  but a stored score may come from other params — and a difference of protocol would
  land in `contribution` looking exactly like a difference of features. The bug this
  whole record is about, reintroduced to save 30 seconds.

## What it costs

**Good**

- Verified on the motivating case: the date columns measure `+1.16`, and the four
  original columns measure `-0.06` — withholding them slightly *improves* the score.
  Importance said they held 77% of the attention. Both were true; only one was the
  question.
- No caveat paragraph. The measure means what its name says, so the explanation
  moved to the system prompt and shrank.
- Costs no experiment and no budget, and is allowed on a table the stop rule has
  already declined — which is exactly when "what happened here" is worth asking.

**Bad**

- `n + 1` cross-validated fits. Bounded by the number of tables in a lineage, but
  real, and on a slow dataset it is the most expensive thing the agent can do.
- **Leave-one-out understates every member of a redundant group.** Two copies of the
  same signal each score ~0, and an agent reading only those numbers could conclude
  neither matters. `explain_table` reports each table *given the others* and says so;
  it does not test subsets.
- `vs_source_table` is `null` in more situations than `vs_parent_table` was, and an
  agent that changes model and features together never sees it at all.
- Still says nothing about interactions *across* tables — inherited from 0016 and
  not addressed here.

## Where it lives in the code

- `src/tabular_agent/harness/tools.py` — `explain_table`, `_columns_by_table`,
  `_vs_source_table`, `_gain`.
- `src/tabular_agent/harness/journal.py` — `columns_by_table`, `update_table`.
- `src/tabular_agent/harness/instructions.py` — the three comparisons.
- `tests/test_harness_tools.py` — `test_credit_is_a_refit_without_the_columns_not_an_importance`,
  `test_vs_source_table_compares_the_same_fit_only`.
