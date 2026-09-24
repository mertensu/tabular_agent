# 0089  Forecasting is an explicit public contract, not an inferred mode

- **Date:** 2026-09-17
- **Status:** accepted and implemented
- **Related:** [0029](../../research/archive/decisions/0029-the-harness-owns-the-exam.md) (the harness owns
  evaluation), [0060](../../research/archive/decisions/0060-a-forecast-is-not-an-iid-table-with-a-date-column.md)
  (ordered evidence), [0062](../../research/archive/decisions/0062-a-stream-is-a-declared-second-input-not-a-hidden-join.md)
  (observation-time-safe streams), [0088](0088-a-run-may-propose-a-capability-gap-but-not-change-the-product.md)
  (agent gaps are not capability gaps)

## What happened

Two otherwise equivalent Rossmann runs diverged sharply. One used trailing sales
history and reached a private score near 0.126; the other never called
`fit_history` and reached about 0.167. The tool existed and worked, so this was
not a missing capability. A single broad tool vocabulary made it too easy for the
LLM to treat a forecast as ordinary row-wise tabular learning.

The user normally knows whether the product must predict a future quantity. That
is part of the use case, not a subtle property to infer from column names.

## The decision

The public API exposes two explicit contracts over one implementation:

- `Agent` handles independent rows, requires a confirmed random split, and does
  not expose history or observation-time stream tools.
- `ForecastingAgent` handles prediction into the future, requires a confirmed
  ordering column and positive forecast horizon, and exposes the time-aware
  tools with a forecast-specific brief.

Neither silently changes into the other. The session persists `workflow` and
`forecast_horizon` and refuses to resume under a different temporal contract.
Markdown, JSON, HTML, and capability-review evidence carry the same facts.

The split does not fork the engine. Inspection, DataPlan, encoders, sandbox,
models, journal, bundle, reports, leakage primitives, and finalization remain
shared. Row-local transforms and categorical interactions remain available to
both agents because interactions are not forecast-specific. Only information
that crosses rows needs the temporal contract.

The forecast brief asks the model to identify repeated series and, when honest
history is expressible, measure at least one trailing-history family before
finalizing. It does not assume history must win and does not invent a series key
when none exists.

## Acceptance test

The motivating claim is variance reduction, not one lucky score. On three to five
predeclared Rossmann runs with the same economical Requesty model, each forecast
run should test at least one valid history family and the distribution should no
longer contain the previous no-history failure mode. All runs are reported; the
best run is not selected as the product result.

## What was rejected

- **Infer forecast mode from a date-looking column.** Dates also occur in IID
  problems, and a wrong silent switch changes both evaluation and leakage rules.
- **Create a second modeling engine.** Two implementations would drift in safety,
  persistence, and serving behavior.
- **Make interactions forecast-only.** Store-by-weekday or category-by-promotion
  interactions are useful in both IID and temporal problems.
- **Add only a generic prompt nudge.** It would leave the public contract and tool
  surface ambiguous, which is the source of the observed variance.
- **Force every forecast to ship history.** Some datasets have no repeated series,
  and measured history can lose. The obligation is to test an honest hypothesis
  when it is expressible, not to keep it.

## What it cost

One public class, one persisted workflow field, a smaller IID tool surface, and a
forecast-specific prompt. Low-level `RunSession` retains a `general` workflow
for backward-compatible research use; the two public facades are strict.
