# Handoff

## Latest session (2026-09-17)

Merged capability-review V1 to master first, then built the explicit forecasting
surface on branch `chad/forecasting-agent`.

- `Agent` is now the public IID contract: it requires `split_on = False` and
  cannot call `fit_history` or `fit_stream_aggregate`
- `ForecastingAgent` is a separate public surface over the same engine: it
  requires a confirmed ordering column and positive `forecast_horizon`
- forecast runs receive a focused temporal brief which asks them to test honest
  history when a repeated series is expressible, while ordinary interactions
  and row-local transforms remain common to both workflows
- workflow and horizon are persisted in the split contract, checked on resume,
  visible in Markdown/JSON/HTML, and available to capability review
- the guided DataPlan summary now surfaces the horizon when an ordered split is
  selected
- the Quickstart documents the explicit choice; no automatic mode switching was
  added

Decision 0089 records why these are two product contracts but one implementation,
and preregisters the repeated Rossmann acceptance test as a variance test rather
than a best-score contest.

## Verification so far

- focused API/tool-surface/prompt/report/reviewer suite passes
- full repository suite: 626 tests collected and all pass
- Wheel and sdist build successfully with `uv build`
- `git diff --check` and compile checks pass
- the real-run acceptance is intentionally still pending: three predeclared
  Rossmann runs with Requesty `deepseek-v4-flash-0731@eu`

## Next action

Commit and push the implementation, then run exactly three Rossmann repetitions.
For each, report whether at least one valid history family was attempted, its
best internal validation/test score, feature count, and failure/refusal state.
Report the distribution; do not select the best run.

## Blocking gotchas

- Requesty model strings used as public API strings normally need a provider/model
  slash to trigger gateway routing; for the exact requested bare model ID, pass a
  preconfigured `gateway_model("deepseek-v4-flash-0731@eu")`
- the forecast brief requires a history attempt only when an honest repeated
  series is expressible; it must not fabricate an entity key on a single-series
  or non-repeated dataset
- low-level `RunSession(workflow="general")` remains backward compatible for
  research scripts; do not weaken the strict public facade contracts to remove it
- do not interpret a single improved repeat as acceptance: the motivating defect
  was run-to-run omission of history
