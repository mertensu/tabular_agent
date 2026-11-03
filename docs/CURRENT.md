# Current state

One page only. Historical reasoning lives in `docs/decisions/`; completed
experiments live in `research/archive/`.

## Product today

`tabular_agent` owns a constrained PydanticAI loop. A run splits its input before
any tool executes, lets the agent build immutable candidate tables through safe
feature operations, ranks them with a fixed model, and packages a replayable
bundle from raw rows.

Core guarantees:

- validation rows are inaccessible to agent tools
- free-form feature code never receives the target
- batch-dependent transforms are refused
- fitted encoders use honest training values and serialize serving state
- unavailable serving columns cannot enter a shipped model
- every table carries replayable lineage
- the raw work table remains eligible at finalization

## Minimal profile implemented (`simplify-v0.2`)

Implemented shadow runtime contract:

1. one training/validation split, plus a test (holdout) set (20% by default)
   scored once at finalize, so the report carries one number the search could
   not tune against (added 2026-10-26, `research/capacity_ship_minimal/`)
2. the fixed probe ranks tables; what ships on the chosen table is the same
   family stopped early on training rows (0098)
3. cross-fitting only as leakage protection for fitted features
4. three visible numbers: baseline score, evaluation score, gain vs incumbent
5. transparent table/time budgets instead of uncertainty-based runtime gates
6. specialized forecasting and research diagnostics outside the default profile

The minimal agent exposes six tools: inspect, fitted encoding, fitted group
statistics, train-only row filtering, row-local transforms, and finalize. It
does not expose residual CV, journal exploration, history, streams or attribution.
`Agent` defaults to the minimal profile. `ForecastingAgent` runs the full profile
(streams, history windows, known-covariate windows) and has no holdout by
default; the forecast workflow cannot run on the minimal profile.

A run is bounded by 60 tables and 30 minutes, not by a streak without
improvement (0097). What ships is refitted on training + validation rows (0092).

## Current evidence

- forecasting runs (`research/capacity_ship_minimal/`): Rossmann r14 0.1186 rmspe
  on our test days and 0.1263 private as shipped (median 0.1297); Restaurant r10
  level with r8/r9, validation no longer optimistic after the stream cut (0096)
- on House and Porto the 200-round probe ranked tables only loosely like an
  early-stopped fit (Spearman 0.06–0.96), and CatBoost on the raw table matched
  what shipped

- the multi-geometry panel was reverted after accepting 1/24 IEEE steps
- random-subset feature selection measured approximately zero and is off-surface
- `0.25 × block_sd` changed none of 18 capacity-labelled incumbent decisions
- parent and global incumbent differed on 20/51 reconstructed steps
- ordered uncertainty thresholds remain unresolved; no tested multiple is suitable
  for runtime gating

## One next action

Re-rank Rossmann r14 and Restaurant r10 with early-stopped fits
(`research/capacity_ship_minimal/rerank.py`, prediction registered in its
README): does the 200-round probe pick the table a fitted model would, on the
forecasting datasets? Then decide whether `full` and `minimal` stay two profiles.
