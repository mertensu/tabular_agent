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

1. one train/evaluation split, plus a sealed audit slice (20% by default) scored
   once at finalize, so the report carries one number the search could not tune
   against (added 2026-10-26, `research/capacity_ship_minimal/`)
2. one bounded model contract for table ranking and shipping
3. cross-fitting only as leakage protection for fitted features
4. three visible numbers: baseline score, evaluation score, gain vs incumbent
5. transparent table/time budgets instead of uncertainty-based runtime gates
6. specialized forecasting and research diagnostics outside the default profile

The minimal agent exposes six tools: inspect, fitted encoding, fitted group
statistics, train-only row filtering, row-local transforms, and finalize. It
does not expose residual CV, journal exploration, history, streams or attribution.
The full profile remains available for compatibility during the observation
window.

## Current evidence

- the multi-geometry panel was reverted after accepting 1/24 IEEE steps
- random-subset feature selection measured approximately zero and is off-surface
- `0.25 × block_sd` changed none of 18 capacity-labelled incumbent decisions
- parent and global incumbent differed on 20/51 reconstructed steps
- ordered uncertainty thresholds remain unresolved; no tested multiple is suitable
  for runtime gating

## One next action

Run the minimal profile as default on two release candidates and compare safety,
replay, fit count, wall time, token use and finalization rate against the frozen
reference. Do not use competition leaderboard movement as the rollback gate.
