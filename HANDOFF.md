# Handoff

## Current state

Branch `simplify-v0.2` separates the autonomous product from its research lab.
The full pre-simplification system is tagged `research-reference-2026-08-21`.
Historical runners, probes, readings, notebooks and the closed handoff log are
preserved under `research/archive/`.

The active target is documented in `docs/CURRENT.md`:

- one train/evaluation split
- one bounded model contract
- cross-fitting only for leakage-safe fitted features
- baseline score, evaluation score and gain vs incumbent
- transparent table/time budgets; no runtime uncertainty gates

Implemented in shadow mode as `RunSession(..., profile="minimal")`: six tools,
one fixed LGBM fit, no terminal CV/OOF, bootstrap, block interval, sealed score or
blend. Regression, binary/string-label classification, non-zero-based multiclass,
ordered evaluation, serving-schema and bundle-replay fixtures pass.

## Next action

Run the minimal profile through two release candidates while `profile="full"`
remains available for comparison.
Compare safety, replay, fit count, wall time, token use and finalization rate —
not competition leaderboard movement.

## Blocking gotchas

- old bundles require legacy history/stream decoder compatibility
- archived scripts require tag `research-reference-2026-08-21` and original data paths
- the final minimal model choice must be judged by generic safety/replay fixtures,
  not competition leaderboard movement

## Uncommitted work

Ready to commit/push the repository reorganization, safe cuts and minimal profile
on `simplify-v0.2`. Final independent re-review was unavailable due provider HTTP
429; local targeted gates and the full suite passed.

## Latest session

2026-08-22, Hermes Linux: reduced the active surface to three prep scripts and one
START_HERE notebook; archived research by `git mv`; removed refuted/unreachable
runtime code; established the product/research split; added the six-tool minimal
profile and hardened serving schemas,
label mapping, resume invariants and crash-safe serialized finalization. No
accepted decision record was edited. Final active suite: **558 passed**, 1,588
known warnings, 10:10. START_HERE and the incumbent archive also smoke-tested.
