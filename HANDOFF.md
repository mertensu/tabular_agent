# Handoff

## Latest session (2026-09-10)

Merged the public reusable-feature contract from `chad/feature-table-api` into
`master`, alongside the previously merged reference-relative target contract.
The two optional bundle/scoring contracts are preserved together:

- `Bundle.describe_features()` returns one readable lineage row per shipped
  feature: kind, source columns, pipeline step, explanation, and stored recipe
- `Bundle.feature_table(new_rows)` returns the exact ordered shared model matrix;
  `include_input=True` appends it to the original rows for IDs and other metadata
- `Run.training_features()` returns the exact selected table fitted by the
  shipped models, preserving OOF target/group-statistic values and optionally
  appending the target
- new bundle manifests retain the run-local selected-table reference; old bundles
  remain prediction- and feature-replay-compatible but cannot recover an exact
  OOF training table
- row-local transforms now retain their output names as explanation-only metadata;
  replay still executes the stored sandbox code

Decision record: `docs/decisions/0083-learned-features-are-a-reusable-artifact.md`.
The earlier handoff is archived verbatim under
`research/archive/handoffs/2026-09-01--2026-09-10.md`.

## Verification so far

- combined API/pipeline/blend/model/finalization/reference-target tests pass
- complete repository suite on the final merged master commit: 592 tests pass
- independent review on the feature branch: 585 tests pass from a fresh checkout
- wheel and sdist build successfully
- a real legacy Titanic bundle describes all 14 shipped features; legacy
  row-transform outputs are explicitly marked unattributed rather than guessed
- provider-free public-facade E2E proves finalization records the selected table
  and `training_features()` returns its model columns plus target


## Next action

Push and review merged `master`. A natural next acceptance check is a fresh
full-profile run whose winning table includes both agent-written columns and a
target-informed encoder, then fitting a small user-owned estimator from
`training_features()` and predicting via `bundle.feature_table()`.

## Blocking gotchas

- `Bundle.feature_table()` is for genuinely new/serving rows. Custom training
  must start from `Run.training_features()`; replaying final target-informed
  lookups over their own fit rows is leakage and differs from the OOF matrix.
- estimator-specific preprocessing is deliberately not part of the shared table;
  a blend can contain several different imputers/scalers.
- `Run.training_features()` depends on the original run directory. A copied
  standalone bundle continues to describe and transform new rows, but does not
  carry the potentially large training table.
