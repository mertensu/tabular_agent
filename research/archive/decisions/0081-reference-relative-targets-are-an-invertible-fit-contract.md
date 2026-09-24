# 0081 — Reference-relative targets are an invertible fit contract

- **Date:** 2026-08-29
- **Status:** accepted
- **Related:** [0009](../../../docs/decisions/0009-target-hidden-from-transforms.md),
  [0021](0021-the-agent-picks-the-target-scale.md),
  [0029](0029-the-data-owner-declares-the-serving-contract.md),
  [0080](0080-revealed-target-history-is-a-bitemporal-stream.md)

## The problem

Forecasting models often learn a smaller and more stable quantity than the raw
target: the change from a serving-known reference, the target divided by a
serving-known scale, or both. Energy forecasting winners repeatedly used forms
such as `(target - target_48h) / installed_capacity`.

The agent cannot express that safely today. `target_transform` deliberately
contains only stateless functions of `y`; adding a column-dependent transform to
that registry would hide a serving-availability question inside a contract that
has none. Agent-written code cannot see the target under 0009, and must not gain
that access merely to construct another target.

## The decision

Add a per-experiment `ReferenceTargetSpec`, separate from `target_transform`:

```text
y_reference = (y - baseline_column) / scale_column
```

Either column is optional. A baseline alone is a residual target, a scale alone
is a ratio target, and both express a scaled residual. `None` is the existing
direct target. The inverse is exact and fixed:

```text
y = y_reference * scale_column + baseline_column
```

The harness, not agent code, applies both directions. Reference columns are read
from the selected feature table and therefore must be numeric, finite,
non-missing, and replayable for every scored or served row. A scale must also be
non-zero. The first version rejects invalid rows rather than silently clipping or
imputing a denominator; either policy would change the target definition and is
a separate measurable decision.

`target_transform` composes outside this contract. Fit applies reference-forward
first and then the stateless transform. Prediction inverts the stateless
transform first and reference-inverse second. Every fold metric, OOF value,
validation score, sealed score, explanation and bundle prediction remains in the
original target units.

The spec is part of the fit signature, experiment metadata, leaderboard and
bundle manifest. Two experiments using different reference targets are not a
feature-only comparison. Blend members must share one spec, because otherwise
their model outputs require different row-wise inverses.

## Safety boundary

- The public tool accepts column names, never code over `y`.
- A reference must already exist in the feature table. If it came from revealed
  target history, 0080's `observed_at <= origin` gate produced it; this contract
  does not bypass or duplicate that gate.
- The ordinary unservable-column and pipeline-replay checks cover both reference
  columns because they remain model inputs. Bundle prediction derives the same
  reference values from the replayed table before applying the inverse.
- Classification refuses a reference target. This is a regression capability.
- Old experiment metadata and bundles omit the field and load as direct targets.

## What this is not

- It is not a second target-history feature source. 0080 owns moving revealed
  history on the X side; this record owns fit/inverse on the y side.
- It is not a learned target transform, an arbitrary expression language, a
  conditional-expert router, or an online-update protocol.
- It does not declare a column serving-safe. The owner/prep contract and lineage
  guards do that; this feature only consumes columns which passed them.

