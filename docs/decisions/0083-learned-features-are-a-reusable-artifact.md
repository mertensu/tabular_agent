# 0083 — Learned features are a reusable artifact

- **Date:** 2026-09-10
- **Status:** accepted and implemented
- **Related:** [0006](0006-lineage-per-table.md) (lineage belongs to each
  table), [0010](0010-pipeline-is-the-deployable-artifact.md) (the fitted
  pipeline ships), [0030](0030-a-model-that-cannot-be-served-is-not-trained.md)
  (training and serving share a contract), [0081](0081-random-classification-preserves-classes-identifiers-are-transport.md)
  (identifiers are transport metadata)

## What happened

The bundle could replay its feature pipeline only as an implementation detail of
`predict()`. A user could neither inspect what the agent learned nor reuse the
same matrix with another estimator. That makes the model reusable but traps the
more general artifact: the fitted feature pipeline.

A naive export has a serious failure mode. The stored target and group-statistic
encoders are final serving lookups fitted on all work rows, while the model was
trained on out-of-fold values. Replaying those final lookups over the same
training rows would silently introduce self-information and hand a custom model
a leaked matrix.

## The decision

1. **`Bundle.describe_features()` returns one row per shipped model column.** It
   names the feature kind, source columns, pipeline step, readable explanation,
   and stored recipe. Fitted lookup mappings are not dumped. Agent-written
   transforms record their output-column names as explanation metadata; replay
   continues to execute the stored code and never trusts that metadata.

2. **`Bundle.feature_table(rows)` is the serving/new-row transform.** It performs
   the same static joins, datetime normalisation, identifier removal, fitted
   encoders, stream checks, and column ordering as `predict()`, but stops before
   any estimator-specific imputer or scaler. `include_input=True` appends those
   exact model columns to a copy of the caller's original table so row IDs and
   other raw fields remain available.

3. **`Run.training_features()` is the custom-model training transform.** New
   bundles record the run-local table actually fitted by the shipped models.
   The method reads its exact ordered model columns and optionally the target,
   preserving the out-of-fold values already materialised by the harness.
   Users pair this with `run.bundle.feature_table(new_rows)`; they do not replay
   final target-informed encoders over their own training rows.

4. **Targets and identifiers never enter bundle replay.** They may remain in the
   `include_input=True` result for alignment or fitting, but are removed before
   agent-written transformations and encoders run. Old bundles remain valid:
   they can describe and transform new rows, but refuse `training_features()`
   because they did not record the exact fitted table.

## What is rejected

- **Materialise one final-encoder table and call it training data.** For
  target-informed encoders it is not the matrix the model saw and can leak.
- **Return each estimator's internal transformed matrix.** A blend can contain
  several incompatible preprocessors; those parameters belong to the estimator,
  not to the shared learned-feature artifact.
- **Recompute batch aggregates from the caller's rows.** The pipeline replays
  only stored fitted recipes, retaining the existing batch-stability contract.
- **Infer old transform provenance.** Legacy manifests did not name which code
  step introduced which output. The description reports that ambiguity instead
  of guessing.

## Cost and evidence

New manifests gain one run-local table reference and row-local transform steps
gain output-name metadata. The bundle itself remains portable for prediction;
only `Run.training_features()` depends on the retained run directory. Regression
tests cover exact ordering, identifiers/targets kept only in the enriched outer
table, readable transform and encoder provenance, and direct retrieval of an
OOF-safe fitted table rather than final-encoder replay.

