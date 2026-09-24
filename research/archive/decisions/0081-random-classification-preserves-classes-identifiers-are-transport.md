# 0081 — Random classification preserves classes; row identifiers are transport, not features

- **Date:** 2026-09-02
- **Status:** accepted and implemented
- **Related:** [0029](0029-the-harness-owns-the-exam.md) (the harness owns the
  evaluation protocol), [0030](../../../docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md)
  (the owner confirms the serving boundary), [0023](0023-the-third-split-has-evidence-now.md) (the sealed third
  split), [0022](../../../docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md) (uncertainty must match the evaluation design)

## What happened

The first run through the public facade on Titanic was operationally correct but
exposed two generic contract gaps.

First, a raw random permutation put materially different positive-class rates in
the three outer parts: 37.2% in work, 44.4% in validation, and 36.0% in the seal.
Accuracy changes when class composition changes. The sealed score therefore mixed
generalisation with a simpler class balance, even though its purpose is to be the
run's unconsulted judge.

Second, `PassengerId` was available in both train and serving data and so passed
the serving-schema guard. It became a model feature and received 21.8% of one
terminal model's feature importance, although it was only a row key and the
serving identifiers occupied a disjoint range. The existing contract answered
*whether a column will be available*, not *what role an available column has*.
That is not Titanic-specific: transaction IDs, claim IDs, and application IDs
have the same shape.

## The decision

1. **Random outer splits for binary and multiclass tasks are stratified by the
   target.** Validation is drawn first. If a seal is enabled, it is then drawn
   from the remainder. This keeps validation rows identical when a seal is
   switched on, preserving 0023's run-to-run comparability invariant. If a class
   is too rare for the requested parts, the split refuses with the class counts
   and requested sizes instead of silently dropping or distorting it.

2. **Ordered splits remain ordered and are never stratified.** Rebalancing a
   future period with rows from the past changes the forecasting question and can
   leak time. Their class composition is reported, not repaired.

3. **The public `DataPlan` has an owner-confirmed `identifier_columns` role.**
   Inspection may conservatively propose a column only when it is unique,
   non-null, available at serving, and has an ID-shaped name. A proposal is not a
   decision: the plan blocks until the owner accepts or rejects it. Confirmed
   identifiers remain available long enough to perform accepted static joins and
   to align predictions with source rows, then are removed before materialisation,
   feature generation, encoding, and model fitting. Bundle replay performs the
   same joins-before-removal sequence. Old bundles without this field replay as
   before.

4. **Classification reports show the class composition of work, validation, and
   seal.** For random accuracy splits they also show a Wilson row-level 95%
   interval. The wording is deliberately `row-level`: grouped observations are
   not independent, and the interval is not a claim about deployment drift.
   Ordered splits receive no binomial interval.

This narrows 0029 only at the role boundary. The agent still owns feature design;
the data owner decides that a column is transport metadata, after which the agent
must not be allowed to reinterpret it as signal. It extends 0030 from column
availability to an explicit serving-time role.

## What is rejected

- **Drop every unique or high-cardinality column automatically.** Continuous
  measurements and useful entity keys can be unique or nearly unique. The
  profiler proposes only the conjunction of name and values, and the owner still
  decides.
- **Drop every column whose name contains `id`.** Repeated keys such as `store_id`
  can be legitimate categorical signal. The proposal requires uniqueness; names
  such as `valid` are not ID-shaped.
- **Keep the identifier visible and ask the LLM not to use it.** A prompt is not a
  structural guarantee. The materialised table and fitted pipeline must never
  contain it.
- **Stratify an ordered split.** That trades temporal honesty for a tidier class
  histogram and evaluates a different task.
- **Add Titanic family/ticket rules or tune the 0.5 threshold from its public
  score.** Those are dataset-specific search decisions, not repairs to a generic
  harness contract.

## Cost and evidence

There is one additional owner decision only when inspection finds a plausible
identifier. Extremely rare classes can now make a requested random three-way
split fail where the old permutation produced misleading numbers. The split
journal grows by a small class-composition record.

Regression coverage proves that all three random classification parts preserve a
90/10 class distribution, regression remains unstratified, rare classes refuse,
identifier proposals do not match an unrelated unique `valid` column or a
repeated `store_id`, accepted identifiers are absent from the prepared table, and
a loaded bundle removes its identifier only after replaying static joins.
