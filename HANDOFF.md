# Handoff

## Current state

Branch `chad/generic-classification-contract` adds the two generic contracts
exposed by the Titanic facade acceptance:

- random binary/multiclass outer splits are stratified; ordered and regression
  splits retain their existing behavior
- `DataPlan` proposes plausible unique row identifiers but blocks until the owner
  accepts or rejects the role; confirmed identifiers are usable for static joins
  and result alignment, then are removed before the agent/model and during bundle
  replay
- split journals and both full/minimal reports show class composition; random
  accuracy reports add a Wilson row-level 95% interval

Decision record 0081 is accepted and implemented. No Titanic-specific family,
ticket, public-label, or threshold rule entered `src/`.

## Verification so far

- directly affected modules: 51 tests passed after final self-review
- full repository suite before the final Minimal-report wiring: 582 passed in 5:47
- provider-free live Titanic facade replay: `PassengerId` proposed and excluded
  from the prepared table, 418/418 finite bundle predictions, report names
  stratification, target composition, and the row-level interval
- wheel and sdist build successfully

The full suite must be rerun after the final commit and attributed to that exact
HEAD before the branch is handed off.

## Next action

Commit the reviewed work, run the complete suite on that exact commit, push the
branch, and report the commit plus verification. Do not submit Titanic again;
its public score has already been consulted and 0081 explicitly rejects fitting
new rules to it.

## Blocking gotchas

- a class too rare for the requested random parts now refuses rather than yielding
  a misleading split; the error includes class counts and part sizes
- `identifier_columns` is an owner role, not an automatic drop rule; repeated
  entity keys such as `store_id` remain ordinary candidate features
- old bundles without `identifier_columns` remain backward compatible
- ordered accuracy reports intentionally omit the binomial interval because rows
  across time are not exchangeable

## Latest session

2026-09-02, Chad: implemented 0081, corrected the Minimal-profile report path
found by the live Titanic replay, updated quickstart and both facade acceptance
runners, and removed stale equal-weight wording left behind by 0080.
