# 0037 — Residuals need folds; the ranking does not

- **Date:** 2026-08-13
- **Status:** accepted and implemented
- **Related:** [0035](../../../docs/decisions/0035-the-search-is-over-features.md) (which removed the
  tool this replaces), [0036](0036-the-pairing-asymmetry-was-not-there.md),
  [0009](../../../docs/decisions/0009-target-hidden-from-transforms.md), [0019](0019-the-leaderboard-is-a-public-leaderboard.md)

## The problem 0035 created and did not notice

0035 took the model tools away from the agent. `explore_experiment` — the tool
that shows an agent *where its model is wrong* — takes an `exp_id`, and after
0035 no experiment exists until `finalize_run`. Removing one tool silently
removed another.

That would have been a real loss disguised as a refactor. `explore_experiment`
closes the only arc of the loop that turns a number back into an idea: without
it an agent sees one score per table and has to guess the next feature from
column names. The record that introduced it put it plainly — the step from a
score back to the next hypothesis had nothing to work with.

A feature-only search makes that arc *more* important, not less. It is now the
only mechanism the agent has for generating hypotheses at all.

## The decision

**`explore_table(table, code)`** — same read-only frame, sourced from a
cross-validated fit of the probe on that table, run when asked for.

**And cross-validation stays off the ranking path.** The two needs are genuinely
different, which is the whole content of this record:

| | needs | costs |
|---|---|---|
| ranking a table | a score on the validation half, from a model fitted on all the table's rows | **one fit** |
| diagnosing a table | an honest prediction for *each row*, i.e. out-of-fold | **`n_splits` fits** |

The ranking number does not improve with more folds because folds never enter
it. Residuals cannot be had without them. So the asymmetry is not an
inconsistency to be tidied up later, and this record exists mostly to stop
someone tidying it up.

## Why the obvious shortcut is wrong

The tempting optimisation is to keep the residuals from the fit the ranking
already did — it is sitting right there, one fit, already paid for.

**Those are in-sample residuals, and they point at nothing.** A gradient-boosted
tree with 200 rounds fits its training rows very closely; the residuals it leaves
behind describe what it failed to memorise, not where it fails to generalise. An
agent grouping them by city would find noise and build features against it. The
whole value of a residual here is that it comes from a model which did *not* see
that row, and the only way to get one for every row is to hold each row out in
turn.

Cheaper-looking variants fail the same way or worse:

**Take residuals from the validation half.** They would be honest, and it is the
worst option on the list. The validation half is what tables are ranked on
(0019). An agent that reads per-row errors there and writes features in response
is fitting features to the ranking set directly — not through the slow channel of
selection that 0034 describes, but immediately and by hand. `explore_table` reads
the **work half only**, and that is a boundary, not a default.

**Hold out one slice of the work half instead of rotating folds.** Honest, and
the diagnosis is then based on 1/k of the rows. The question being asked —
*which segment is this failing on* — is exactly the one that needs enough rows
per segment to see anything.

## What it costs

`n_splits` fits per call, charged to no budget and refused by nothing. That is
deliberate: understanding why a table failed is most valuable precisely when the
run is going badly, and a diagnostic that gets rationed when things go wrong is a
diagnostic that is absent when it is needed.

It is not free in wall clock, and on a large dataset it is the most expensive
thing an agent can ask for. The tool says so; nothing enforces it. If that turns
out to be the wrong call, the evidence will be a run that spent its wall clock
diagnosing instead of building, and that is a measurement we do not have.

## What is unchanged

The guard is the same one `explore_experiment` had, and it is the absence of a
persistence path rather than the absence of a number: the frame carries the
target and the residual, the result is shown and discarded, and nothing computed
inside it can become a feature except by going back through `fit_encoding`, which
fits per fold. The frame is still the **whole** table including the string
columns a model cannot consume — those are exactly the ones worth grouping
residuals by, and they cost nothing here because nothing here becomes a feature.
