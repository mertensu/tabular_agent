# 0048 — A target lag is not target encoding; the guard it needs is the horizon

- **Date:** 2026-08-14
- **Status:** accepted, **not implemented** — this settles the last question
  `docs/time_aware_features.md` left open (§3.4), so that slice 2 can be built in
  full rather than in its non-target half. No code exists.
- **Related:** [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md)
  (frozen history and per-row staleness at *predict* time; this record is the
  same answer relocated to *fit* time), [0032](0032-fitted-aggregates-and-a-step-that-is-not-replayed.md)
  (`fit_group_stats`, whose "folds are not optional" rule this deliberately does
  **not** inherit for the reason it was written), [0011](../../../docs/decisions/0011-stability-check.md)
  (batch-independence), [0043](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md)
  (ordered splits), [0046](0046-the-mean-is-a-lossy-summary-of-a-group.md)
  (slice 1), `docs/time_aware_features.md` §3.4.

## The problem, in one sentence

`docs/time_aware_features.md` §3.4 says a lag of the target "is target encoding's
near relative and needs the same care: out-of-fold construction, or an argument
for why strict backwardness is sufficient" — and that framing is wrong in a way
that would have produced the right mechanism for the wrong reason, and therefore
the wrong *scope*.

## Two failures wear the same word

"Leakage" is doing double duty here. Separated:

**1. Self-inclusion — the row's own target inside its own feature.**
This is what out-of-fold construction exists to remove. `GroupStatsEncoder`'s
docstring states it exactly: a column correlated with the target makes its
per-group mean nearly a per-group mean of the target, and computing it including
the row's own contribution is target encoding with the guard removed. Large
groups hide it rather than fix it.

**For a strictly backward lag this failure is impossible by construction.** The
row at `t` reads `t-k` for `k ≥ 1`. Its own target is never in the window, at any
group size, under any fold assignment. Out-of-fold buys exactly nothing against
it. The analogy to target encoding is false, and following it would have added a
fold pass whose stated purpose the mechanism does not serve.

**2. Availability skew — reading a label that will not exist when it is served.**
Real, severe, and *not* what an out-of-fold pass fixes. A validation row at `t`
reading the true value at `t-1` is reading a label that, in the serving contract
0047 settled, has not happened yet: history is frozen at `T`, and the row at
`T+2` gets the last known value at staleness 2, not yesterday's. A validation
score computed on true lag-1 measures a model that will never be served.

This is 0047 §3.3's "get this wrong in the permissive direction and every
validation score in every future run is inflated, invisibly" — the same failure,
one stage earlier. 0047 refused it at predict time. Refusing it there and
permitting it at fit time would be worse than permitting both, because the two
numbers would disagree and only the flattering one gets looked at.

## The decision

### 1. History is frozen per fold, at the end of the training fold

Validation rows read history the way served rows do: frozen, with a per-row
`staleness` counted from the freeze point. Within a validation block the feature
degrades exactly as it degrades across a real forecast horizon — the first row
gets something close to a true lag-1, the last row gets a value as stale as its
distance from the fold boundary.

The mechanism is fold-shaped, and so it will *look* like the out-of-fold path
`fit_group_stats` takes. It is not, and the difference matters when the next
person extends it: `fit_group_stats` iterates folds to exclude **the row's own
contribution**; this iterates folds to reproduce **the serving horizon**. A
change that preserved the first property while breaking the second — say, fitting
history on all rows outside the validation fold, including later ones — would
look correct by the older rule and be a straight future leak.

**The cost, stated plainly so it is not mistaken for a bug later: this makes the
feature look much weaker than a naive lag, and it should.** A naive backward lag
on Rossmann validates beautifully and cannot be served. The gap between the two
numbers is the size of the lie, not the size of the loss.

**Rejected: strict backwardness is sufficient at fit; freeze only at predict.**
Simpler, no coupling between the history step and the fold structure, and the
feature scores well. It is train/serve skew by construction — the model is fitted
against one input distribution and served another — which is 0030's failure and
0047's rejected recursion in a third costume.

**Rejected: freeze the target's history per fold, let non-target columns take the
strict-backward path.** This one is not obviously wrong and may come back. A
promo calendar or a holiday flag genuinely *is* knowable at `T+5`, and freezing
it throws away real information. But "known ahead of time" is a property of the
column that nothing in the harness can currently express, and inferring it from
whether a column happens to be the target is not that property — it is a proxy
that is wrong for every known future covariate and every observed-only one. It
needs its own declaration, in `TaskSpec`, with its own record. Until then 0047's
uniform freeze stands, and covariates pay for the missing concept.

### 2. Target history under random k-fold is refused

Under forward chaining, "history frozen at the end of the training fold" is a
prefix in time and `staleness` counts real elapsed periods. Under random k-fold
(`make_folds`'s default) the training fold is scattered through time: a
validation row's last known value comes from whichever of its neighbours happened
to land in the other fold, and `staleness` becomes an artifact of the seed. The
number is computable and means nothing, and it corresponds to no serving
contract.

So it is refused, as a return value (0013) naming the reason, rather than warned
about. A warning that does not block is a warning that gets ignored, and this is
the same call `fit_group_stats` made when it declined to let callers judge
per-column whether folds were needed — the judgement is one a caller has to get
right silently every time, and getting it wrong is invisible.

Note this is a constraint on the *task*, not a preference: a dataset where a
target lag is meaningful is a dataset whose rows are ordered, and `cv.py` already
says choosing wrong between the two fold strategies "is not a small error".

## What this costs

- **The first block is never validated.** Forward chaining already has this
  property; history inherits it. Rows in the first block get `NaN`, exactly as
  out-of-fold target encoding leaves them.
- **A whole family of tasks cannot use target history at all** — anything
  exchangeable. That is correct, and it is a smaller set than it looks, since a
  lag on exchangeable rows was never meaningful.
- **Known future covariates are under-served** until they get a declaration.
  Named above as rejected-for-now, not overlooked.

## What is left open

- What `staleness` is measured in when the ordering key is not a date — inherited
  unresolved from 0047.
- Whether the per-fold freeze should reuse the fold boundary or a separately
  declared horizon per fold. The boundary is the obvious choice and is what will
  be built; a task whose real serving gap is larger than one fold block would
  want the declaration instead, and `TaskSpec.forecast_horizon` (0047) is where
  it would go.
- Nothing here has been validated by a run.
