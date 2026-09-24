# 0047 — The forecast horizon is a column, not a constant

- **Date:** 2026-08-14
- **Status:** accepted, **not implemented** — this settles the two questions
  `docs/time_aware_features.md` §3.2 and §3.3 named as blocking, so that slice 2
  can be built. No code exists.
- **Related:** [0011](../../../docs/decisions/0011-stability-over-syntax.md) (the batch-independence
  property this decision spends nothing of, and the permissive alternative would
  have), [0030](../../../docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md) (the
  serving schema, which turns out to be a different axis), [0029](0029-the-harness-owns-the-exam.md)
  (task properties belong to the task's owner), [0046](0046-the-mean-is-a-lossy-summary-of-a-group.md)
  (slice 1, which needed neither of these answers), `docs/time_aware_features.md`
  (the full argument for why time-aware features are the gap at all).

## The problem, in one sentence

A lag feature needs history stored at fit time and replayed at predict time. At
predict time the served rows are **later than every stored row**, often by an
unknown gap — so "the value 7 days ago" is undefined for a row 60 days past the
end of training, and the harness has to decide what it means anyway.

## The two questions, and why they are one question

They were posed separately:

- **§3.2** — what does a lag serve when the predict rows are past the end of
  stored history?
- **§3.3** — may validation rows use their own past? At real serving time you
  often know yesterday's value; in a contiguous unlabelled future block you do
  not.

The answer to §3.3 collapsed them. The realistic forecasting contract is: you
have history up to `T`, and you are asked for `T+1 … T+k`. For `T+1` you *do*
know yesterday's value. For `T+2` you do not — the label for `T+1` has not
happened yet. So availability is not a property of the *set* of served rows; it
is a property of **each row's distance from the end of history**.

That is exactly what §3.2 was asking about. One mechanism answers both.

## The decision

### 1. Last-known value plus staleness, and staleness is per row

For a per-group history ending at `T`, a served row at `T+h` gets:

- the **last known value** for its group as of `T`, and
- **`h`** — how stale that value is, as its own column.

The row at `T+1` gets the last known value with staleness 1, which *is*
"yesterday's value". The row at `T+5` gets the same value with staleness 5. The
feature degrades across the horizon, visibly, in a column the model can read.

**Rejected: carry-forward alone.** Serving the last known value with no
staleness column is the same number with the honesty removed — silently wrong,
drifting further from the truth the further out you predict, and saying nothing
about it. This project's characteristic failure is a number that looks fine and
describes something else, and this is that failure in feature form.

**Rejected: refuse offsets entirely** (expanding/whole-history aggregates only).
Safe, and gives up most of the value. It is what we already have.

### 2. `forecast_horizon` in `TaskSpec` is optional, and is a *minimum*

When the task's owner knows the gap — the earliest `h` in the served block — they
declare it, and lag offsets at or beyond it become legal because they are
answerable from stored history for every served row. This is 0029's placement:
a property of the task, stated by whoever owns the task, not inferred by the
library from the shape of the data.

The refinement that came out of the discussion and is easy to get wrong: **it
cannot be one number that gates lags globally, because `h` varies per row within
a single predict batch.** A declared horizon is the floor across the block;
staleness remains a per-row column. A design that stored one horizon and treated
every served row as sitting at it would be wrong by construction for every row
but the first.

Without a declaration, staleness is what you get. The two compose; neither
replaces the other.

### 3. History is frozen at fit. No feedback, no reading the predict batch.

Every served row reads history as of `T`. It does not read other rows in its own
batch, and the model's own prediction for `T+1` is **not** appended to history
for `T+2`.

**Rejected: recursive forecasting** (predict `T+1`, append it, use it for `T+2`).
It is closer to how a real k-step forecaster works and it keeps lag-1 meaningful
throughout the horizon. It costs, in order of increasing seriousness:

1. Prediction becomes **sequential and order-dependent** — `predict_bundle`
   currently answers a frame in one pass and would have to answer it in time
   order instead.
2. Errors **compound** across the horizon, and nothing in the harness measures
   compounding.
3. **Train/serve skew, structurally**: training lags are true values, serving
   lags are the model's own predictions. The model would be fitted on one input
   distribution and served another, by design. This is the same shape of failure
   as 0030's unservable columns, one level subtler.

It is registered as rejected-for-now rather than refused: a later run with
evidence can reopen it, and it would be its own record.

**Rejected: letting predict rows read their own batch's past.** This is the one
that costs a guarantee. 0011's stability check exists because a feature whose
value depends on which other rows are in the frame gives a different answer to
one row alone than to that row in a batch. A lag reading the batch's own earlier
rows *is* that, exactly. Frozen history keeps batch-independence by construction
— the same property that makes `fit_group_stats` legal where
`groupby().transform()` is not (0032).

There is a real serving contract this refuses to express: a live system that
observes yesterday's label before predicting today's. That contract is worth
supporting eventually and needs its own declaration and its own record. It is not
worth acquiring by silently making every validation score in every future run
optimistic, which is what the permissive default would do on any task whose test
period is unlabelled.

## What was checked before deciding

**The serving schema (0030) does not already express §3.3.** It is a
*column-level* statement — `_unavailable_at_predict_time` answers "which raw
columns exist in a request". §3.3 is *temporal*: whether rows near the predicted
row are available with their labels. Note the target is excluded from
`serving_columns` by construction, so a lag of the target can never be derived
from a served frame at all; it can only come from stored history. That is what
makes §3.3 a question about whether stored history is ever *extended*, which is a
new declaration and not an extension of the existing one.

**0011's stability check passes a stored-history step as-is.** `unstable_columns`
only ever runs against sandbox `transform_features` code, so a structured
`history` lineage step bypasses it exactly as `encode` does — and, under this
decision, would pass it if it could see it. That equivalence is only true because
history is frozen; it is the concrete thing the permissive answer to §3.3 would
have spent.

## What is left open

- **Everything downstream.** The `history` step type, bounded storage (the last
  `L + W` observations per group), replay, and the leakage discipline for lagging
  the *target* specifically — a strictly backward-looking lag of the target is
  target encoding's near relative and needs the argument in
  `docs/time_aware_features.md` §3.4 settled or the out-of-fold path taken.
- **What `staleness` is measured in** when the ordering key is not a date.
- Nothing here has been validated by a run, and the estimate of what it is worth
  is one number on one dataset (`t8`, +0.0243 on the probe).
