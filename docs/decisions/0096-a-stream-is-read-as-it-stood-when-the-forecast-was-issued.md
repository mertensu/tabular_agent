# 0096 — A stream is read as it stood when the forecast was issued

- **Date:** 2026-10-28
- **Status:** accepted and implemented (`encoders.fit_stream_aggregate`,
  `StreamAggregateEncoder.transform(..., issued_at=)`, `tools._streams_as_issued`,
  the stream step in `refit.refit_lineage`). Tests in `tests/test_stream_aggregate.py`
  and `tests/test_harness_stream_aggregate.py`.
- **Decided by:** Ulf
- **Revises:** [0062](../../research/archive/decisions/0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md)
  — its "known limit". The stream as a declared second input, the recipe-only
  encoder, the leads, the counts and the refusal of leads below the horizon stand.
- **Evidence:** `research/capacity_ship_minimal/` (README, "Restaurant r9" and
  `terminal_check.py`).

## The problem

A stream aggregate counts, per row, the records of a second input (bookings,
orders, claims) observed at least `lead` before the row. A forecast covers a
block of rows and is issued once, before the block's first row. A served row 30
periods into the block is predicted from the stream as it stood at the issue
time, so a one-period lead reaches it only through records that already existed
30 periods earlier.

Inside a run the stream file runs past every row. 0062 restricted each row by its
lead alone, so a short lead was as generous on the far edge of a block as on the
near one, in training and in validation. 0062 named this as a known limit and
left it, because closing it seemed to need a declared far edge.

It costs twice. The model trains on short-lead columns fuller than any served row
will get. And validation confirms them, because it is drawn from the same file:
in r9 a one-day-lead booking table gained 0.010 on validation and 0.004 on the
test days. That inflated score became the incumbent, every later table was judged
against it, and the stop rule ended the search.

## The decision

**Every row is computed from the stream as it stood when its forecast would have
been issued: the first row of its block minus the declared `forecast_horizon`.**
No new field: the block is one the run already has.

- **Training rows** get it per forward-chaining fold, the freeze `fit_history`
  already uses: each fold's validation block is cut at its own issue time. The
  first block has nothing before it to forecast from and keeps `NaN`, as history
  does.
- **Held-out rows** (validation, the holdout) are scored from the stream cut at
  the first scored row minus the horizon.
- **The refit for shipping** (0092) rebuilds the training values the same way.
- **Serving is unchanged.** The caller hands in the stream as it stands when it
  asks for a forecast, which is the cut this reproduces everywhere else.

A short lead stays useful: it adds records for the rows near the start of a
block, where a served row really has them, and for no others.

## What it costs

- **The first block loses its stream values.** A model sees `NaN` there and never
  at serving. History has the same gap and it has not been measured as costly.
- **A forecast issued once per block is assumed.** A served forecast refreshed
  every period would see more than this gives it. Then it is conservative, never
  optimistic.
- **Scores move.** Any table built with a lead shorter than a block now scores
  lower on validation than it did. That is the overstatement being removed, and
  runs before this record are not comparable on such tables.

## Rejected

- **Refuse leads shorter than the block**, i.e. declare the far edge. Honest, but
  it removes the near-edge records a served row does have; r9's one-day bookings
  were worth 0.004 on the test days.
- **Cut only validation.** Scores become honest while the model still trains on
  columns fuller than serving, so training and serving disagree on what the
  column means.
- **Let the agent choose the shipped table** to work around an inflated score. It
  fixes neither the search nor the stop rule, and in r9 the agent's choice would
  have been the worse table.
