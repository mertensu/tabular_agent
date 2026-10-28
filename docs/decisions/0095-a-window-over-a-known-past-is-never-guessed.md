# 0095 — A window over a known covariate's past is never guessed

- **Date:** 2026-10-28
- **Status:** accepted and implemented (`FutureWindowEncoder._with_stored_past`,
  `known_past=` on `Bundle.predict` / `explain` / `feature_table`, the holdout
  replay in `tools._score_on_held_out`). Tests in `tests/test_future_window.py` and
  `tests/test_future_window_tool.py`.
- **Decided by:** Ulf
- **Revises:** [0093](0093-a-known-covariates-past-is-known-too.md) — what happens
  when the stored past does not reach the delivered rows. Its stored tail and its
  rule for using it stand.

## The problem

A trailing window over a column declared known in advance ("open days in the
last 14 days, per store") reads the periods before each row. 0093 let it read
back across the forecast start through a short tail of values stored at fit, used
only if it ends right before the delivered rows. When it does not, 0093's encoder
computed the window over the delivered rows alone.

That is a different feature. On the first forecast day "open days in the last 14"
is 1, on the second 2, and so on: to the model it reads as a store that was
closed for two weeks, and stores sell more after reopening. The model never saw
this in training, where the window was always full.

It happened the first time a bundle forecast a period that did not start right
after its data. Rossmann r12's shipped bundle (data to 13 June) asked for Kaggle's
test (from 1 August) predicted about 2× too high on 1 August, back to normal by
mid-August: Kaggle public 0.479, against 0.107 for the same recipe refitted up
to 31 July. The run's own test days start the day after its data, so nothing in
the run could see it. A gap between the end of the data and the start of a
forecast is ordinary in use (data arrives late, models are retrained weekly).

The product had the same gap inside it: finalize scores the training-only fits
on the holdout slice, and the validation period lies between their data and the
holdout. Those trailing windows were silently partial too.

## The decision

**A trailing window is computed only from a past that reaches the first
delivered row. Otherwise the bundle refuses, and the caller can supply that past
as `known_past`.**

- `known_past` is earlier rows of the declared future-known columns: series key,
  ordering column, the columns themselves, no target. Only future-covariate
  windows read it, only for periods before each series' first delivered row, and
  in preference to the stored tail. Nothing is predicted for its rows.
- The refusal names the column, how many rows each series needs, where the
  bundle's record ends and where the forecast starts, and both ways out: pass
  `known_past`, or refit on rows that reach the forecast start.
- A series with nothing recorded before its first delivered row is at its start,
  as it was in training, and is not refused (a replay of the training rows, a
  series that did not exist at fit).
- Finalize's holdout replay passes the validation rows as `known_past`. Those are
  serving columns declared known before and after the forecast start, so the
  training-only fits are now scored on the windows they were trained on.
- It applies to every encoder that reads back across the block start (0093
  onwards), including bundles already on disk: on a gap they were wrong silently
  and now refuse. Encoders written before 0093 reset at every block start, were
  trained that way, and are unchanged.

## Rejected

- **Missing values instead of partial windows.** Measured on r12's bundle: worse
  (about 1.4–1.8× too high for 14 days). The model only saw a missing window on
  the first days of the whole dataset, so the trees send it down an arbitrary
  branch. A missing value is honest only if the model learned what it means.
- **Partial windows (0093's behaviour).** The feature above.
- **Filling the gap from the stored tail anyway.** Those values are not the
  window's predecessors.

## What it costs

- A caller forecasting after a gap has to pass `known_past` or refit. For a
  known-in-advance column that is the calendar or plan they already use to fill
  the forecast rows.
- It covers the future-covariate windows only. Target history (`fit_history`) is
  frozen at fit and grows stale over a gap in a different way; that is not
  addressed here.
