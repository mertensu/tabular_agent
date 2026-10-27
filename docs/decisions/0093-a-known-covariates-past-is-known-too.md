# 0093 — A known covariate's past is known too

- **Date:** 2026-10-27
- **Status:** accepted and implemented (`FutureWindowEncoder` stores a tail;
  trailing windows reach back across the block start). Tests in
  `tests/test_future_window.py`.
- **Decided by:** Ulf
- **Revises:** [0091](../../research/archive/decisions/0091-future-known-covariates-are-a-declared-class.md)
  — its trailing-window rule and "nothing is stored"; its declaration and its
  forward rules stand.

## The problem

0091's tool aggregates a column declared known in advance over the forecast
block: holidays, promotions, a weather forecast. Every window was cut at the start
of the block, because at prediction time the model is handed only the block's rows
and the encoder stored nothing from before them.

For looking *forward* that cut is right: nothing after the block end is delivered.
For looking *backward* it is not. "School-holiday days in the last 7 days" on the
first day of a forecast needs the 6 days before the block. Those values are in the
training data; they are history. With the cut, the feature means "holiday days
since the forecast started" for the first week of every block. In training it was
worse: blocks are tiled every horizon back from the end of the series, so the
window dropped to one row at dates with no meaning.

`fit_history` cannot fill the gap either. It is built for the target, which is
unknown after the end of training, so it only allows windows that end at least one
horizon before the row. A column known in advance needs neither restriction.

Found on Rossmann r2, where the winner's "holidays last week" and "days since the
store reopened" were the features in question. The gap is not Rossmann's: any
forecast whose known covariates have a recent past that matters hits it.

## The decision

**Trailing windows of a known-in-advance column reach back across the block start.**

- In training they read the rows before them in the frame, across block starts;
  only a series' very first rows have nothing before them.
- At fit the encoder stores each series' last `window − 1` values. At prediction
  time, when a block arrives alone, those stored values sit in front of it, so the
  first row's window is full, exactly as in training.
- A stored tail is used only if it ends right before the delivered rows (within
  1.5 typical steps). Older values are not the window's predecessors; then the
  window starts with the delivered rows.
- Forward and whole-period windows are unchanged: they stop at the block end,
  because that is where the delivered data ends.
- Encoders written before this keep resetting, so a bundle on disk predicts what
  its model was trained against.

## What was rejected

- **Let `fit_history` take known covariates without its horizon gap.** That mixes
  two contracts in one tool: the gap is the whole safety rule for the target, and
  a tool that drops it on a flag is one wrong declaration from a leak.
- **Store the whole covariate history.** A trailing window needs `window − 1`
  values; anything more is state with no reader.

## What it costs

- The encoder now carries state, so its training values and its stored tail must
  come from the same fit; refitting before shipping (0092) refits both.
- A gap between the stored tail and the delivered rows silently falls back to a
  shorter window. That matches what the rows can support but is not reported.

## Also in this change

The window encoders (future-window, history, stream aggregates) read their value
through `pd.to_numeric(errors="coerce")`, so a text column did not fail: it became
missing wherever a value was not a number. `StateHoliday` ('0', 'a', 'b', 'c')
would have kept the '0's and lost every holiday. They now refuse a non-numeric
value with a message that says what to pass.
