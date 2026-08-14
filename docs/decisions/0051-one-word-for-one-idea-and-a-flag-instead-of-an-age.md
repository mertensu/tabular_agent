# 0051 — One word for one idea, and a flag instead of an age

- **Date:** 2026-08-14
- **Status:** accepted and implemented. Renames `HistoryEncoder`'s output columns,
  replaces the per-offset age with a `_exact` flag, and makes `fit_history` report
  the block span and each offset's answerable share.
- **Related:** [0050](0050-the-anchor-is-what-matters-not-the-history.md) (which
  named the missing report and measured the `lag7` failure this is meant to make
  visible), [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md) (the
  floor, unchanged; this adds the ceiling as a *report* rather than a rule),
  [0049](0049-the-last-observation-is-a-sample-of-size-one.md),
  [0029](0029-the-harness-owns-the-exam.md) (why the span is reported and not
  enforced), `src/tabular_agent/encoders.py`,
  `src/tabular_agent/harness/tools.py`.

## The problem

Slice 2 shipped four column names that had accumulated rather than been chosen:

    <base>_last          <base>_last_age
    <base>_lag364        <base>_lag364_age

Three separate complaints, and only the third is about naming at all.

**`_last` is ambiguous in the one way that matters here.** It reads as "final" or
as "previous" depending on the reader, and 0050 had just established that *what
the value is anchored to* is the whole story. A word that does not distinguish
"most recent" from "one step back" is the wrong word for this column.

**`_age` and `_lag` say the same thing in two vocabularies.** `_lag364` and
`_last_age` are both distances backward in time, named as if they were different
kinds of quantity, and a reader has to learn both to read one table.

**The offset's age column carries almost no information.** `lag364_age` *equals*
364 whenever the lookup succeeded. It varies only when the lookup fell further
back — so the column is a constant with a few exceptions, spending a full float
column and a name on the exceptions.

## The decision

| was | now |
|---|---|
| `<base>_last` | `<base>_latest` |
| `<base>_last_age` | `<base>_latest_ago` |
| `<base>_lag364` | `<base>_364_ago` |
| `<base>_lag364_age` | **`<base>_364_ago_exact`** |

**One word for one idea: `_ago`.** Both the offset and the measured distance are
"how far back", so both use the same word. In `<base>_364_ago` the distance is a
literal in the name; in `<base>_latest_ago` it is unknown until you look, so it
becomes a column. Same word, same meaning — one fixed, one measured. A reader who
learns `_ago` once can read every column the tool emits.

**The offset's age becomes a 1/0/NaN flag.** `_exact` is `1` when the observation
really was `k` back, `0` when the lookup fell further, `NaN` when there was
nothing to read. All the information the age column had is in the failures, and
the flag states them legibly — a model splits on it directly, and a human reading
`mean(_364_ago_exact) = 0.11` sees the problem 0050 needed an investigation to
find.

**`_latest_ago` stays a real measured distance**, because there it genuinely
varies per row — 1 to 45 days across a Rossmann validation block. Collapsing it to
a flag would delete the column that lets a model discount a stale value, which is
[0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md)'s entire point.
The asymmetry between the two is not an inconsistency; it is the difference
between a quantity that varies and one that does not.

## What was rejected

**`_days_old`, which reads better.** The ordering column is not always days. It is
whatever `to_periods` returns — days for a date column, and the column's own units
for a numeric one, which may be seconds, weeks, or an event index. A column named
`_days_old` over a series counted in something else is a confidently-wrong label,
and this is the area of the codebase that keeps producing exactly that failure.
**No unit appears in any of these names, deliberately.** The cost is real: a reader
must look up what the ordering column counts. That is the correct cost to pay, and
it is paid once per table rather than silently mis-paid per row.

**`_history`**, e.g. `<base>_364_history`. It suggests the column *contains* the
group's history when it holds a single observation — the same over-promise `_last`
made from the other direction.

**Keeping the offset age as a number "for symmetry"** with `_latest_ago`. Symmetry
between a variable and a near-constant is a cosmetic property bought with a column
that is 99% redundant. The shapes differ because the quantities differ.

## The report `fit_history` now makes, and why it is a report

0050 ended with a rule: the declared floor makes an offset *legal*, the span of the
serving block makes it *useful*. The obvious implementation is to refuse an offset
shorter than the block span. **That was rejected**, and the reason is
[0029](0029-the-harness-owns-the-exam.md): a short offset is not wrong, it is
*partially* answerable, and whether 11% coverage is worth having is a modelling
judgement about this data. Every such decision the library makes is one the agent
did not — so the tool states the number and warns, and the agent chooses.

Two new fields:

- **`block_span`** — the median width of a scored fold block, in the ordering
  column's units.
- **`offsets_answerable`** — `{"7": 0.112, "364": 0.805}`, the share of rows where
  each offset was answered exactly. Below 0.5, the note carries a WARNING saying
  the column is mostly the latest-known value wearing a name that promises a fixed
  lag.

**`block_span` exists because the agent could not have worked it out.** The advice
"use an offset longer than the period you are predicting" names a number that
appears nowhere in the agent's inputs: `inspect_data` reports `n_val` as a *row
count*, and nothing anywhere converts a row count to a span. The tool knows the
fold blocks and the ordering column, so the tool says. **Advice that names a
quantity the recipient cannot obtain is not advice** — this one had been sitting in
a tool note since slice 2, and v8 followed it into an 11%-answerable `lag7`.

## What it cost

**`_runs/rossmann_v8/bundle` no longer replays.** Its `feature_names` hold the old
names, and nothing reconstructs them. Accepted rather than versioned: no run
depends on that bundle, and a compatibility shim for a name that existed for one
day would outlive the reason for it. A bundle format that survives renames is a
real question and this is not the occasion to answer it.

Every name in the notebook, the tool docstrings, the agent's tool description and
the tests moved together; 426 tests pass. The rename is mechanical and total,
which is the only kind of rename worth doing.

## A limitation this made visible, and did not fix

`to_periods` converts **every date column to days**, whatever its granularity.
Right for a daily panel; unusable for a second-resolution event log, where an age
reads `0.00347` and an hour has to be asked for as an offset of `0.0417`. The
arithmetic is correct and the numbers are not usable.

**The fix is a declared unit, not an inferred one.** Inferring it from observed
spacing would make a column's values depend on whatever spacing happened to be in
the data — the same class of failure as inferring a date format. A numeric ordering
column sidesteps the whole question, since it keeps its own units. Documented in
`to_periods`, not built, and it needs its own record when it is.
