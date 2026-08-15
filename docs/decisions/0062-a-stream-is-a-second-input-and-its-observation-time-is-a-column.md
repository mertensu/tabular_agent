# 0062 — A stream is a second input, and its observation time is a column

- **Date:** 2026-08-15
- **Status:** **accepted, being built.** This is the record 0057 asks for before
  its item 1 is implemented — the design and the build order, written before the
  code so the argument can be checked against something other than the diff. No
  measurement is claimed here and none should be read into it.
- **Related:** [0057](0057-the-capabilities-the-agent-does-not-have.md) (item 1,
  the inventory entry this implements — and whose justification is narrowed
  below), [0058](0058-the-ranking-failure-was-one-feature-not-the-block.md) (the
  build order that puts this first),
  [0029](0029-the-harness-owns-the-exam.md) (why the harness supplies the
  capability and does not decide whether to use it),
  [0010](0010-store-code-not-columns.md) (store the recipe, not the values —
  which is the whole architectural choice here),
  [0011](0011-stability-check.md) (why this cannot be a `transform_features`
  snippet), [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md),
  [0050](0050-the-anchor-is-what-matters-not-the-history.md) (the frozen anchor
  failure the rejected option would repeat),
  [0052](0052-a-window-is-a-worse-estimator-of-a-better-idea.md) /
  [0053](0053-the-fold-count-moves-a-window-and-we-cannot-say-why.md) (the
  vocabulary-outgrowing-judgement pattern this record deliberately accepts a
  version of), `scripts/prep_restaurant.py`.

## The problem

A dataset can ship a **stream**: a related file at a finer grain than the table
being modelled, whose records carry *two* times — the time they are **about**,
and the time they became **knowable**. Restaurant reservations are the shape:

    air_store_id   visit_datetime      reserve_datetime    reserve_visitors
    air_0164…      2017-03-13 18:00    2017-01-20 16:00    5     <- 52 days ahead
    air_0164…      2017-03-13 18:00    2017-03-08 20:00    2     <-  5 days ahead
    air_0164…      2017-03-13 18:00    2017-03-11 23:00    3     <-  2 days ahead
    air_0164…      2017-03-13 18:00    2017-03-13 17:00    1     <-  same day

The table being modelled has one row per store-day, so the stream cannot be a
column: four records collapse into one number. `prep_restaurant.py` does collapse
them, into `air_reserve_visitors = 11`, **and the collapse discards
`reserve_datetime`.** That is the whole defect, because the second timestamp is
the only thing that says which of those four records a *forecast* could have seen.

The stream is cut at the moment the forecast is issued (2017-04-22). So:

| | training row (13 Mar) | forecast row (31 May) |
|---|---|---|
| booked same day | counted | cannot exist |
| booked 2 days ahead | counted | cannot exist |
| booked 52 days ahead | counted | counted |

**36% of all reservations in this file were booked with zero days' notice.** The
column is therefore systematically larger in training than at serving, by an
amount that *grows with the horizon* — 1 day at the near edge of the forecast
window, 39 at the far edge.

**No guard in this harness detects it.** No fold boundary is crossed, no encoder
is misfitted, the column exists at predict time, and validation *confirms* the
feature is excellent because validation is drawn from the same uncut half. It is
the project's characteristic failure — a number that looks fine and describes
something else — in a form nothing here can currently see or express.

## Why it is general

The concept is standard and predates every dataset in this repo: **bitemporality**
— *valid time* (when a fact holds) against *transaction time* (when it was
recorded). Data warehouses have modelled it explicitly for decades. Insurance has
a name for the gap and reserves money against it: **IBNR**, "incurred but not
reported". Order books, sales pipelines, shipment notices, clickstreams,
claims feeds and medical records all have the shape.

The CLAUDE.md test passes without strain: *"an aggregate of a related stream,
restricted to the records already observed a stated span before the row being
predicted"* names no dataset.

### Where 0057's justification is narrowed

0057 argues item 1 was "found independently twice" — once in this dataset, once in
the Rossmann winner's **"month ahead models"**, which discard the most recent
month of data when computing features so that a training row sits at the same
distance from its history as a row deep in the forecast window.

Those are the same *idea* and not the same *capability*. The winner's version
restricts an aggregate of **the main table's own past**, which needs no second
input — and `fit_history` substantially does it already, since it freezes history
at the end of each fold's training prefix and anchors windows at the row. What
needs a second input is the **bitemporal stream at a different grain**, and that
has been found once, in one dataset, out of the four this project has run.

So the "twice, independently" support belongs to the principle. The plumbing rests
on one dataset plus the external argument above. That is written down here rather
than smoothed over, because 0052/0053 are the record of this project's vocabulary
growing faster than its ability to judge the vocabulary, and this is a
deliberately-taken instance of the same risk.

### Why measuring first was considered and rejected

A `probe_restaurant_lead.py` — aggregate at several leads, run both ways, score
against the private board — was proposed and dropped. It answers "does the skew
cost us on Restaurant", which is the leaderboard-shaped question CLAUDE.md rules
out: *"If the argument for a feature is 'it would help on Rossmann', it is not an
argument."* The library question is whether the agent can express the honest
feature at all, and per 0029 the harness supplies the capability and reports;
whether to use it is the agent's call. Ulf made this argument and it is correct.

## What is built

A tool the agent can reach, `fit_stream_aggregate`, and the input channel it needs.

**The stream is a declared second input**, named on `RunSession` the way
`serving_columns` is — the run's owner knows the file is a stream and nothing
recovers that by inspection. It is surfaced in `inspect_data` and the brief, or
the agent never finds it.

**The encoder stores the recipe, not the numbers.** `StreamAggregateEncoder`
persists `by`, `value`, `event_time`, `observed_at`, `lead`, `stat` and nothing
else; the stream is handed in again at transform time and the aggregate
recomputed. This is 0010 applied to an aggregate: the same move
`transform_features` already makes when it stores code and re-runs it rather than
storing the columns it produced.

**Emitted as a pair, never a bare value.** `..._lead7_sum` beside
`..._lead7_count`, for the reason every window in `HistoryEncoder` carries a
count: a sum over one record and a sum over forty are the same number wearing the
same name, and only the count says which one a given row got.

**Leads below the declared `forecast_horizon` are refused**, on the pattern
`offsets` and `windows` already use. Several leads are allowed at once and become
separate columns, because a short and a long lead are different features rather
than competing settings for one.

**The known limit, stated rather than hidden:** `forecast_horizon` is the
*floor* of the block (0047), so a lead at the floor is honest for the near edge
and under-covers the far edge. The `_count` column is what makes that visible per
row. Declaring the far edge as well would close it by construction and is not
built — it adds a field the data owner must get right on every ordered run, and
the count already reports the damage where it lands.

## What was rejected

**Materialising the aggregate into stored state.** The encoder could precompute
`{(key, event_time) -> (stat, count)}` over the whole stream at fit time — 29,830
entries for `air_reserve` — and store it in the lineage. It needs *no* plumbing
change at all: `predict_bundle(bundle, df)` stays one-frame.

It is rejected for the reason 0050 exists. That option only works while the stream
file already contains the future, which is true of a downloaded competition file
and false of every real serving situation, where next month's reservations arrive
after the fit. Baking in "the file already has the future" freezes an input that
keeps moving, and 0050 is the record of a frozen anchor costing a working model
two thirds of its score. It also scales badly — the same design over
`hpg_reserve.csv` (2M records) is roughly half a million entries in a JSON
lineage.

**A `transform_features` snippet.** 0011 refuses any column whose value depends on
which other rows are in the frame, and an aggregate is exactly that. The refusal
is correct and is why this is a tool.

**Leaving it in `prep_*.py`.** Sanctioned for dataset-specific knowledge, and it
is where the *cut* is documented today. But a prep script that quietly hands over
an honest column teaches the harness nothing, and the agent would still be unable
to express the feature — or to get it wrong, which is equally the point.

## The cost

- `Pipeline.transform` and `predict_bundle` grow a `streams` argument. Default
  `None`, so every existing bundle keeps predicting exactly as it did; a bundle
  whose manifest *requires* a stream and is handed none fails loudly.
- One more thing the agent has to understand, on a surface that is already ten
  tools. 0052/0053's hazard, accepted knowingly.
- The prep script gains a second output file and the run script a declaration.

## Build order

1. `StreamAggregateEncoder` in `encoders.py` + tests — the aggregation, the lead
   restriction, the count column, round-trip through `to_dict`/`from_dict`.
2. Stream registry on `RunSession`; surfaced in `inspect_data` and the brief.
3. `fit_stream_aggregate` in `harness/tools.py` + `harness/agent.py`, with the
   refusals above.
4. `Pipeline.transform(df, streams=...)`, `predict_bundle`, bundle manifest.
5. `prep_restaurant.py` writes the stream out; the collapsed column stays (0029).
6. Notebook — one slice, driven by hand first, per CLAUDE.md.
