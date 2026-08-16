# 0064 — "Month ahead" is the same idea and does not get the same tool

- **Date:** 2026-08-15
- **Status:** accepted. **A decision not to build**, and the record exists mainly
  so the not-building is legible: the capability passes every test this project
  applies to a new capability and is still refused.
- **It also corrects a claim in
  [0062](0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md).**
  0062 is accepted and stands as written; the sentence at issue is quoted and
  answered below rather than edited, which is the whole point of not rewriting
  history.
- **Related:** [0057](0057-the-capabilities-the-agent-does-not-have.md) (item 1,
  which this closes as half-built on purpose),
  [0062](0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md)
  and [0063](0063-the-grain-is-measured-from-the-table-not-required-of-the-caller.md)
  (the half that *was* built), [0049](0049-the-last-observation-is-a-sample-of-size-one.md)
  (the measurement that decides this),
  [0052](0052-a-window-is-a-worse-estimator-of-a-better-idea.md) /
  [0053](0053-the-fold-count-moves-a-window-and-we-cannot-say-why.md) (why a
  second knob on this feature is worse than none),
  [0056](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
  (history refuted externally), [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md) /
  [0043](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md) (where the real
  problem is filed).

## What prompted it

Ulf, reading the new tool's description: *"in der tool beschreibung schreibst du
'this is the tool if there is a second input'. aber 'month-ahead models' hat
keinen zweiten input oder?"*

Correct, and the description is right about its own tool. The Rossmann winner's
**month-ahead models** discard the most recent month of data when computing a
training row's features, so that a training row sits at the same distance from
its history as a row deep in the forecast window. One input, one timestamp, two
different aggregates of it. `fit_stream_aggregate` cannot express that and should
not: it needs two times per record, and here "when it happened" and "when it
became knowable" are the same moment.

So 0057's item 1 — *aggregate restricted to what was knowable a stated span
before the row* — is **two capabilities in one sentence**:

| | input | second timestamp | built |
|---|---|---|---|
| bitemporal stream | a second file, finer grain | yes | `fit_stream_aggregate` |
| "month ahead" | the main table's own past | no | **no, deliberately** |

## What 0062 claimed, and why it does not hold

0062 narrowed 0057's "found independently twice" and, in doing so, wrote:

> *The winner's version restricts an aggregate of **the main table's own past**,
> which needs no second input — and `fit_history` substantially does it already,
> since it freezes history at the end of each fold's training prefix and anchors
> windows at the row.*

That is true **in shape and false in scale**, and the scale is the entire point
of month-ahead models.

`fit_history` has no gap argument. A window is `[row - w, row)`, right endpoint
strict at the row. A gap does appear anyway — a row's usable history stops at its
fold's freeze — but it arrives as a *by-product of the fold geometry*, not as
something anyone declared:

- a **training** row sits `0` to `block_span` past its own block's freeze,
  depending where in the block it falls;
- a **served** row sits `forecast_horizon` to `forecast_horizon + block_span`
  past the work half's freeze.

0049 measured exactly this on Rossmann and named it: training ages ran **1–305
(median 74)**, validation ages **1–45 (median 23)** — *"0048's serving symmetry
holds in shape but not in scale"*. Month-ahead models make the gap **explicit and
equal on both sides**. `fit_history` leaves it to `n_splits` and therefore gets it
**different** on the two sides. Those are not the same thing; the second is the
condition the first exists to remove.

## What building it would be

One parameter: a declared gap on the window, `[row - gap - w, row - gap)`. It is
the same idea as `lead` with the second timestamp collapsed away, since with one
input "knowable" and "happened" coincide. It passes CLAUDE.md's test without
strain — *"a trailing aggregate restricted to end a stated span before the row"*
names no dataset — and it is perhaps twenty lines.

## Why it is refused anyway

Passing the generality test is necessary and not sufficient. Three reasons, in
increasing order of force.

**1. It has been found once, not twice.** 0057 claimed the capability was found
independently twice; 0062 established that the second finding was the stream.
What remains is one competition writeup — and per 0056, that same author reports
recent data barely mattering to him.

**2. The feature family has never paid.** Five negative readings for history, one
of them external and decisive: −0.0301 on Rossmann's private board (0056). Adding
a knob to a feature that has not yet earned its existing knobs is the pattern
0052/0053 were written to record — this project's vocabulary growing faster than
its ability to judge the vocabulary.

**3. And the one that decides it: the existing knob cannot be searched.** 0053
measured the same window at `n_splits=3` and `5` and got **+0.047 and −0.093**,
with identical train/serve counts on both sides, concluding that *"window length
cannot be searched over here and the number believed"*. A second free parameter on
a feature whose first parameter the harness cannot advise on does not enlarge what
the agent can express usefully; it enlarges what the agent must guess. The probe
cannot tell it whether a gap of 30 beat a gap of 7, for the same reason it cannot
tell it whether 364 beat 728.

### The remedy is misfiled, which is the real objection

The asymmetry 0049 measured is genuine and unfixed. But it is a property of the
**evaluation geometry**, not of any feature: *every* backward-looking column in
this harness is fitted against a freeze whose distance from the row differs
between validation and serving. A `gap` argument does not repair that. It lets the
agent opt one feature out of it, by hand, if it happens to choose the right
number — while every other feature stays exactly as asymmetric as before.

A capability that lets the agent locally work around a defect in the exam is the
wrong shape of fix. The exam belongs to the harness (0029), and the gap between
the block it freezes at and the block it will be served at is 0041's and 0043's
question.

### Why the stream was different, stated as the discriminating test

Worth keeping, because it is the line that separates the two halves of 0057 item
1 and it is not "one has two files":

> For the stream, **no combination of existing tools could express the honest
> feature at all.** For month-ahead, the agent can already build a trailing
> window; it is merely asymmetric — and asymmetric in a way it shares with
> everything else here.

A missing capability gets built. A shared defect gets measured and fixed at its
source, or it gets recorded and left alone.

## The cost of refusing

- **`fit_history` stays asymmetric and nothing says so at the point of use.** The
  tool reports `shortest_training_prefix`, `longest_training_prefix`,
  `block_span` and `window_observations`; none of them is "how old was what a
  training row read, against what a served row will read". 0049 computed it once,
  by hand, and it has not been surfaced since. That is an *instrument*, and per
  0058 item 3 instruments get built when a run needs one, not pre-emptively — so
  it is named here and not built either.
- **0057 item 1 is closed as half-done.** Anyone reading 0057 alone will expect
  both halves; the table at the top of this record is the answer.

## What would change this

Any one of:

- a second dataset where the honest feature demonstrably cannot be expressed
  without a declared gap — not merely where a gap would help;
- history paying at all, anywhere, so that a knob on it is a knob on something
  that works;
- 0041/0043 resolving how the harness picks its evaluation block, after which a
  declared gap might fall out of that decision instead of competing with it.
