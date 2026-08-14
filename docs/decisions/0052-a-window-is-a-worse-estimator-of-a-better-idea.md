# 0052 — A window is a worse estimator of a better idea

- **Date:** 2026-08-14
- **Status:** built, measured, and **the measurement is negative**. The capability
  ships; the claim slice 3 was built on does not survive. Kept deliberately (Ulf's
  call) rather than reverted — see "Why it ships anyway".
- **Related:** [0046](0046-the-mean-is-a-lossy-summary-of-a-group.md) (whose
  all-time target mean beats every window measured here),
  [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md),
  [0048](0048-a-target-lag-is-not-target-encoding.md) (whose serving symmetry is
  what the long windows break), [0049](0049-the-last-observation-is-a-sample-of-size-one.md)
  (which predicted windows would be *the fix*), [0050](0050-the-anchor-is-what-matters-not-the-history.md)
  (the anchor argument, which held), [0051](0051-one-word-for-one-idea-and-a-flag-instead-of-an-age.md),
  `docs/slice3_windows.md`, `scripts/probe_rossmann_windows.py`,
  `scripts/probe_rossmann_windows_control.py`,
  `scripts/probe_rossmann_window_length.py`.

## What was built

`fit_history(windows=[(90, "mean"), ...])`. Aggregates of the stored tail over
`[row - w, row)`, emitting `<base>_<w>_<stat>` beside `<base>_<w>_count`.
`WINDOW_STATS` is `(mean, std, median)`; `min`/`max`/`sum` are off the menu for
0046's reasons at shorter range. The window is **anchored at the row**, and its
right endpoint is **strict** so a row is never inside its own summary — `_latest`
can lean on the per-fold freeze for that, a window must not, because a
self-included point in a mean over ninety is diluted to invisibility where a
self-included `_latest` shows as a perfect correlation.

## The result, and it is not the one the slice was built for

Rossmann, v7's split, `rmspe`, one `fit_history` call per row of the table so the
only thing varying is the parameter. Positive is better.

| | probe | vs baseline |
|---|---|---|
| baseline (`Customers` dropped, `Store_DOW` key built) | 0.21810 | |
| **all-time target mean of `Store_DOW` (0046)** | **0.15762** | **+0.06048** |
| trailing 364 mean of `Store`+`DayOfWeek` (slice 3) | 0.17352 | +0.04458 |
| both together | 0.16944 | +0.04866 |

**What slice 3 bought over what 0046 already had: −0.01590.** The trailing window
loses to the unbounded mean of the same key, and adding it *on top of* the
unbounded mean makes the table worse than the unbounded mean alone.

`docs/slice3_windows.md` said before any of this was written: *"If a window adds
nothing over [364, 728], that is a publishable result and the slice should stop,
not be argued into existence."* This is that result, and it is stronger than
"nothing" — it is negative on both comparisons.

**The +0.045 was almost all 0046.** The first probe read a trailing
`Store`+`DayOfWeek` mean at +0.04458, the largest single-feature gain this project
has recorded, and the obvious reading was that slice 3 had delivered. The control
says most of that number belongs to the *key*, not to the trailing bound. The
competition winner's feature — *"sales for this store on a typical Monday over the
last 6 months"* — is two halves, and **the half this project was missing was the
periodic key, which 0046 had already built.**

## The key is the whole story, not the averaging

From the first probe, same tool, same 364 days, same statistic, one difference:

| key | probe | vs baseline |
|---|---|---|
| `Store` | 0.22072 | −0.00263 |
| `Store` + `DayOfWeek` | 0.17352 | **+0.04458** |

Averaging *across* weekdays destroys the periodicity; averaging *within* the
weekday keeps it. This is 0050's shape a second time — a large win and a loss
separated by one parameter, and again **not the parameter the slice is named
for**. 0050 said the anchor mattered more than the history; 0052 says the key
matters more than the window.

## Two failure modes, at opposite ends, and the second one is new

Sweeping the length at a fixed key exposed a curve with an interior optimum. The
registered prediction was **monotone improvement with length**, on the theory that
a window is just a smaller sample of the same quantity. **That was wrong.**

| window | probe | vs baseline | observations |
|---|---|---|---|
| 91 | 0.23365 | −0.01555 | 3 |
| 182 | 0.18072 | +0.03738 | 16 |
| **364** | **0.17352** | **+0.04458** | 39 |
| 546 | 0.21840 | −0.00030 | 58 |
| 728 | 0.31097 | **−0.09287** | 59 |
| all-time (0046) | 0.15762 | +0.06048 | all |

Coverage does not explain it: 546 and 728 hold **58 and 59** observations and are
0.093 apart. The mechanism at the long end is different, and it is the finding
worth keeping:

> **A window longer than the shortest fold's training prefix cannot be filled in
> training, while a served row fills it completely.**

Under forward chaining the first fold trains on a short head of the data. Measured
here: the work half spans 850 days and fold 0's prefix is **136**. So

| window | train median obs | served median obs | ratio |
|---|---|---|---|
| 364 | 39 | 49 | 1.26x |
| 728 | 59 | 101 | **1.71x** |

The model learns the relationship between "a mean over ~59 observations" and the
target, then is served "a mean over ~101" — the same column name, a different and
less noisy quantity. That is train/serve skew, which is what 0048 exists to
prevent, arriving through a door 0048 did not cover: not *when* the history stops,
but *how much* of it there was to summarise.

So the two ends fail for unrelated reasons, which is why no single number guards
the parameter:

- **too short** — the window holds almost nothing (91 → 3 observations);
- **too long** — the earliest fold cannot fill it (728 → 1.71x skew).

## The reporting defect this found, in code written the same day

`window_observations` was added this session precisely to make a badly-chosen
window obvious. **It reported 59 observations for the 728-day window and 39 for
the 364-day one** — i.e. it made the catastrophic column look *better* than the
good one, because the damage is in the **ratio between the training and serving
sides** and the count only ever sees one of them.

This is 0031's shape in the reporting layer for the second time in three records
(0050 caught it in `summarise_history`): a number that looks like a check and only
describes one half of the thing it is guarding.

**Fixed, and the fix is general rather than fitted.** `fit_history` now returns
`shortest_training_prefix` — the narrowest history any fold was fitted on — and
warns when a window exceeds it, saying explicitly that *the count is no guide
here*. Forward chaining always makes early prefixes shorter than full history;
nothing about this is Rossmann-specific, which is what makes it admissible under
`CLAUDE.md`.

**And the guard over-fires, which has to be said plainly.** Rossmann's shortest
prefix is 136 days, so the warning flags **364** — the best window in the sweep,
at a ratio of 1.26x — as well as 728 at 1.71x. The threshold marks where the
mechanism *starts*, not where it starts costing anything, and the two are far
apart. A check that warns about the winner is a check a reader learns to skip,
which is its own failure mode.

The obvious repair is to warn on the ratio, or on some multiple of the prefix,
rather than on the prefix itself. **Both were rejected**: this run has exactly two
points on that curve (1.26x good, 1.71x ruinous), and picking a threshold between
them is fitting a constant to two observations of one dataset — 0034's refusal,
which declined to set a tiebreak margin from three points. The conservative
version states a real structural boundary and is honest about being conservative;
a tuned version would look sharper and mean less. Left as a stated limitation, and
the number the reader should actually act on — the prefix — is printed either way.

Also worth stating: the *training-side* clipping is not uniform, which is what
makes this worse than "less data". Fold 0 can hold ≤19 weekly observations in a
728-day window, fold 4 up to ~101, and serving delivers 97–104 to every row. The
trained column is a **mixture of differently-truncated windows** under one name,
against a uniformly full served column. Less data everywhere would merely be a
noisier feature; differing amounts between training and serving is a different
feature.

## Why it ships anyway

Reverting was offered and declined. The arguments for keeping it:

- **One dataset is one reading**, and this one is unusually favourable to the
  all-time mean: Rossmann's per-store-weekday level is stable across 2.5 years, so
  bounding the aggregate throws away evidence and buys no adaptation. The
  motivating case — *a store whose trade level moved a year ago* — does not
  obviously occur here, so the feature was tested where it had least to offer.
- **The capability is correct, cheap and general.** Storage stays bounded, the
  cost is a cumulative sum, and the mechanism survived every property test.
- **A negative result that is reachable is worth more than one that is not.**
  Deleting the code makes the finding un-retestable on the next dataset.

What ships with it: the record leads with the loss, so a later session does not
rediscover windows as an obvious win. `median` is **kept but explicitly
unvalidated** — its a-priori argument (a median survives the spike 0050
documented) was not refuted, merely unsupported: on the good key it landed 0.0003
from the mean, and on `Store` it was clearly worse. Dropping a statistic because
one dataset did not need it is the mirror image of adding one because it did.

## What this does not establish

**None of these differences has a yardstick.** Under an ordered split validation
is one period, `val_se` is `None`, and n=1 in the unit that varies — 0022, 0036
and 0041 all say so. The *direction* is consistent across three independent
comparisons and the 0.093 at w=728 has a mechanism behind it, but a 0.016 gap
between the window and the all-time mean carries no error bar. The paired
bootstrap of 0036 is the tool that would fix this and it is still not wired in.

**The offsets contradicted v8 and this run does not say why.** `[364, 728]` scored
−0.02290 here against **+0.0217** in v8. The likely difference is the table
underneath: v8's agent had already filtered `Open == 1`, so a 364-day lookup could
not land on a closed day, where this baseline still holds `Sales == 0` rows.
Plausible, untested, and named here so it is not later assumed.

**The super-additive row is unexplained.** On the `Store` key, offsets alone
scored −0.02290 and the window alone −0.00263, and together they scored
**+0.04584**. Two features that each hurt, helping substantially in combination,
is not a result this run can account for and it is recorded as an oddity rather
than narrated.
