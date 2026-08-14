# 0049 — The last observation is a sample of size one

- **Date:** 2026-08-14
- **Status:** a measurement, and a negative one. Slice 2 is built and tested and
  **does not earn its place on Rossmann as it stands**. No code changed as a
  result; what changed is what slice 3 is for.
- **Related:** [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md)
  (which chose last-known-value + age, and is not wrong about what it claimed),
  [0048](0048-a-target-lag-is-not-target-encoding.md) (the per-fold freeze, whose
  predicted *direction* held and whose predicted *sign* did not),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md) (why a feature
  that only helps one dataset is not a feature), `scripts/probe_rossmann_history.py`
  (the instrument; both readings reproduce from it).

## What was measured

Not an agent run. `fit_history` driven by hand against v7's split, so the number
moves for one reason. Probe = the run's fixed ruler, metric `rmspe`, lower is
better. 915,744 work rows, 50,175 validation rows.

| table | probe | vs baseline |
|---|---|---|
| baseline (`Customers` dropped) | 0.21810 | |
| + `Store` history of `Sales` | 0.35054 | **−0.13244** |
| + `Store`+`DayOfWeek` history of `Sales` | 0.31530 | −0.09720 |
| + `Store` history of `Sales`, `lag7` | 0.35468 | −0.13659 |
| + `Store` history of `Customers` | 0.35920 | −0.14111 |

Every variant is worse, by a lot. Meanwhile the **naive** backward lag — every row
reading the true previous day, validation rows included — scores **0.09999**, or
+0.11811 *better* than baseline.

0048 predicted the honest feature would look weaker than the naive one. It did
not predict that the honest feature would be **worse than not having it at all**.
That is the finding.

## Why, and it is not a defect in the code

The work half ends **2015-05-01**: a Friday, `StateHoliday='a'`, German Labour
Day, on which **98.21% of stores were shut and sold zero**.

| | training rows | validation rows |
|---|---|---|
| median `Store_Sales_last` | 6488 | **0** |
| fraction `== 0` | 0.1% | **98.2%** |

The training columns are healthy, because forward chaining freezes at five
different fold boundaries and they mostly land on ordinary days. But **every
validation row reads one freeze date**, and that date was a holiday. The model
learned a relationship from plausible recent sales and was then served a column
that is zero almost everywhere.

This is not an accident of this split. 143 of the 851 candidate work-end dates —
**16.8%** — are days on which more than 90% of stores sold nothing, because
Rossmann closes on Sundays and public holidays. **Roughly one boundary in six
destroys the feature.** This one did.

## The general statement

With the dataset removed, which is the test `CLAUDE.md` sets:

> **The most recent observation is a sample of size one.** 0047 chose the last
> known value because it degrades *honestly* across the horizon — and it does.
> But honesty about staleness says nothing about whether the observation being
> degraded from was representative in the first place. On any panel with
> structural zeros — closures, outages, off-season, an account with no activity
> that week — it frequently is not, and the failure is silent because the column
> looks fine in training.

`_age` reports how old the number is. **Nothing reports how typical it is.** That
is the gap this run found, and it is a gap in 0047's design rather than in its
implementation.

## The control: how much was the holiday, and how much was the feature

The same comparison with `val_fraction=0.06`, which moves the boundary to
2015-04-22 — an ordinary Wednesday, no stores shut:

| freeze date | shut that day | baseline | + `Store` history | delta |
|---|---|---|---|---|
| 2015-05-01 (holiday) | 98.2% | 0.21810 | 0.35054 | **−0.13244** |
| 2015-04-22 (ordinary) | 0.0% | 0.22460 | 0.23602 | **−0.01142** |

So the holiday accounts for about **92%** of the damage. The mechanism is not
broken.

### One key survives, and why it is the same finding

The full variant set on the clean boundary:

| variant | probe | vs baseline |
|---|---|---|
| baseline | 0.22460 | |
| `Store` history of `Sales` | 0.23602 | −0.01142 |
| **`Store`+`DayOfWeek` history of `Sales`** | **0.22188** | **+0.00272** |
| `Store` history of `Sales`, `lag7` | 0.22987 | −0.00527 |
| `Store` history of `Customers` | 0.23392 | −0.00932 |

**Keying on `Store`+`DayOfWeek` is the only one that helps**, and the reason is
the thesis of this record rather than an exception to it. That key asks "what did
this store sell *the last time it was this weekday*", which puts the closed days
into **their own groups**. A Sunday's zero can then only ever be read by another
Sunday. The single observation stops being unrepresentative because the group was
made narrow enough that its members resemble each other.

Stated generally: **a group's last observation is only usable if the group is
homogeneous enough for one of its members to stand for the rest.** That is the
same requirement a group *mean* has (0046), arriving from the other direction —
and it is why the fix is either a narrower key or a wider window, both of which
reduce the variance of what is being read.

The gain is small (+0.00272 on one probe reading) and is not claimed as more than
directional. What it establishes is that the mechanism can help at all, which the
first reading did not show.

**The other three still hurt on a clean boundary**, `Store` history of `Sales` by
−0.01142. Even under favourable conditions, one store's single most recent sales
figure does not earn its place next to the columns Rossmann already has. Two
contributors are visible in the data and neither is the holiday:

- **17% of training rows have no value at all** (152,754 of 915,744) — the first
  forward-chaining block, which nothing precedes. Validation rows all have one.
- **The age ranges do not match.** Training ages run 1–305 with a median of 74,
  because fold blocks are ~150 days wide. Validation ages run 1–45, median 23.
  0048's serving symmetry holds in *shape* — age climbs across a block, as it
  climbs across a horizon — but **not in scale**: the model learns the
  age-to-discount relationship over a range four times wider than the one it is
  served. This is a second thing 0048 did not anticipate, and it is structural
  rather than incidental: block width is set by `n_splits`, and the serving
  horizon by the task.

## What follows

**Slice 3 (rolling and expanding windows) is no longer "cheap given slice 2". It
is the fix.** Averaging the same stored tail over 7 days instead of taking its
last element: **0%** of stores get zero, median 5771.7. A window is robust to
exactly the failure above, because it is not a sample of size one. This result is
a far stronger argument for windows than "the storage is already there".

**And the tool should say so.** `fit_history`'s note tells the caller that values
get older across a block; it says nothing about the key. The `Store`+`DayOfWeek`
result suggests the guidance a caller actually needs is "a narrow key is worth
more than a broad one here", which is the opposite of the advice that suits
`fit_group_stats`, where a narrow key thins the sample the statistic stands on.
Not changed in this record — it is a wording decision that should follow slice 3
rather than precede it.

**Nothing is being changed to make this number better.** Reversing the freeze
date, dropping closed days from history, or picking the last *non-zero*
observation would each improve Rossmann and none can be defended without naming
Rossmann — which is 0022's failure one level up. If "last known value should skip
rows the metric does not score" is a real capability, it is the row-filter
concept (`filter_rows`) reaching into history, and it needs its own argument and
its own record.

**0047 and 0048 are not superseded.** Both decided what they claimed to decide and
both still hold: staleness is per row, history is frozen, the fold loop mirrors
the serving horizon. What this record adds is that those properties are
*necessary and not sufficient*, and that a feature can be perfectly honest and
still be a bad feature.

## What this cost, and what it is worth

One negative reading on one dataset, which is exactly the kind of thing the
dataset-as-probe rule exists to buy. The capability stays: it is tested, it is
servable, it replays, and it is the substrate slice 3 needs. What it does not
have is evidence that anyone should use it on its own.
