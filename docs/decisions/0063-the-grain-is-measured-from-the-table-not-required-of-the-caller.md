# 0063 — The grain is measured from the table, not required of the caller

- **Date:** 2026-08-15
- **Status:** accepted, built (step 1 of 0062's build order).
- **Supersedes nothing.** It settles a question
  [0062](0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md)
  left open and did not notice it was leaving open. 0062 stands as written.
- **Related:** [0062](0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md)
  (the design this implements), [0011](0011-stability-check.md) (why the bucket
  may not be read off the frame in front of it),
  [0051](0051-one-word-for-one-idea-and-a-flag-instead-of-an-age.md) (the inference this is
  *not*, and the declaration it falls back to),
  [0010](0010-store-code-not-columns.md) (why the grid is stored arithmetic
  rather than a list of dates), [0029](0029-the-harness-owns-the-exam.md),
  `CLAUDE.md` (`scripts/prep_*.py` as a sanctioned home for dataset knowledge —
  and its limit).

## The problem

`StreamAggregateEncoder` has to decide **which row a stream record belongs to**.
The table has one row per store-day; the stream stamps a booking
`2017-03-13 18:00`. Something must say that the 18:00 record is *this day's*.

The first implementation matched a row's ordering value against a record's
`event_time` by equality, and left aligning the two to `prep_restaurant.py`.
`fit` refused a stream that matched nothing, so the failure was at least loud on
that dataset.

Ulf rejected it, and the reason is not about Restaurant:

> *etwas dem prep skript überlassen ist gefährlich … vor allem wenn ich das
> projekt der allgemeinheit zur verfügung stellen möchte. es sollte so wenig wie
> möglich in prep stehen müssen und alles was dort drin stehen muss, muss
> festgehalten werden. sonst können andere mein tool nicht nutzen.*

That is right, and it sharpens a line CLAUDE.md draws loosely. `prep_*.py` is a
sanctioned home for *dataset knowledge* — this join, this metric, this split.
It is not a home for **library preconditions**. A rule that every user of the
library must obey, which nothing in the library states, checks or records, is not
a design decision; it is a trap with a docstring. And the trap here has this
project's characteristic shape: get it wrong and you do not get an error, you get
a well-formed column of zeros reading *"nothing was ever booked"*, which
validation will happily confirm.

## What was chosen

**The library measures the table's grain once, at fit time, stores it, and
replays it.**

`fit` reads the distinct ordering values, takes the **commonest gap** between
consecutive ones as `grain_`, and keeps it — with `origin_`, the phase — on the
encoder and in `to_dict`. Both sides are then floored onto that grid, so a record
at 18:00 and a row at midnight meet in the same bucket whichever way round the
resolutions happen to be.

Three properties make this defensible rather than merely convenient:

- **The lead keeps full resolution.** Flooring decides *which row*; it never
  touches *what was knowable*. The restriction is still the row's true ordering
  value against the record's true `observed_at`, both unfloored. Blunting the lead
  would defeat the encoder's only purpose.
- **The grid is stored arithmetic — a width and a phase — not the dates that were
  seen.** It has to be: every serving row sits *past* the end of the table the
  encoder was fitted on, so a stored calendar could not place them. This is 0010
  again, and it is why the grid survives into the bundle instead of being
  re-derived at predict time from whatever frame turns up.
- **Raggedness does not move it.** The gaps are taken over the ordering values
  present *anywhere in the table*, so a store reporting four days a week does not
  make the grain four days — the failure 0054 found in this dataset's spine.

**When the measurement cannot be trusted, it refuses.** The commonest gap is only
believed if every distinct ordering value sits on the grid that gap implies. A
daily panel missing a few table-wide holidays passes (gaps of 1, 1, 2, 1; every
value a whole number of days from the first). An irregular event log fails, and
gets an error naming the offending value and asking for `event_grain` to be
declared. A declared grain is used as given: the data's owner outranks the
measurement, per 0029's division of labour.

## Why this is a measurement and not 0051's inference

0051 left `to_periods`' unit problem standing on the rule that the fix must be a
**declared** unit and never an inferred one: every date becomes days, and
deducing "these look like hours" from the data would make a column's values
depend on the spacing that happened to be present. The objection stands.

This is a different act. The unit is already fixed by `to_periods` and is not in
question. What is measured is the **spacing of the rows in that unit** — a fact
about the table, in units already agreed, which the class can state and *check
against the data it measured it from*. The check is the whole difference: an
inference that cannot be falsified is a guess, and this one is falsified by any
row off the grid.

## What was rejected

- **Requiring an aligned stream (the original).** Above. It is the rule that
  cannot be written down anywhere the user will read it.
- **A mandatory `event_grain` argument.** Honest, and it makes the common case —
  a daily panel, where the answer is on the face of the data — into a question
  the agent must answer correctly on every run, with a silent column of zeros as
  the penalty for a wrong answer. Kept as the *fallback* for the case where the
  data genuinely does not say, which is where a declaration earns its cost.
- **Bucketing against the ordering values present in the frame** (`merge_asof`
  onto the rows themselves). Batch-dependent by construction: one row alone and a
  batch of a million would bucket differently. 0011 refuses it, correctly.
- **Taking the *minimum* gap rather than the commonest one.** More conservative
  in appearance, and wrong: one stray timestamp a second off its neighbour would
  set the grain to a second and quietly empty every bucket. The mode is robust to
  exactly that, and the on-grid check catches the stray row loudly instead.

## The cost

- Two more fields in the lineage (`grain`, `origin`) and one more optional
  argument. Both are the price of the rule being *recorded* rather than assumed.
- The refusal has a threshold in it — `1e-6` of a grain, for "sits on the grid".
  It is a float-comparison tolerance rather than a tuned constant (0034), but it
  is a number in the code and it is named here so it is not later assumed to have
  been measured.
- The grain is a property of the **table**, and a run whose table changes shape
  between fit and predict gets the old grid. That is the intended behaviour — the
  alternative is a column that means different things on either side of the seam —
  but it means a bundle carries an assumption about its input's spacing, and
  nothing currently checks it at predict time.
