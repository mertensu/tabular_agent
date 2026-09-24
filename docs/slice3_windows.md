# Slice 3 — rolling windows over stored history

> **Superseded by [0052](../research/archive/decisions/0052-a-window-is-a-worse-estimator-of-a-better-idea.md)
> (2026-08-14). This was the plan; it was built and it measured negative.** The
> trailing window loses to 0046's unbounded mean of the same key, the gain the
> first probe showed belongs to the periodic key rather than the averaging, and a
> window longer than the shortest fold's training prefix is train/serve skew. Read
> 0052 first — this file is kept as the argument that was made beforehand, and
> section 5's exit condition is the part that paid off.

**Status:** a plan, written 2026-08-14 so a fresh session can start from the
argument instead of rebuilding it. Nothing here is built.

**Read first, in this order:** `research/archive/decisions/0047`, `0048`, `0049`, `0050`, then
`src/tabular_agent/encoders.py` (`HistoryEncoder`, `fit_history`) and
`research/archive/notebooks/phase5_history_and_the_horizon.ipynb`. `docs/time_aware_features.md` is
the original three-slice plan; slices 1 and 2 are done.

---

## 1. Where slice 2 left things

`fit_history` stores each group's last `keep` observations and replays them. It
emits, per row:

| column | meaning |
|---|---|
| `<by>_<value>_latest` | the newest observation for this row's group, as of the freeze |
| `<by>_<value>_latest_ago` | how far back that observation sits from *this row* |
| `<by>_<value>_<k>_ago` | the newest observation at or before `row - k` |
| `<by>_<value>_<k>_ago_exact` | `1` when it really is `k` back, `0` when the lookup fell further |

History is frozen **per fold** at the end of the training prefix (0048), so a
validation row reads what a served row the same distance ahead would read.
Offsets require `TaskSpec.forecast_horizon` and must be at least that large.

## 2. What Rossmann v8 established, and it should shape this slice

Full detail in 0050. The three findings that matter here:

**The anchor is what matters.** `_latest` is anchored to the *freeze boundary* —
one date shared by every served row — so the whole batch inherits whatever was
true on that one day, and no day is typical. On v8 the boundary was a public
holiday; after the agent filtered closed days out, `_latest` was *still* ruinous,
because the last open day before a holiday is a pre-holiday spike (median 9,697
against 6,337 overall). It took a working table from 0.200 to 0.464, the worst of
25. An offset is anchored to *the row*, so each row draws independently: the four
best tables in that run were all offsets.

**Long offsets already deliver most of what a window would.** The agent chose
`[364, 728]` — one and two years, picked to keep the weekday aligned — and shipped
it. That is close to the competition winner's *"sales for this store on a typical
Monday, averaged over the last 6 months / 1 year"*. **Offsets give the recency and
the periodicity; the window would add only the averaging.** That is a real but
much smaller gap than 0049 claimed, and this slice should be sized accordingly.

**Short offsets are a trap.** An offset shorter than the *span of the block being
predicted* is answerable for the block's earliest rows and silently becomes
`_latest` for the rest. On v8 a 7-day offset over a 44-day block was honoured for
**11%** of rows. `fit_history` now returns `offsets_answerable` and `block_span`
and warns below 50%, which is the mitigation; a window has the same exposure and
will need its own version of it.

## 3. What to build

A window is an aggregate over the tail already stored, so the storage and the
per-fold freeze need no change. The shape that fits the existing tool:

```
fit_history(by=[...], value="...", windows=[(90, "mean"), (28, "std")])
```

emitting `<by>_<value>_90_mean` and so on.

**Design constraints, most of them from 0050:**

- **`window` is a free number in the ordering column's units, never a menu.**
  A blessed list of lengths would be us deciding that 180 and 365 are the good
  ones — true of one dataset, and the agent's call anyway (0029). This is the
  point Ulf made directly: the agent must be able to *build* the winner's feature,
  we just must not bake it in.
- **Anchor the window at the row, not at the freeze.** `[row - w, row)` — the same
  property that makes offsets survive a bad boundary. A window anchored at the
  freeze is `_latest` with extra steps and will fail the same way.
- **Report coverage, as offsets now do.** How many observations actually fell in
  the window, or what share of rows got a full one. A window that silently
  averages two points on the rows that matter is the `lag7` failure in new
  clothing.
- **`keep` must cover the longest window**, and `MAX_KEEP` is 1000. A 365-day
  window on daily data is expressible; a multi-year one is not. **Make that a
  stated refusal, not a silent truncation** — a quietly shortened window is a
  wrong number with a confident name, which is the failure this whole area keeps
  producing.
- **Watch the storage.** `Store`×`DayOfWeek` over a year is ~52 observations ×
  7,805 groups; `Store` over a year is 365 × 1,115. Both ~400k floats — bounded,
  which is the property that matters, but no longer small, and it goes in every
  bundle. Measure before shipping.
- **Which statistics?** `mean` certainly. Reuse `GROUP_STATS` if it fits, but note
  0046's reasoning about which statistics are defensible — `min`/`max` over a
  short window is close to a single observation, which is the thing 0049 warns
  about.

## 4. Two open questions that are not this slice's to settle

**Should `_latest` still be emitted unconditionally?** It is currently produced on
every call whether asked for or not, so v8's best tables — and its shipped bundle
— all carry the pair that destroyed t7. Making it optional is a small change with
a real argument behind it now. It is listed here because slice 3 will make the
output wider still, and the two decisions interact.

**The time unit.** `to_periods` converts every date column to **days**. Right for a
daily panel, unusable for a second-resolution event log, where an age reads
`0.00347` and an hour is an offset of `0.0417`. The arithmetic is correct and the
numbers are not usable. The fix is a *declared* unit rather than an inferred one —
inferring it from observed spacing makes column values depend on whatever spacing
happened to be in the data. A numeric ordering column sidesteps it entirely, since
it keeps its own units. Not built, and it needs its own record.

## 5. How to know whether it worked

**Do not tune against Rossmann's score.** 0022 and `CLAUDE.md`. The instrument is
`research/archive/scripts/probes/probe_rossmann_history.py`, which drives the tools by hand so the number
moves for one reason; `research/archive/runs/rossmann/run_rossmann_v8_history.py` is the agent-run
version, and it takes ~11 minutes.

The comparison worth making is **window versus long offset on the same table** —
whether averaging adds anything once the recency and periodicity are already
there. 0049 predicted windows would fix the structural-zero problem, which is
true and, after v8, mostly irrelevant: offsets already avoid it. If a window adds
nothing over `[364, 728]`, **that is a publishable result and the slice should
stop**, not be argued into existence.

Register predictions before running, in the script's docstring, and split them by
*mechanism* — v8's registered prediction 3 was wrong precisely because it treated
`_latest` and offsets as one capability when they behave oppositely.
