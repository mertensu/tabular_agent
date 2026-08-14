# Time-aware features — the gap, and what building it requires

**Status:** a plan. Written 2026-08-14 so a fresh session can start from the
argument rather than rebuild it.

**Updated 2026-08-14, later the same day:**

- **Both blocking questions (§3.2, §3.3) are settled** — see
  [0047](decisions/0047-the-forecast-horizon-is-a-column-not-a-constant.md). The
  short version: frozen history + a per-row staleness column, optional
  `TaskSpec.forecast_horizon` as a *minimum*, no recursion, predict rows never
  read their own batch. §3.2 and §3.3 below are left as written; the record is
  the decision.
- **Slice 1 is built** —
  [0046](decisions/0046-the-mean-is-a-lossy-summary-of-a-group.md). Gap A is
  closed. Slices 2 and 3 are untouched.

This is longer-form reading in the sense of `docs/decisions/README.md`: it
records what we know and what has to be settled, and the decisions it leads to
get their own records when they are made.

---

## 1. The evidence that this is the gap

Rossmann v7's fifteen tables, by probe score:

| table | probe | gain | what it did |
|---|---|---|---|
| t7 | 0.1587 | | built `Store_DOW = Store*10 + DayOfWeek` |
| **t8** | **0.1344** | **+0.0243** | target-encoded it — "average sales for this store on a Monday" |
| t9–t15 | 0.1342 | **+0.0006 total** | nine more composite keys, same tool, three negative |

The per-group-target idea was the largest late gain in the run. Then **half the
run was spent re-drilling the same hole for 2.5% of the return.**

That is not the agent being weak. Every group feature in v7 goes through one
mechanism — target-encode a hand-made composite key — and after `t8` there was
nothing else it could say. **The agent exhausted its vocabulary and then repeated
itself.**

## 2. What the vocabulary cannot express

The harness can carry exactly one kind of thing across the fit/predict seam: **a
mapping**, category → number. `fit_encoding` stores one; `fit_group_stats` stores
one. Two gaps follow.

### Gap A — only the mean

`fit_group_stats` has a `stat=` parameter but **correctly refuses the target as
its `value` column** (that is target encoding's job, and it needs smoothing).
`fit_encoding(method="target")` produces only a smoothed mean.

So "this store's average Monday sales" is sayable and **"its standard deviation,
its median, its count" are not.** The spread matters: it tells the model how much
to trust the mean, and a group with 3 observations behind it should not be read
like one with 300. The count is already computed as the smoothing denominator and
then thrown away.

### Gap B — no feature can look at another row in time

No lag, no rolling mean, no expanding mean, no days-since-last-event, no
days-until-next-event, no "was this group inactive yesterday". On a forecasting
problem that is most of the signal.

It is structural, not a missing convenience:

- `transform_features` receives the whole frame, so it *could* write
  `groupby().shift()`. **0011's stability check would kill it, correctly** — one
  row alone gets a different answer than the same row in a batch.
- At predict time the served rows are *later than every training row*. The lag has
  to be looked up from history the transform does not have.

**A lag needs the harness to carry a time series across the seam, and it can only
carry a mapping.**

### The general statement

> *A feature whose value depends on rows other than the current one, ordered by a
> known key, whose history is stored at fit time and replayed at predict time.*

No dataset in that sentence. It passes the `CLAUDE.md` test — any panel, sensor,
transaction, clickstream or subscription problem hits it. **Rossmann is the probe
that exposed it, not the reason to build it.**

---

## 3. The design questions, and which are blocking

### 3.1 What is stored — and it must be bounded

A per-group time series, truncated to the tail actually needed: for maximum
offset `L` and maximum window `W`, the last `L + W` observations per group.
Bounded by construction, which matters — 1,115 stores × 942 days is fine, but a
transaction table with 10M entities is not, and the tool must not have a size
that depends on the data's length.

Stored in the table's lineage like an encoder's mapping, so `Pipeline.transform`
replays it. Probably a new step type (`history`) rather than an overloaded
`encode`.

### 3.2 **BLOCKING — the horizon problem**

This is the one that needs Ulf's call before code is written.

At predict time the served rows are later than all stored history, often by an
unknown gap. **"Sales 7 days ago" is undefined for a row 60 days past the end of
training.** Four options, and they are genuinely different products:

| option | what it does | cost |
|---|---|---|
| **carry-forward** | serve the last known value | silently wrong, drifts without saying so — the worst kind of failure this project has |
| **last-known + staleness** | two columns: the most recent value, and how old it is | honest, always servable, and the model can learn to discount stale values. Not a lag. |
| **declare the horizon** | `TaskSpec` states the forecast gap; offsets below it are refused | correct, and pushes a task property to the task's owner where 0029 says it belongs |
| **refuse entirely** | only expanding/whole-history aggregates, no offsets | safe, and gives up most of the value |

**Recommendation to argue about: `last-known + staleness` as the default, with
`declare the horizon` available in `TaskSpec` when the owner knows it.** The two
compose — a declared horizon makes real lags legal, and staleness is what you get
without one.

### 3.3 **BLOCKING — may validation rows use their own past?**

At serving time you often *do* know yesterday's value. In this competition you do
not: the test set is a contiguous future block with no labels in it.

That difference is a property of **the serving contract, not the library** — which
means it hangs off the same concept as 0030's serving schema, and is probably an
argument the same machinery can already carry. Needs checking against
`_unavailable_at_predict_time`.

Get this wrong in the permissive direction and every validation score in every
future run is inflated, invisibly.

### 3.4 Leakage and folds

A strictly backward-looking lag of a **non-target** column is not leakage. A lag
of **the target** is target encoding's near relative and needs the same care:
out-of-fold construction, or an argument for why strict backwardness is
sufficient. `fit_group_stats`'s docstring already contains the reasoning to
reuse — "a value column correlated with the target makes its per-group mean
nearly a per-group mean of the target".

### 3.5 Why this is legal where `groupby().shift()` is not

Worth stating in the record when it is written, because it looks like a
double standard: the value comes from **stored state**, not from the frame in
front of it. That is exactly what makes `fit_group_stats` stable under 0011 while
`transform` doing the same arithmetic is not. One row alone and a batch of a
million get identical answers, because neither reads the other's rows.

---

## 4. Suggested slicing

Three chunks, smallest first, each with a record, tests and a notebook.

**Slice 1 — per-group target statistics beyond the mean (Gap A).**
Smallest, reuses the existing out-of-fold machinery, no new step type, no horizon
question. Extend the target encoder to emit `count` (already computed), `std`,
`median`. Ships value immediately and is a clean warm-up to the harder half.

**Slice 2 — the `history` lineage step (Gap B core).**
The new step type, bounded storage, replay at predict, the stability property
under 0011, and whichever answer 3.2 and 3.3 get. Lags and last-known/staleness.
This is the real build.

**Slice 3 — rolling and expanding windows on top of slice 2.**
Once history is stored and replayed, windows are aggregations over it. Cheap
given slice 2, and worth its own notebook.

---

## 5. Before writing code

- Settle **3.2** and **3.3** with Ulf. Everything else follows from them; nothing
  should be built first.
- Check whether `_unavailable_at_predict_time` / the serving schema (0030) already
  expresses 3.3, or needs extending.
- Confirm the stability check (0011) passes a stored-history step as-is, or what
  it needs.
- **Do not size the design against Rossmann.** The test is whether the tool's
  docstring can be written without naming a dataset. "Per-group history of a
  column, stored once and replayed" passes. "Sales for this store last week" does
  not.

## 6. What this is worth, honestly

Unknown, and larger than anything else on the board. For calibration, on Rossmann:

| lever | measured |
|---|---|
| blending instead of argmax | **+0.00184** on Kaggle (0044, 0045) |
| the whole feature search from t8 on | +0.0024 on the probe |
| **the single feature `t8` added** | **+0.0243 on the probe** |

`t8` is the one time the run said something new, and it was worth ten times
everything after it. Time-aware features are the next thing it has no way to say.
