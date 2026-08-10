# 0019 — The input is split, and the leaderboard ranks by the held-out half

- **Date:** 2026-08-10
- **Status:** accepted
- **Decided by:** Ulf (the Kaggle framing, and the call to rank by validation)
- **Related:** [0012](0012-harness-owns-the-stop.md),
  [0014](0014-two-level-stop.md), [0017](0017-credit-by-ablation.md)

## The problem

`best` was the maximum `cv_score` over every experiment in the run. Each of those
scores is a fine estimate of one model's generalisation, taken on its own. The
maximum of many of them is not: it is biased upward, and the bias grows with the
number of experiments.

That was tolerable at ten experiments and is not at the target of 100-200 in one
overnight run. Two things make it concrete rather than theoretical:

- A `city` column built from `rng.choice` — independent of the target *by
  construction*, containing no signal at all — cleared the bar and shipped as
  `best` after **five** experiments.
- lgbm on the raw table beat the trivial baseline by 0.0022 with a fold spread of
  0.0416. The real effects being chased are an order of magnitude smaller than
  the noise being maximised over.

And the search is adaptive, which is the worse half: the agent reads `cv_score`
and picks its next features from it. After 150 rounds, the folds are not
independent of the search path that produced the winner. This is the same failure
already logged for residual feedback and for the metadata kernel — **the leak is
through the model's reasoning, not through the data**, so every leakage guard in
the library holds while it happens.

## The decision

The input table is split **once, before the first tool call**, into

    run_dir/data/work.parquet    the agent's world — CV runs inside it
    run_dir/data/val.parquet     never handed to a tool; scored by the harness

`table=None` means the work half. Every experiment is scored on the validation
half by replaying its table's lineage over those raw rows, and **the leaderboard
ranks by `val_score`**. `cv_score` stays on the row beside it.

The split is time-ordered when a `split_on` column is given, random-but-seeded
otherwise. It is recorded in the journal (`split`), because two runs over the
same file are only comparable if they validated against the same rows, and a run
directory can be deleted.

## What this is, and what it is not

It is a **public leaderboard**, in the Kaggle sense, and the report says so in
those words. The agent sees `val_score` on every training result and steers by
it. After 150 experiments those rows have been consulted 150 times, so the final
number is the score the search was aiming at — not an independent check on it.

It is **not** a private leaderboard. There is no second held-out set, so nothing
in this run measures how much the search overfit its own feedback. That was a
deliberate call, taken with the alternative on the table.

## What was rejected

**Rank by `cv_score`, seal the validation half, score it once at finalize.**
The clean version: the reported number would be an honest audit of the winner,
and the agent's per-experiment signal would stay CV. Rejected because it gives
the agent no held-out signal at all during the night, and the point of the run is
that the agent gets better as it goes.

**Three splits — work / val / test**, val queried freely as a public leaderboard,
test scored once at finalize as the private one. This is the structure that
delivers both, and it is what Kaggle actually does. Rejected on cost: it spends
another ~15% of the rows on instrumenting a process that is not being iterated
on across runs. One dataset, one overnight run, ends at `finalize_run`.

**Two things worth writing down about the rejected option**, because the reasons
for it have not gone away:

- The single number this run reports cannot be checked by this run. If a future
  run wants to know whether 150 experiments of searching produced real
  improvement or elaborate noise-fitting, it needs the third split.
- A single 20% validation split is a *noisier* estimator than 5-fold CV on the
  same data — CV validates every row exactly once, so it rests on all *n* rows
  where validation rests on 0.2*n*. Ranking by `val_score` therefore trades some
  precision for independence from the folds. That trade is the whole point, but
  it is a trade.

## What was deliberately left on cross-validation

`explain_table` still ablates on CV while everything ranked moved to `val_score`.
Kept there because it ranks nothing — it writes no leaderboard row and picks no
winner — and because CV rests on all the work rows, making it the lower-variance
instrument for the question it answers, which is "what are these columns worth",
not "which experiment wins".

The cost is a real one and is the reason this needed deciding rather than
drifting: contribution is now quoted in a different currency from the number the
run is judged by. A table can measure `+0.10` by ablation and move `val_score`
not at all, and the agent has no field that says so. What limits the damage is
that `cv_score` and `val_score` sit side by side on every training result, so a
fold-only gain is visible one call later — but the agent has to notice it there,
not in the tool that invited it to keep the columns.

## What it cost

- **Every score in the repo moved.** Models now train on 80% of the rows.
- **`resolve_table` became an allowlist.** It used to accept any path that
  existed, including absolute ones, which made `available_tables()` a display
  convention rather than a boundary. With validation rows on disk inside the run
  directory, "the agent has no reason to read them" is not the same guarantee as
  "the agent cannot".
- **`finalize_run` is now genuinely once.** The finalized check in
  `_refuse_if_finished` sat *after* an early return that fired whenever the stop
  rule said keep going, so the guard only worked on a run the stop rule was
  already refusing — a finalized run with budget left could train again, and
  finalize again.

  Note what this is and is not. With `val_score` on every training result the
  agent already queries the validation half continuously, so a second
  `finalize_run` is not the thing eroding it; the erosion is the design. What the
  guard buys is **run integrity** — one deliverable, and a `run_report.md` that
  describes a leaderboard which still exists — plus the invariant that a
  finalized run is over. It would become a leakage guard the moment a sealed
  third split is added, which is precisely when nobody would think to re-check
  this path.
- **A floor, but a low one.** Fewer than 5 rows in either half is refused, because
  no metric can be computed. A *thin* validation half (under 30 rows) is flagged
  in the split record and warned about in the report, not refused: how many rows
  make a score trustworthy is a judgement about the data, and a 90-row dataset is
  small rather than invalid. The first draft refused below 25 a side and would
  have turned the library away from data its owner had every right to model.
- **A missing `val_score` cannot be `best`.** When replaying a table's lineage
  over the validation rows does not reproduce the model's columns, the row keeps
  its `cv_score` and records why, and is not eligible to win.
