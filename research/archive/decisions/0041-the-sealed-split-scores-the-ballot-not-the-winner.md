# 0041 — The sealed split scores the ballot, not just the winner

- **Date:** 2026-08-13
- **Status:** accepted, implemented
- **Implements:** [0023](0023-the-third-split-has-evidence-now.md) — whose
  decision stands unchanged; this records the four calls 0023 left to whoever
  built it
- **Related:** [0019](0019-the-leaderboard-is-a-public-leaderboard.md) (the two-split
  choice this revises the deferral of), [0029](0029-the-harness-owns-the-exam.md)
  (the exam is the harness's, and this adds a second one),
  [0039](0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md) /
  [0040](0040-seguro-v4-a-near-copy-of-the-ruler-is-not-the-ruler.md) (the
  hypothesis this makes measurable without a Kaggle submission)

## What 0023 decided, and what it did not

0023 decided the *what*: a sealed holdout no tool can reach, scored once at
`finalize_run`, reported beside `val_score` rather than instead of it, with the
gap between them as the diagnostic. That is unchanged and is not re-argued here.

Four things it did not settle, all of which a reader of the code would otherwise
have to reverse-engineer from the parameter list.

## 1. Opt-in per run, off by default

`RunSession(holdout_fraction=...)`, `None` unless a driver script passes it.

The alternative — on by default, skipped below some row count — is what 0023's
"optional, and off by default below some size" sounds like, and it is wrong for
a reason that only shows up in this repo's workflow. **Turning the third split on
takes rows out of the work half, so the first run with a holdout is not
comparable to any run before it.** House v9 beat v8 by a number measured on 1168
work rows; a v10 cut at 876 would be a different experiment wearing the same
name. Every A/B this project has run has been pre-registered against a specific
earlier run, and a library default that silently re-cuts the data breaks that
discipline without appearing in the diff of the driver script.

So the cost is visible where the decision is made. The price is that it only
happens when someone remembers.

## 2. A fraction of the whole input, not a slice of the validation half

`n_holdout = round(n * holdout_fraction)`, `n_val = round(n * val_fraction)`,
work is the remainder. At 0.2/0.2 house-prices becomes 876 / 292 / 292.

The rejected alternative was to carve the holdout out of the existing held-out
20%, leaving the work half untouched — which sounds free, and is not: it halves
the validation half, and `val_se` grows by ~1.4x on the number that ranks every
table in the run. 0022 exists because a 0.008 gain on a score with a 0.012
standard error was celebrated. Widening that yardstick to buy a diagnostic would
be paying for honesty in reporting with noise in selection.

There is a second, quieter payoff. Under a **random** split the validation rows
are taken off the front of one permutation, so *which rows are validation depends
only on `n` and the seed* — not on whether a holdout was asked for. Turning the
seal on leaves the validation half byte-identical and shrinks only the work half.
Every val-denominated comparison keeps its meaning.

## 3. Ordered splits seal the *latest* period, and pay for it

`work | val | holdout` in time order. The sealed slice gets the last rows,
because it is the one standing in for unseen data and unseen data is always the
future — the same argument 0029 makes for the validation half, applied one slice
further out.

Two consequences, both accepted deliberately:

- **The comparability property from §2 does not hold here.** Adding a holdout
  moves the validation rows earlier. An ordered run with a seal is not
  row-for-row comparable to one without.
- **The gap is not purely selection.** val and holdout are *different periods*,
  so their difference mixes how much the search gamed the validation half with
  how much the world moved in between. The report says so, in the run's own
  words, whenever `split_on` is set. Rossmann v5 already showed the sign can flip
  from this alone: its ordered exam was **pessimistic** by 0.0026 where every
  random-split run had been optimistic.

The alternative — holdout in the middle, val last — preserves comparability and
destroys the number: the sealed rows would sit *closer in time* to the training
rows than the validation rows do, so the sealed score would be the optimistic one
and the gap would be measuring the wrong sign of the wrong thing.

Mechanically this needed one change outside the split: `boundary_cut` takes an
`exclude`, because two honest boundaries landing in the same place is an empty
slice, which looks exactly like a split that worked.

## 4. Every terminal candidate is scored, not only the winner

This is the substantive addition to 0023, and the one most in tension with it.

0023 says "`finalize_run` reads it once". Read strictly that is one model, one
number. What is implemented scores **the shipped model and the terminal
candidates it was picked over** — currently four fits — in one pass, and writes
them all under `holdout` in the journal.

The argument for: the fits already exist, so the numbers are free, and they are
exactly the measurement the last three records could not get. 0039 and 0040
argued that optimism attaches to the **selected measurement** rather than to the
model or the table, on three pairs — and named the confound in advance, because
in every pair the selected model was lgbm and the unselected one elasticnet. A
winner and its runners-up, same table, same sealed rows, is that comparison with
the confound removed, and it costs a Kaggle submission today.

The argument against, which is real: a number in the journal is a number someone
will eventually select on. Four scores on a sealed set, ranked, is a third
leaderboard waiting to happen — and the moment anything ranks by them, 0023 has
bought nothing and the project has three splits with two splits' worth of
protection.

Three things hold the line, and they are structural rather than advisory:

- `_score_on_holdout` is called **after `best` is read** in `finalize_run`. The
  winner is an argument to it. There is no ordering in which it could influence
  the choice.
- The scores are written under `journal["holdout"]`, **never onto the
  leaderboard**, and `journal.set_holdout` is **write-once** — so "scored exactly
  once" is auditable from the journal alone, not a property of the caller.
- Each entry carries `selected: bool`. The distinction between "the number the
  search steered towards" and "the number that judged it" is on the row, and the
  report prints, under the table: *these rows ranked nothing*.

Models are **not refitted** on work+val before being scored. Each is scored
exactly as it shipped, fitted on the work half alone, so the difference between
its `val_score` and its sealed score is two sets of rows and nothing else.
Refitting would produce a better model and a meaningless gap.

## What it costs

**Good**

- The report answers "how much of this is real" with a number, for the first time
  and without a Kaggle account. Every optimism figure this project has (house v8
  +0.01127, v9 +0.00458, seguro v2 +0.01828, v4 +0.00838) came from a submission
  to an external leaderboard, on the two datasets that happen to have one.
- The selected-vs-unselected pairs arrive as a byproduct of every sealed run,
  which is what would let 0039's hypothesis be tested on a dataset with no
  competition attached.
- `_score_on_val` became `_score_on_held_out(rows_path=...)` with a thin wrapper.
  The sealed score is computed by *the same code* as the validation score, which
  is what makes the gap attributable to the rows.

**Bad**

- **Rows.** On house-prices the work half drops 1168 → 876 at 0.2. The agent
  builds on 25% less data, and 0033's own finding — a linear model at ~930 rows is
  on a steep learning curve — says that is not free.
- **Two exams to keep aligned.** Anything that changes how `val_score` is
  computed now has to change how the sealed score is computed, or the gap starts
  measuring the difference between two procedures. The shared function is the
  mitigation and it is not a guarantee.
- **The seal is a convention enforced by an allowlist, not by physics.**
  `resolve_table` admits nothing outside `tables/`, which is the same defence the
  validation half has had since 0019 — but the labels are on the same machine, in
  the same run directory, in a file the agent could name if the allowlist ever
  regressed. `wandb_fc/tabular-nomad` keeps its held-out labels on a *different
  machine*. That is the stronger version and it is not built.
- **A single sealed score on a few hundred rows has its own standard error**, as
  0023 already noted. `holdout_is_thin` flags it; it does not fix it.
- **No run has used it yet.** This is machinery and a test suite, not a
  measurement. The first sealed run is the thing that makes it worth anything.

## Where it lives

- `src/tabular_agent/harness/session.py` — `holdout_fraction`, the three-way
  `_ensure_split`, `holdout_path`, `n_holdout`/`holdout_is_thin` in the split
  record.
- `src/tabular_agent/cv.py` — `boundary_cut(exclude=...)`.
- `src/tabular_agent/harness/tools.py` — `_score_on_held_out` (generalised),
  `_score_on_holdout` (the one reader), `finalize_run` wiring.
- `src/tabular_agent/harness/journal.py` — `set_holdout`, write-once.
- `src/tabular_agent/report.py` — the sealed score in the header, and the
  "What the search cost" section.
- `tests/test_holdout.py` — 17 tests, organised as the three claims: unreachable
  during the run, read once and only after the winner is picked, and does not
  silently re-cut what earlier runs were measured against.
