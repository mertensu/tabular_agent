# tabular_agent

Autonomous tabular-ML product with a leakage-safe internal kernel. Start at
`README.md`; `docs/CURRENT.md` is the authoritative current snapshot.

**New machine?** `docs/SETUP.md` gets a fresh clone running — env, API key, and
the deliberately untracked Kaggle datasets.

## Decision records

`docs/decisions/` holds one file per design decision — the problem, what was
chosen, what was rejected, and what it cost. Start at `docs/decisions/README.md`,
which also defines the project vocabulary (run, table, experiment, journal,
lineage, bundle). It holds only the records that still carry the product; the
rest — measurements, refuted mechanisms, superseded plans — are in
`research/archive/decisions/` with the full index. Numbers are one sequence
across both; new records go in `docs/decisions/` and continue it.

- **Read them** when touching an area you did not build: they explain *why* the
  code has the shape it has, which the code itself cannot.
- **Add one only** for a product contract, safety invariant, architecture boundary,
  or rejected alternative likely to be rebuilt. Per-run measurements, threshold
  probes and dataset findings belong in one directory under `research/`, not in
  the global decision sequence.
- **Never edit an accepted record to match new reality.** They are history. If a
  decision is revised, write a new one that says which it supersedes and leave the
  old reasoning intact — being able to see what we believed and why is the point.
- They are *not* documentation of how the system currently works. Per-line
  rationale belongs in docstrings and comments, where it cannot drift.

## Datasets are probes, not targets

We run real datasets (house-prices, Rossmann) to **find out what the harness
cannot express**. The dataset is the instrument; the library is the subject. A
run's score is a reading, not a goal.

So: **nothing dataset-specific goes into `src/`.** Every capability added because
a run exposed a gap has to be justified as a gap *any* tabular problem could hit,
and named in those terms. If the argument for a feature is "it would help on
Rossmann", it is not an argument.

The test is whether the feature's docstring can be written without naming the
dataset. "Per-group aggregates of a column, fitted once and replayed" passes.
"Exclude rows where the store was closed" does not — the general form is "some
rows should not be trained on", and that is what gets built.

Dataset-specific knowledge has exactly two sanctioned homes, both outside the
library: `scripts/prep_*.py` (the join, the typing, the data dictionary, the
metric and split for *that* competition) and `TaskSpec.notes` / `column_notes`,
which the data's owner fills in per run.

Two things that follow, and both have already been tempting:

- **Do not tune the library against a leaderboard.** A change that improves one
  score and cannot be defended on its own terms is leaderboard-fitting, which is
  the failure 0022 exists to record — one level up.
- **A finding can be real and still not be ours to fix.** Rossmann's winner used
  external weather data; that stays out (0001). The 0.985 prediction multiplier
  stays out because its own author reports the effect was inside the CV noise.

## Keep Ulf in the loop without growing the product surface

Ulf is building this to *understand* it, not just to have it built. Explain every
meaningful product change and ask a short concrete quiz question about its design
call. Do **not** create a notebook or global decision record by default.

- A notebook is justified only for a major new product concept or a deliberately
  archived research experiment. Extend an existing chapter before adding one.
- Research artifacts are bundled by question under `research/archive/experiments/`;
  never spread one experiment across root `scripts/`, `results/`, `notebooks/` and
  multiple global decisions.

## Multi-machine workflow

This repo is worked on from two Macs. `HANDOFF.md` is the short live state that bridges them:

- **At session start:** run `git pull`, then read `HANDOFF.md` to learn what was done last (possibly on the other machine) and what's next.
- **Before the session ends:** replace its fixed Current state / Next action /
  Blocking gotchas / Uncommitted work / Latest session sections. Move the closed
  entry verbatim to `research/archive/handoffs/`; do not accumulate a second
  history in the live file.
- Always push at the end of a session, even work-in-progress (use a WIP branch if master shouldn't hold it).
