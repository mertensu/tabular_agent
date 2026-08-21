# tabular_agent

Leakage-safe tabular-ML core library. See `docs/` for Phase-2 requirements.

**New machine?** `docs/SETUP.md` gets a fresh clone running — env, API key, and
the Kaggle datasets (all three are deliberately untracked).

## Decision records

`docs/decisions/` holds one file per design decision — the problem, what was
chosen, what was rejected, and what it cost. Start at `docs/decisions/README.md`,
which also defines the project vocabulary (run, table, experiment, journal,
lineage, bundle).

- **Read them** when touching an area you did not build: they explain *why* the
  code has the shape it has, which the code itself cannot.
- **Add one** when a decision is made that a newcomer would otherwise have to
  reverse-engineer — especially when an obvious-looking alternative was rejected.
  Keep it to the existing template and write for a data scientist who has never
  seen this repo.
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

## Keep Ulf in the loop (quizzes + notebooks)

Ulf is building this to *understand* it, not just to have it built. Code that lands
without him following it is a failure even if the tests pass. So, for every
meaningful chunk:

- **Quiz him on it.** Short, concrete questions about the chunk just built — the
  design call and the failure it prevents, not trivia. Ask them one at a time and
  wait for the answer.
- **Ship a notebook with it**, in the style of `notebooks/phase3_harness_step1.ipynb`:
  drive the new machinery by hand first, LLM/automation only at the end; ships
  unexecuted, but every code cell smoke-tested in order beforehand. One notebook
  per slice — extend the existing one if the chunk is a direct continuation of it,
  otherwise start a new one.

## Multi-machine workflow

This repo is worked on from two Macs. `HANDOFF.md` is the session log that bridges them:

- **At session start:** run `git pull`, then read `HANDOFF.md` to learn what was done last (possibly on the other machine) and what's next.
- **Before the session ends** (or after completing a meaningful chunk of work): add a new entry at the top of `HANDOFF.md` — date, machine, branch, what was done, what's next, any gotchas — then commit and push so the other machine sees it.
- Keep entries short; prune entries older than ~2 weeks.
- Always push at the end of a session, even work-in-progress (use a WIP branch if master shouldn't hold it).
