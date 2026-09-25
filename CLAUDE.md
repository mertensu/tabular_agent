# tabular_agent

Tabular-ML agent with a leakage-safe kernel. Start at `README.md`;
`docs/CURRENT.md` is the current snapshot. New machine: `docs/SETUP.md` (env, API
key, the untracked Kaggle datasets).

## Decision records

`docs/decisions/` holds the records that still carry the product; the rest are in
`research/archive/decisions/` with the full index. Numbers are one sequence across
both; new records go in `docs/decisions/`. Start at `docs/decisions/README.md`.

- Read them before changing an area you did not build: they explain *why* the
  code has its shape.
- Add one only for a product contract, safety invariant, architecture boundary,
  or a rejected alternative likely to be rebuilt. Measurements and dataset
  findings go in one directory under `research/`.
- Never edit an accepted record. If a decision is revised, write a new record
  that supersedes it.
- They are not documentation of current behaviour; that lives in docstrings and
  comments.

## Datasets are probes, not targets

Real datasets are run to find out what the harness cannot express. A run's score
is a reading, not a goal.

- **Nothing dataset-specific goes into `src/`.** A capability added because a run
  exposed a gap must be a gap any tabular problem could hit. Test: its docstring
  can be written without naming the dataset ("some rows should not be trained
  on", not "exclude days the store was closed").
- Dataset knowledge lives only in `scripts/prep_*.py` and in
  `TaskSpec.notes` / `column_notes`.
- Do not tune the library against a leaderboard: a change that improves one score
  and cannot be defended on its own terms is leaderboard-fitting (0022).
- A real finding can still be out of scope (e.g. external data, 0001).

## Working with Ulf

Explain meaningful product changes in plain terms. Do not create notebooks or
decision records by default.

- A notebook only for a major new product concept or an archived research
  experiment; extend an existing one first.
- One research experiment lives in one directory under
  `research/archive/experiments/`, not spread across `scripts/`, `results/`,
  `notebooks/` and several decision records.

## Multi-machine workflow

Worked on from two Macs; `HANDOFF.md` is the live state between them.

- Session start: `git pull`, then read `HANDOFF.md`.
- Session end: replace its Current state / Next action / Blocking gotchas /
  Uncommitted work / Latest session sections, move the closed entry verbatim to
  `research/archive/handoffs/`, and push (WIP branch if master shouldn't hold it).
