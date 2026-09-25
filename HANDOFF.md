# Handoff

## Current state

Preparing the repository for publication (it is still **private** on GitHub).

- History rewritten 2026-09-24: the committed Rossmann competition data and the
  0080–0081 artifact archive (competition validation rows) are gone from every
  commit. Only `master` and tag `research-reference-2026-08-21` remain on GitHub.
- MIT license added (`LICENSE`, `pyproject.toml`, README section).
- Decision records trimmed to the 12 that still carry the product in
  `docs/decisions/`; the other 80 moved unchanged (link paths only) to
  `research/archive/decisions/` with the full old index. Numbering stays one
  sequence; new records go in `docs/decisions/`.

## Next action

Work through `docs/pre-publication-plan.md` in its suggested order: Windows note,
sandbox hint and vocabulary, round agent-facing scores to 3 decimals, ruler
sentence, sealed split in the product run, then one findings view (needs Ulf's
Option A/B call on `explore_table`).

Also still open from the first review: implicit Requesty routing for any `a/b`
model string; README table says `claude-sonnet-4-6` while `DEFAULT_MODEL` is
`anthropic:claude-opus-5` (check the ID exists); README "Long runs stay robust"
cites 0024/0025, which the old index marks as not implemented; absolute
`/Users/ulfmertens` paths in `research/archive/runs/house-prices/*.py`; a small
preregistered external comparison.

Carried over and status unknown: the three predeclared Rossmann
`ForecastingAgent` repetitions from the 2026-09-17 session.

## Blocking gotchas

- **Other Mac: do NOT pull.** History was rewritten. Move the old clone aside,
  re-clone, then copy the untracked dataset folders and `config.env` across.
  Pulling or merging into the old clone reintroduces the removed data.
- The 28 deleted side branches (15 had unmerged work) and the pre-rewrite
  history are only in `~/Documents/tabular_agent-pre-rewrite-2026-09-24.bundle`
  on the MacBook Air. It contains the competition data: keep it private.
  Recover a branch with `git fetch <bundle> <branch>:<branch>`.
- On this Mac, `refs/stash` and `refs/claude/checkpoint-9e2193fb` still pin the
  old objects in `.git` (deletion was blocked in-session); both are in the bundle.
- GitHub may keep serving old commits by SHA until it garbage-collects; ask
  GitHub Support to purge before making the repo public if that matters.

## Uncommitted work

- `notebooks/concepts/` and `zomato_model_comparison.ipynb` are untracked (the
  concept notebook's decision links were updated to the archive paths).

## Latest session (2026-09-24, MacBook Air)

Pre-publication review: history rewrite, MIT license, decision records trimmed
to 12. Tests were not re-run; the changes touched no code.
