# Handoff

## Latest session (2026-09-15)

Fixed concurrent journal corruption on branch `chad/journal-transaction-lock`,
based on `origin/master` at `1780c7b`.

- every product journal mutator now holds `.run-journal.lock` across the complete
  load, mutation, fsync, and atomic replace
- the lock uses `fcntl.flock`, so independent threads and processes sharing one
  run directory serialize their updates; separate runs remain independent
- every physical write uses a unique hidden temp file in the run directory, so
  even full-state writers can never interleave bytes in one temp inode
- readers remain lock-free because `os.replace` still exposes only a complete
  old or new JSON document
- the public full-state `journal.save` remains for initialization and tests; the
  dedicated product mutators are the safe read-modify-write API

The triggering House Prices A/B run ended with trailing non-JSON bytes. The old
implementation used one shared `run_journal.json.tmp` path and protected only
the destination rename, not the preceding read-modify-write transaction. That
allowed both malformed bytes and valid-but-lost facts.

Decision record:
`docs/decisions/0086-a-journal-update-is-one-filesystem-transaction.md`.
The previous handoff is archived verbatim under
`research/archive/handoffs/2026-09-14--2026-09-15.md`.

## Verification so far

- full repository suite: 612 tests pass, exit 0
- focused journal/holdout/resume suite passes
- a new stress test launches 64 concurrent journal mutations and proves all 64
  facts survive, the final file reparses exactly, and no temp files remain
- an additional direct 200-writer stress check passed before the test was added
- `git diff --check` passes

## Next action

Review and merge `chad/journal-transaction-lock`. No model, feature, score, or
report behavior changed.

## Blocking gotchas

- `.run-journal.lock` is a durable coordination file, not stale lock state;
  `flock` attaches to the open descriptor and is released on close or process
  death, so the file should not be deleted as cleanup
- readers must stay on the atomically replaced `run_journal.json`; do not make
  them inspect a temp file or require the write lock
- new journal mutations must use a decorated mutator or explicitly hold
  `_write_lock` across load and `_save_unlocked`; calling load and public save as
  two separate operations does not create a transaction
- the package already depends on Unix `fcntl` for minimal finalization; this
  change does not introduce a new platform assumption
