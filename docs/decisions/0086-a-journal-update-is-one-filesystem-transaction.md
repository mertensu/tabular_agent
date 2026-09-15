# 0086 — A journal update is one filesystem transaction

- **Date:** 2026-09-15
- **Status:** accepted, implemented
- **Supersedes:** the journal-write mechanism described in
  [0031](0031-the-guard-was-decorative.md), not its decision that correctness
  belongs in filesystem primitives.
- **Related:** [0003](0003-file-based-tools.md),
  [0004](0004-tools-write-the-journal.md),
  [0008](0008-serialise-tool-calls.md)

## What happened

A real run ended with seven non-JSON bytes after the closing journal object:
a closing brace followed by 3, another brace, and a second closing brace.
Re-running succeeded. The file was not merely stale; two writes had interfered.

The journal already used flush, fsync, and atomic replace, but every writer used
the same sibling path, run_journal.json.tmp. Atomic replacement protected a
reader from a half-renamed destination. It did not stop two writers from opening
and writing the same temporary inode.

There was a second race even if every temp name were unique. Every mutator did:

1. load the journal,
2. alter its private dictionary,
3. save the whole dictionary.

Two callers could load the same version and each atomically replace it. Both
files would be valid JSON, but the later replace would silently erase the other
fact. A valid journal with missing experiments is still a corrupt run record.

## Decision

Every journal mutation holds an advisory exclusive lock on
.run-journal.lock across the entire load, mutation, fsync, and replace.
fcntl.flock applies across threads and processes on the supported Unix
runtimes and is released automatically if a process dies.

Each physical write also gets a unique hidden temporary file from
tempfile.mkstemp in the run directory. It is fsynced and atomically renamed,
and cleaned up if serialization or replacement fails. The lock preserves all
updates; the unique temp name independently prevents two full-state writers
from sharing bytes.

Readers remain lock-free. Atomic rename means they see one complete version or
the next, and blocking every inspection behind a writer would add latency
without increasing correctness.

The public full-state save remains for initialization and tests and takes the
same lock. Product updates use the dedicated mutators, whose decorator holds the
lock before they load.

## Rejected

**Unique temp names only.** This fixes malformed bytes but still loses one of two
concurrent read-modify-write updates.

**An in-process threading lock.** Tool calls can come from another process, and
the run directory is deliberately resumable outside one Python object.

**Locking readers.** Replacement is already atomic for readers; a shared read
lock would add coordination to the hot inspection path without closing another
failure mode.

**Retrying JSON parse errors.** That hides evidence after corruption and cannot
recover a fact already overwritten.

## Cost

Journal writes are serialized per run directory and leave one tiny lock file
beside the journal. Writes were already fsync-bound and small, so the added
critical section is negligible compared with feature and model work. Different
runs use different lock files and remain independent.

The regression test launches 64 concurrent mutations, requires every unique fact
to survive, reparses the final file, and asserts that no temporary journal file
remains.
