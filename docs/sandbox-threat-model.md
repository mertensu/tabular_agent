# Sandbox threat model

A note, not a decision. It records what the sandbox around agent-written code
protects against, what it does not, and a hardening plan that was worked out and
deliberately **not built** (2026-09-25).

## What runs where

The agent writes pandas code for three tools: read-only exploration, row-local
transforms, and train-only row filters. `sandbox.py` runs each call in a fresh
Python subprocess with:

- CPU, memory and wall-clock limits
- its own temporary directory as `cwd`, `HOME` and `TMPDIR`
- an allowlisted environment (`_child_env`), so exported API keys are unreadable
- an import hook that blocks model libraries with a hint to use the fitted tools

Bundle replay goes through the same `run_transform`, so serving inherits all of it.

## The threat worth taking seriously

Not a malicious user — it is their machine and their data. The threat is **a
model steered by its input**. `inspect()` passes a data description and cell
values to the LLM, and the LLM writes code that runs as the user. A crafted
description could make the model write code that:

1. **reads secrets** from the environment — *closed* by the empty environment
2. **reads files** the user can read, including `train.csv` (target-blindness is
   bypassed) and `run_dir/data/val.parquet` (validation rows are only kept from
   the *tools*, not from the disk) — **open**
3. **sends data out** over the network — **open**

A weak or confused model can reach 2 by accident; 3 needs intent, which is what
injection supplies.

Blocking imports cannot close 2 or 3: the standard library is allowed, and
pandas reads paths and URLs by itself (`pd.read_csv("http://…")`). Any block has
to sit below Python.

## The plan, if it is ever needed

In order of value per effort:

1. **OS sandbox for the child.** Deny network; allow file reads only of the
   Python installation and the work directory, writes only in the work
   directory, and no process launch other than the interpreter.
   - macOS: `sandbox-exec` with a Seatbelt profile (deprecated but used by
     Chrome, Bazel and coding-agent CLIs for exactly this).
   - Linux: `bwrap --unshare-net` with the interpreter bind-mounted read-only and
     the work directory read-write; Landlock + seccomp as the dependency-free
     alternative.
   The child needs little: `frames.pkl` and `config.json` in, `result.json` out.
2. **Fail closed.** Where no OS sandbox is available, refuse to run agent code
   unless the caller passes an explicit opt-out, and record in the report which
   mode ran.
3. **Attack tests as the contract.** Agent code that reads `train.csv` by
   absolute path, opens a socket, fetches with `urllib` *and* with
   `pd.read_csv(url)`, reads the environment, spawns a process, or writes outside
   the work directory — each must fail. The README may only claim what these
   tests prove. (The environment test exists: `test_child_cannot_read_caller_secrets`.)
4. **Reframe the import hook as teaching.** It stays: it steers the agent to the
   fitted tools. It is not a security layer and should not be described as one.

Rejected:

- **A restricted-Python checker or a mini-language** as the security layer.
  Python has too many escape routes (dunder walks, pandas' own I/O,
  `df.eval`/`df.query`), and a mini-language would cap what the agent can
  express.
- **Prompt-level defences** (marking the description as untrusted). Cheap
  hygiene, no guarantee. The goal is that a successful injection has nothing to
  reach.

## Why it is deferred

The expected user base is small and runs the tool on their own data. For them
the two changes that do not depend on scale were made: the empty environment
(protects each user's own keys whatever the user count) and README wording that
claims only what holds.

Revisit when any of these becomes true:

- the tool runs on data its operator did not produce (uploads, scraped tables)
- it runs as a service, or on shared machines
- someone reports reading files or the network from agent code in practice

Validation rows being readable from disk is the one gap that affects *score
honesty* rather than security. If a run is ever suspected of it, the recorded
transform and filter code in the journal is where to look.
