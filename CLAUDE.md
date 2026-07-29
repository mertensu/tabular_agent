# tabular_agent

Leakage-safe tabular-ML core library. See `docs/` for Phase-2 requirements.

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
