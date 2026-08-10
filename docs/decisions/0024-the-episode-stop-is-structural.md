# 0024 — The episode stop is structural, and the brief is not optional

- **Date:** 2026-08-10
- **Status:** accepted, **not yet implemented** (the loop lives in
  `scripts/run_house_v5_episodes.py`, not in `harness/`)
- **Revises:** the parked episode design in the 2026-08-06 handoff entry — its
  mechanism stands, its allocation of the episode stop does not
- **Related:** [0012](0012-harness-owns-the-stop.md) (the run stop),
  [0020](0020-the-sandbox-has-no-kernel.md) (why `run_dir` is the state),
  [0003](0003-file-based-tools.md)

## The problem

There is no compaction, so a single `run_sync` driving 150 experiments hits the
model's context window and dies. Measured on three live runs, context grows
linearly at ~830–990 tokens per turn and peaks at 29k–44k over 15–24
experiments; extrapolating to 150 puts it past any window, and because the whole
history is re-sent every turn, the *cost* grows quadratically.

The fix has never been a bigger window. It is a driver loop where each iteration
is a **fresh agent on the same `run_dir`**:

    while not journal.load(run_dir)["stop"]["finalized"]:
        agent = build_agent(model)          # new context every time
        agent.run_sync(EPISODE_PROMPT, deps=Deps(session=session))

`RunSession` holds only pointers, and its docstring has promised this since
phase 3. This is the first thing that relied on it, and it worked.

## What the parked design got wrong

The 2026-08-06 note allocated the two stops like this:

> the *run* stop is a correctness concern so 0012 rightly gives it to the
> harness; the *episode* stop is only context hygiene, cheap and reversible, so
> the agent can own it

The premise is correct — an episode boundary really is cheap and reversible.
The conclusion does not follow, and a run tested it directly.

**v4**: the agent was asked, in plain terms, to "follow one idea far enough to
know whether it worked, then stop and summarise." It ran **25 experiments** — the
entire `max_experiments` budget — in one episode, then finalized. There was never
a second episode. That an agent *may* own the episode stop does not mean it
*will exercise* it, and this project has recorded the same under-compliance with
advisory guidance since July: the t6 redirect, "features over tuning", the
kitchen-sink transforms. Guidance under-complies until something structural
intervenes.

## The decision

**1. The episode boundary is a hard cap the harness enforces.**
`UsageLimits(tool_calls_limit=N)`, with the resulting exception treated as the
boundary rather than as a failure. This is safe for exactly the reason the
architecture exists: an episode cut mid-thought loses a *conversation*, and the
run directory does not care. Nothing durable is in the context (0020).

The agent may still end an episode early by replying without a tool call —
`run_sync` returns and the loop continues. Structural cap, advisory floor.

**2. A resumed agent orients itself, and it does so with `explore_journal`.**
Given a prompt that says only "earlier episodes did things you have no memory
of" — no counts, no scores, and **no mention of the tool** — every resumed
episode opened the same way:

    ep2:  inspect_data -> explore_journal -> explore_journal -> inspect_data -> train
    ep3:  inspect_data -> explore_journal -> explore_journal -> train -> explore_experiment
    ep4:  inspect_data -> explore_journal -> explore_journal -> explore_experiment -> ...

Its first query is the obvious one — dump `run`, `lb` and `tables` and look. So
`explore_journal` is discoverable from its docstring alone, and resumable memory
works.

**3. The derived brief is required anyway.** This is the part that reverses an
expectation held while setting the experiment up ("if it queries its history, the
brief may be unnecessary"). Orientation *works* but it is **not free**:

| episode | tool calls | spent orienting | new experiments |
|---|---|---|---|
| 1 | 7 | 0 (nothing to orient to) | 2 |
| 2 | 7 | 3 | 2 |
| 3 | 11 | 3 | 1 |
| 4 | 9 | **8** | **0** |

Episode 4 called `explore_journal` seven times and trained nothing — it spent the
whole episode reconstructing where it was, and the loop guard had to stop it.
That is not "re-derives an idea"; it is a degenerate mode where orientation
consumes the episode. The tax is roughly a fixed 2–3 calls, so it is proportional
to `1 / episode_length` and becomes ruinous exactly where short episodes are most
wanted.

A ~120-token brief rendered from the journal — task, counts, baseline, best,
`since_improvement`, not-finalized — removes the first query. The brief and the
tool are **complementary**: the brief answers what is always needed, the tool
answers what a brief cannot anticipate.

**4. A loop guard is mandatory, not a nicety.** An episode that adds no
experiment means the loop is spinning, and spinning bills all night. It fired on
the very first experiment run.

## What it costs

**Good**

- **Peak context flattens.** Per-episode peaks of 9,170 / 11,170 / 12,274 / 9,971
  against v3's single climb to 43,691 — a 3.6x lower peak that, crucially, does
  not grow with run length. Linear becomes sawtooth.
- Resumable memory is real, and it needed no new mechanism beyond
  `explore_journal` and the run directory.

**Bad**

- **The orientation tax is the whole trade.** Episodes buy bounded context and
  pay in redundant re-derivation. Below some episode length the tax exceeds the
  work, and this run found that regime immediately at 6 calls.
- **A degenerate orientation mode exists** and only the loop guard catches it.
- **`SYSTEM_PROMPT`'s "`inspect_data` first, always" is a run-level rule being
  obeyed per-episode.** Every episode opened with a redundant `inspect_data`; the
  task and metric are already in the journal. One wasted call per episode, and it
  is the prompt's fault, not the agent's.
- Episode length remains "a seam, not a count" — a cap will sometimes cut
  mid-idea, and this experiment did not isolate what that costs, because
  orientation overhead dominated at cap=6.
- v5 scored 0.12873 against v3's 0.11354, but from 5 experiments against 24. This
  measured orientation and context, not model quality.

## What was rejected

- **The agent owning the episode stop.** Tested; it does not exercise it.
- **A brief with no `explore_journal`.** The brief is fixed-size by construction,
  so it can only answer anticipated questions. Episode 3 used
  `explore_experiment` after orienting — a follow-up no brief would have served.
- **`explore_journal` with no brief.** Tested; that is the table above.
- **Compaction of the message history instead.** Keeps one conversation alive and
  makes the run's memory a summary of a summary. The journal is already the
  record (0004); re-deriving state from it is exact, and a compacted transcript
  is not.

## Where it will live

- A driver loop in `harness/` — currently `scripts/run_house_v5_episodes.py`.
- `harness/instructions.py` — the dynamic brief (PydanticAI's
  `@agent.system_prompt` takes `ctx.deps`), and the `inspect_data`-first rule.
- `harness/journal.py` — an `episode` column on experiments, for the loop guard
  and for "did that episode accomplish anything".
