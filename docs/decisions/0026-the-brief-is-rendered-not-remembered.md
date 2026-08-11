# 0026 — The brief is rendered, never accumulated

- **Date:** 2026-08-11
- **Status:** accepted, implemented (`harness/brief.py`); **unmeasured on a live run**
- **Related:** [0024](0024-the-episode-stop-is-structural.md) (which decided the
  brief must exist), [0025](0025-episodes-buy-feasibility-not-cost.md) (which
  measured why it is the primary lever), [0018](0018-the-payload-is-a-budget.md)
  (whose subtractive rules this record has to extend), [0020](0020-the-sandbox-has-no-kernel.md)
  (the state is the run directory)

## The problem

An episode is a fresh agent on the same `run_dir`. It wakes up knowing nothing:
not the task, not the metric's direction, not that eleven experiments have
already happened and one of them is winning.

0024 established that it will find all of that by itself — the v5 resume prompt
never mentioned `explore_journal` and every resumed episode opened with it. 0025
measured what that costs: **26–50% of every episode, at both caps tested**, and
one episode spent seven `explore_journal` calls and trained nothing. Given more
room the agent orients *more*, so a bigger cap does not dilute the tax.

So the tool is not the fix. The agent re-derives its position because nothing
tells it, and the cheapest possible telling is a few hundred tokens of text that
costs no tool calls at all.

## The decision

**A `RUN STATE` block, rendered from the journal at the start of every episode,
carrying only what is O(1) in the length of the run.**

What goes in: task, target, metric and its direction; the split; counts of
experiments and tables; baseline and best (with the table the best was trained
on); the stop state — budget remaining, experiments since a real improvement, or
the stop reason when the run is over; tables built but never trained on, capped;
and one line saying that everything else is in the journal and should be queried.

Three properties, in the order they matter:

1. **Rendered, not accumulated.** It is a pure function of `journal.load(run_dir)`.
   It holds no state, caches nothing, and writes nothing, so it cannot drift from
   the run it describes and a resumed process sees exactly what the previous one
   would have. This is 0020's "the state is the run directory" applied to
   orientation.
2. **Flat forever.** Same size at experiment 150 as at experiment 2. This is the
   property that makes the sawtooth a sawtooth; the moment the brief grows with
   the run, an episode's fixed cost becomes a function of run length and episodes
   stop solving what they exist to solve.
3. **A starting point, not the record.** It deliberately does not answer "which
   encoding paid" — that is `explore_journal`'s job, and the brief says so
   explicitly.

`None` on a run where nothing has happened, rather than an empty-state message: a
run with no position has nothing to report, and this keeps a first episode
byte-identical to a run that never had episodes, which is what makes the two
comparable.

Wired as PydanticAI's **dynamic** system prompt (`@agent.system_prompt(dynamic=True)`,
which receives `ctx.deps`) so it is evaluated when the episode starts rather than
when the agent object is built.

## What was rejected

**Listing recent experiments.** The obvious addition, and the one that breaks
property 2. "The last five experiments" is bounded, but it is the thin end: five
becomes ten, and every reader of the brief who wants one more fact adds a line
that scales. The rule is easier to hold than the judgement, so the rule is
absolute — if a fact needs history, it is a query.

**Making the brief a tool** (`get_run_state()`). Costs a tool call, which is
exactly what is being eliminated, and a tool can be ignored. The system prompt
cannot.

**Injecting it into the user prompt instead.** Works, but it puts run state in
the episode driver, which means every caller — script, notebook, future
`harness/` loop — has to remember to do it, and one that forgets gets an agent
that silently re-derives everything. In the agent, it is impossible to forget.

**Leaving `SYSTEM_PROMPT` alone.** Its "`inspect_data` first, always" is a
run-level rule that was being obeyed *per episode*: v5 and v6 both opened every
episode with a redundant `inspect_data`, one wasted call each time, and 0024
called it the prompt's fault. It now says the first steps only apply to a run
that has not started, and points at the run state for the rest.

**Uncapping the idle-tables list.** It is the most useful non-numeric thing the
brief can say — an episode cut mid-idea leaves exactly that trace — but an agent
that builds ten tables and trains none would turn its own idle work into a
permanent context cost. Capped at three, and it says when it truncates.

## The tension with 0018

0018 gave the system prompt three subtractive rules: nothing derivable, nothing
tool-specific, nothing run-specific. The brief is **entirely run-specific**, and
it names `explore_journal`.

That is not a violation so much as a missing category. 0018's "run-specific goes
in the payload" was written when the only run-varying text was `note` and
`vs_source_basis`, both of which ride on a tool result — which presumes the agent
*made* the call that produced them. A resumed agent has made no calls yet, so
there is no payload to ride on, and the position on the board has nowhere else to
go.

The distinction to hold: **`SYSTEM_PROMPT` is the rules of the game and is
identical for every run; the brief is the position, and is identical for nobody.**
0018's rules still bind the static prompt in full.

## What it costs

**Good**

- Orientation that costs zero tool calls and cannot be skipped, forgotten, or
  answered wrongly.
- The metric's *direction* now reaches a resumed agent without a query. It is the
  fact most likely to be assumed rather than checked, and assuming it inverts
  every comparison silently.
- A resumed agent is told about work an earlier episode started and abandoned,
  which is the seam cost 0024 named and nothing else surfaces.

**Bad**

- **~170 tokens on every episode, whether or not the agent needed them.** Cheap
  against 2–8 orientation calls, but it is not free and it is paid every time.
- **Unmeasured.** The claim is that orientation drops from v6's 26/50/26/43%. That
  needs a v7 run to confirm, and if the share does not move, this record is a
  well-argued no-op. It should be read as a hypothesis with an implementation
  until then.
- **It can be wrong in a new way.** A brief that says "22 experiments left" when
  the policy changed mid-run would mislead precisely because it is trusted more
  than a tool result. Rendering from the journal makes this unlikely, not
  impossible.
- **Two things now say what the run is doing** — the brief and `note` on every
  training result — and they can disagree in wording while agreeing in fact. The
  duplication is deliberate (one is for a fresh agent, one is for a working one)
  and is the first place to look if a run starts contradicting itself.

## Where it lives

- `harness/brief.py` — `render_brief(session)`, `MAX_LISTED_TABLES`.
- `harness/agent.py` — the `run_state` dynamic system prompt.
- `harness/instructions.py` — the relaxed order-of-work paragraph.
- `tests/test_harness_brief.py` — flatness at 151 experiments, renderedness, the
  cap, and the stop state; `tests/test_harness_agent.py` for the wiring.
