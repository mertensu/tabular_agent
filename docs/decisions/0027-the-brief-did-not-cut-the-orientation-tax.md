# 0027 — The brief did not cut the orientation tax; orientation is a fraction, not a deficit

- **Date:** 2026-08-11
- **Status:** accepted (a measurement, not a design)
- **Supersedes:** the central claim of [0026](0026-the-brief-is-rendered-not-remembered.md) —
  that rendering the run's position would reduce the share of each episode spent
  re-deriving it. **It did not.** 0026's *implementation* stands and the brief stays
  (see "What we are keeping and why"); its motivating hypothesis does not.
- **Related:** [0024](0024-the-episode-stop-is-structural.md),
  [0025](0025-episodes-buy-feasibility-not-cost.md) (whose "given room, the agent
  orients more" observation this generalises),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md) (whose lesson had
  to be applied to this record's own numbers, twice)

## What was measured

v7 is a controlled repeat of v6: same dataset (house-prices), model
(`tensorx/deepseek-v4-flash-0731`), cap (16 tool calls), budget (30 experiments),
and prompts — including the resume prompt's now-false "earlier episodes did things
you have no memory of", kept verbatim so the brief was the only difference.

Orientation counted identically for both: every `explore_journal` call plus every
`inspect_data` after the run's first, over the episode's tool calls.

| | ep1 | ep2 | ep3 | ep4 | ep5 | **resumed mean** |
|---|---|---|---|---|---|---|
| v6, no brief | 8% | 32% | 50% | 32% | 50% | **41%** |
| v7, brief | 0% | 40% | 44% | 35% | — | **40%** |

**40% against 41%.** The brief did not move the number it was built to move.

Note these are not the percentages 0025 published for v6 (26/50/26/43). Those were
counted by hand; the table above recomputes both runs with one function
(`orientation_share` in `scripts/run_house_v7_brief.py`), which is the only way the
comparison means anything. 0025 is history and stays as written; where the two
disagree, this is the like-for-like count.

## The finding, which is not the one the experiment was designed to get

**The composition of orientation changed completely while its size did not.**

| | `inspect_data` per resumed episode | `explore_journal` per resumed episode |
|---|---|---|
| v6 | 3, 2, 3, 3 | 3, 8, 3, 4 |
| v7 | 1, 1, 2 | 7, 7, 4 |

The redundant `inspect_data` is gone — that is the `SYSTEM_PROMPT` fix landing
exactly as intended, and it is the one prediction that held. The agent then spent
the freed calls on `explore_journal` instead.

So orientation is **not an information deficit that can be paid off**. It behaves
like a roughly constant fraction of whatever episode it is given. 0025 saw one face
of this — a bigger cap buys proportionally more orientation, not less — and read it
as the agent orienting more *thoroughly*. v7 shows the same behaviour when the
information is handed over for free: the agent does not conclude it now knows where
it is and get to work. It re-derives anyway.

**The implication for the whole approach: more text will not fix this.** Every
remedy so far has been informational — a better docstring, a better prompt, now a
brief — and this is the third measurement in a row where an advisory change did not
alter behaviour. It is the same pattern as every steer since July, and 0024 already
named the general form: guidance for the cooperative case, *structure* for the rest.
Whatever fixes orientation will have to be structural.

## What we are keeping, and why it is not sunk cost

The brief stays, demoted from "the primary lever" to a cheap correctness aid. The
case for keeping it does not depend on the failed claim:

- **It carries the metric's direction to an agent that has no way to derive it.**
  `inspect_data` reported it to somebody else. An agent that assumes the direction
  inverts every comparison it makes, silently. This is a correctness argument, not
  an efficiency one, and it was true before the experiment and remains true after.
- **It killed the redundant `inspect_data`** — measured, ~2 calls per episode.
- **It surfaces abandoned tables**, which nothing else does.
- **It costs ~170 tokens** against episodes that spend 4-7 tool calls on orientation
  regardless. Even at zero behavioural benefit it is close to free.

What changes is the claim made *for* it. It is not the answer to 0025's tax, and the
next person reading 0026 must not size an episode budget on the assumption that it is.

## What this record explicitly does not claim

**v7's best `val_score` was 0.11977 against v6's 0.12525.** That is not evidence the
brief improves quality. Per 0022, `val_score` on this split has a standard error of
~0.010, so the gap is inside one standard error. Same trap 0025 declined to fall
into, same answer: unmeasured.

**v7 used 73 tool calls and 65 requests against v6's 98 and 90, for the same 18
experiments.** Tempting, and not attributable to the brief either. Episode 1 is
configured *identically* in both runs — a fresh run gets no brief in either version
— and it differed by **8 calls and 2 experiments**. That is the noise floor of a
single-run comparison, and it is large next to the 25-call total difference. Two runs
is not a sample.

The only claim this record makes is the pre-registered one: the orientation share did
not move.

## What we decided to do about it: nothing, for now

Ulf's call, and it came from checking whether the tax blocks anything before
designing a fix for it. It does not. Extrapolating v7 — 18 experiments in 4
episodes and 15.3 minutes:

| | v7 measured | at 150 experiments |
|---|---|---|
| episodes | 4 | ~33 |
| wall clock | 15.3 min | **~2.1 hours** |
| requests | 65 | ~540 |

The 150-experiment overnight run — the thing episodes exist to make possible —
fits in roughly two hours of an eight-hour window **with the tax included**.
Removing it entirely would save perhaps 40 minutes of that night.

Against that, every fix on the table restricts the agent's ability to look at its
own work: charging `explore_journal` against a budget, or capping it per episode,
undoes the "free diagnosis" property that 0020 and the step-5/6 work were built
around — charging for a look at a failure prices *understanding* the same as
*causing* an experiment. The third option (the episode driver runs fixed queries
and pastes the answers in) does not restrict anything, but v7 is direct evidence
against it: the brief already is information handed over for free, and the agent
queried anyway.

So the tax is **real, measured, and not currently on the critical path**. Recorded
here so it is not re-derived from scratch later. Revisit it if episode counts
start mattering — a run budgeted well past 150, or a model whose orientation share
is materially worse than this one's.

## What it costs

**Good**

- 0026 is now falsifiable-and-falsified rather than plausible-and-standing. The
  cost of finding out was one 15-minute run.
- The mechanism it does deliver (prompt scoping, metric direction, abandoned tables)
  is measured and small.
- The real problem has a sharper shape: orientation is a behavioural constant, not a
  gap in what the agent knows.

**Bad**

- **We still have no fix for the tax**, and now we know the cheap class of fix does
  not work. The remaining options are structural and all of them are more invasive:
  charging orientation against a budget, capping `explore_journal` per episode, or
  letting the episode driver make the first query itself and hand over the answer.
- **n=1 against n=1.** The pre-registered metric moved by one point (41 → 40), which
  is well inside what episode 1 shows to be run-to-run noise. This record is
  confident that the brief did not produce a *large* effect and cannot rule out a
  small one.
- The v7 script hardcoded 0025's hand-counted v6 numbers as its baseline and printed
  a misleading comparison on the way past. Fixed to the recomputed values, but the
  lesson is the usual one: a baseline you did not compute with the same code is not
  a baseline.

## Where it lives

- `scripts/run_house_v7_brief.py` — the experiment and `orientation_share`.
- `_runs/house_prices_v7/episodes.json` — per-episode calls, briefs, and transcripts.
- The brief itself is unchanged: `harness/brief.py`, wired in `harness/agent.py`.
